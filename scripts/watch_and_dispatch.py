#!/usr/bin/env python3
"""
watch_and_dispatch.py — wait for Jupyter allocations to land, and keep putting
one worker on each queued root until every root that has work also has a live
worker, or the deadline passes.

Why this exists: allocations on `alpha` sit PENDING for days (the v3 queues
were staged 2026-08-30 and the oldest request had waited five days by 08-31).
Nobody wants to poll `squeue` by hand across that, and a node that lands
unattended is a node burning its 3-day TimeLimit doing nothing.

Run it on the login node, detached:

    cd ~/LLM_research/thinking-gating
    setsid nohup .venv/bin/python scripts/watch_and_dispatch.py \
        --roots shared/dispatch/capture_qwen3v3_redo shared/dispatch/capture_nemotronv3_redo \
        > shared/logs/watch_dispatch.out 2>&1 &

    tail -f shared/logs/watch_dispatch.log      # what it is doing
    touch shared/dispatch/STOP_WATCH            # make it exit at the next poll

It only ever calls `gpu_dispatch.py sync-jupyter` (read-only), `gpu_dispatch.py
run` (dispatch) and `git fetch`. It never submits, cancels, or kills a SLURM job.
See agent.md, "what needs approval". Launching an allocation is still a
separate, deliberate act.

Each poll:

  1. **Which roots need a worker?** A root needs one when it has pending cells,
     or cells held by a *stale* claim (a dead worker; the new worker's startup
     GC returns them to pending), and no live worker. "Live" means a
     heartbeat in `<root>/claimed/<worker>/` younger than the GC's stale window
     (claim.DEFAULT_STALE_SECONDS). A dead worker's leftover claim does not count
     (2026-10 fix: it used to block the root forever). A worker this
     watcher dispatched within --startup-grace seconds also counts, since it
     may not have written its first heartbeat yet.
  2. If no root needs a worker, every root is either drained or served: exit 0.
  3. **Which allocations are free?** RUNNING `jupyter_*_<port>` allocations
     hosting no live worker. A worker's id is `<DISPATCH_NODE>_<pid>`, so live
     heartbeats in *any* watched root name their node. Recent dispatches by
     this watcher and fresh `running` records in gpu_jobs.json (within the
     startup grace) count too.
  4. **Clean tree**, checked only when there is something to dispatch:
     `git fetch` (best effort, with a timeout), then refuse if the code paths
     are dirty or HEAD differs from upstream. Without the fetch, "behind
     upstream" compared against whatever was last fetched and missed pushes.
  5. `sync-jupyter`, then dispatch needy roots onto free allocations, one
     worker per allocation. A failed dispatch is logged and retried next poll
     (it used to exit with 4).

The watcher used to dispatch once and exit, so when two allocations landed
hours apart the second root never got a worker (2026-10 fix).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.dispatch import claim  # noqa: E402

VENV_PY = REPO / ".venv" / "bin" / "python"
STOP_FILE = REPO / "shared" / "dispatch" / "STOP_WATCH"

# A dispatched worker gets this long to write its first heartbeat before the
# watcher stops trusting "I dispatched it" and treats the root as unserved.
DEFAULT_STARTUP_GRACE_S = 900

logger = logging.getLogger("watch")


def sh(cmd: list[str], timeout: int = 300) -> tuple[int, str]:
    """Run a command in the repo, returning (exit_code, combined output)."""
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s: {' '.join(cmd)}"
    except FileNotFoundError:
        # squeue/git missing means this is not the login node. Report it as a
        # failed command rather than dying, so the caller's own guard fires.
        return 127, f"command not found: {cmd[0]}"


# What a dispatched worker actually executes. Deliberately not the whole tree:
# paper/results/ and output/ churn constantly on the cluster (a full analysis
# run leaves ~60 files behind), and none of it changes what the worker runs.
# Gating on those would mean the watcher never fires.
CODE_PATHS = ["scripts", "tasks", "utils", "configs", "requirements.txt"]


def checkout_is_clean(fetch_timeout: int = 60) -> tuple[bool, str]:
    """Dispatching code that is not committed and pushed is not reproducible.

    Checks the code paths only (tracked *and* untracked, so a stray edited
    script cannot slip through), then HEAD against upstream, after a
    best-effort `git fetch` so that "behind upstream" means behind the remote
    as it is now. If the fetch fails (no network, auth), the comparison falls
    back to the last-fetched upstream and the detail says so.
    """
    rc, dirty = sh(["git", "status", "--porcelain", "--"] + CODE_PATHS)
    if rc != 0:
        return False, f"git status failed: {dirty}"
    if dirty:
        return False, "code paths are dirty:\n" + dirty
    frc, fout = sh(["git", "fetch", "--quiet"], timeout=fetch_timeout)
    fetch_note = "" if frc == 0 else f" (git fetch failed rc={frc}: {fout[:200]}; " \
                                     "compared against the last-fetched upstream)"
    if frc != 0:
        logger.warning("git fetch failed rc=%s: %s", frc, fout[:500])
    rc, head = sh(["git", "rev-parse", "HEAD"])
    rc2, upstream = sh(["git", "rev-parse", "@{u}"])
    if rc != 0 or rc2 != 0:
        return False, f"cannot resolve HEAD/upstream: {head} {upstream}"
    if head != upstream:
        return False, (f"HEAD {head[:8]} != upstream {upstream[:8]}: pull (or "
                       f"push) first{fetch_note}")
    return True, head[:8] + fetch_note


def queue_counts(root: Path) -> dict[str, int]:
    """Cell counts per state. `claimed` counts every claim, live or stale."""
    counts = {state: len(list((root / state).glob("*__*.json")))
              if (root / state).is_dir() else 0
              for state in ("pending", "done", "failed")}
    counts["claimed"] = sum(len(w["cells"]) for w in claim.live_workers(root))
    return counts


def root_liveness(root: Path, stale_seconds: float, now: float | None = None) -> dict:
    """Live workers, stale-held cells, and the nodes the live workers run on."""
    live, stale_cells = [], 0
    for w in claim.live_workers(root, stale_seconds, now=now):
        if w["alive"]:
            live.append(w["worker_id"])
        else:
            stale_cells += len(w["cells"])
    return {"live_workers": live, "stale_cells": stale_cells,
            "live_nodes": {claim.node_of_worker(w) for w in live}}


def _parse_ts(value: str) -> float | None:
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def recent_job_records(grace_s: float, now: float) -> list[dict]:
    """gpu_jobs.json `running` records younger than the startup grace.

    Older ones are not trusted. Nothing refreshes that file when a node dies,
    so a record can say `running` forever. Past the grace, the worker's own
    heartbeat is the only evidence of life.
    """
    manifest = REPO / "shared" / "gpu_jobs.json"
    if not manifest.exists():
        return []
    try:
        jobs = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("gpu_jobs.json is unreadable — ignoring it")
        return []
    out = []
    for job in jobs:
        if job.get("status") != "running":
            continue
        started = _parse_ts(job.get("started_at", ""))
        if started is None or now - started > grace_s:
            continue
        cmd = job.get("command", "")
        root = (cmd.split("--root", 1)[1].split()[0].rstrip("/")
                if "--root" in cmd else None)
        out.append({"node": job.get("node_name"), "root": root})
    return out


def running_allocations() -> list[dict]:
    """RUNNING jupyter_* allocations, keyed the way gpu_dispatch keys nodes."""
    rc, out = sh(["squeue", "--me", "--noheader", "-o", "%j|%T|%N"])
    if rc != 0:
        logger.warning("squeue failed: %s", out)
        return []
    allocs = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            continue
        name, state, node = (p.strip() for p in parts)
        if state == "RUNNING" and name.startswith("jupyter") and node:
            # gpu_dispatch keys nodes as "<hostname>-<port>" (what
            # sync-jupyter writes to nodes.json), and `run --node` wants that
            # key, not the bare hostname. The port is the trailing digits of
            # the job name once the allocation has renamed itself. Passing
            # the hostname alone gets "node 'alphagpuNN' not found in config"
            # -- the 2026-09-12 failure -- so an allocation still carrying
            # the PORT placeholder is skipped rather than guessed at.
            m = re.search(r"(\d+)$", name)
            if not m:
                logger.warning("allocation %s on %s has no port yet — skipping", name, node)
                continue
            allocs.append({"name": name, "node": f"{node}-{m.group(1)}"})
    return allocs


def dispatch(root: Path, node: str) -> bool:
    """One worker onto one node. Quoted per agent.md: `run` takes nargs='+'."""
    # The "--" is load-bearing: `run` takes the command as nargs="+", so
    # without it argparse reads the worker's own --root/--wait as unknown
    # gpu_dispatch options and the dispatch dies before submitting anything
    # (that is how the 2026-09-03 watcher fired on a landed node and did
    # nothing). "--" ends option parsing; everything after it is the command.
    cmd = [str(VENV_PY), "scripts/gpu_dispatch.py", "run",
           "--node", node,
           "--desc", f"capture worker ({root.name}) [watch_and_dispatch]",
           "--",
           ".venv/bin/python", "scripts/dispatch/worker.py",
           "--root", str(root.relative_to(REPO)), "--wait", "900"]
    logger.info("dispatching: %s", " ".join(cmd))
    rc, out = sh(cmd, timeout=600)
    logger.info("dispatch rc=%s\n%s", rc, out)
    return rc == 0


def plan(roots: list[Path], allocs: list[dict], recent: dict[str, tuple[str, float]],
         *, now: float, stale_seconds: float, grace_s: float) -> dict:
    """Decide, from the filesystem alone, which roots need a worker and which
    allocations are free. Pure apart from reading the queue roots.

    `recent` maps root (relative path) -> (node, dispatch time) for this
    watcher's own dispatches.
    """
    busy_nodes: set[str] = set()
    needy: list[Path] = []
    status: dict[str, str] = {}
    jobs = recent_job_records(grace_s, now)
    job_roots = {j["root"] for j in jobs if j["root"]}
    busy_nodes |= {j["node"] for j in jobs if j["node"]}

    for root in roots:
        rel = str(root.relative_to(REPO))
        info = root_liveness(root, stale_seconds, now=now)
        busy_nodes |= info["live_nodes"]
        counts = queue_counts(root)
        recent_entry = recent.get(rel)
        recently_dispatched = (recent_entry is not None
                               and now - recent_entry[1] < grace_s)
        if recently_dispatched:
            busy_nodes.add(recent_entry[0])
        has_work = counts["pending"] > 0 or info["stale_cells"] > 0
        if info["live_workers"]:
            status[rel] = f"served by {info['live_workers']}"
        elif not has_work:
            status[rel] = f"drained {counts}"
        elif recently_dispatched:
            status[rel] = (f"waiting for first heartbeat from the worker "
                           f"dispatched to {recent_entry[0]}")
        elif rel in job_roots:
            status[rel] = "fresh gpu_jobs.json record, waiting for a heartbeat"
        else:
            status[rel] = (f"NEEDS A WORKER: {counts['pending']} pending, "
                           f"{info['stale_cells']} held by stale claims")
            needy.append(root)

    pending_start = any("waiting for" in s or "fresh gpu_jobs" in s
                        for s in status.values())
    free = [a for a in allocs if a["node"] not in busy_nodes]
    return {"needy": needy, "free": free, "status": status,
            "settled": not needy and not pending_start}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Queue roots to serve, in priority order")
    ap.add_argument("--interval", type=int, default=120, help="Poll seconds")
    ap.add_argument("--max-hours", type=float, default=96.0,
                    help="Give up after this long (allocations expire too)")
    ap.add_argument("--stale-seconds", type=float, default=claim.DEFAULT_STALE_SECONDS,
                    help="Heartbeat age past which a worker counts as dead "
                         "(default: the GC's own threshold)")
    ap.add_argument("--startup-grace", type=float, default=DEFAULT_STARTUP_GRACE_S,
                    help="Seconds a just-dispatched worker has to write its "
                         "first heartbeat before its root counts as unserved")
    ap.add_argument("--log-file", default="shared/logs/watch_dispatch.log")
    args = ap.parse_args(argv)

    log_path = REPO / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for old in list(logger.handlers):   # main() may run more than once (tests)
        logger.removeHandler(old)
        old.close()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    roots = [REPO / r for r in args.roots]
    for root in roots:
        if not root.is_dir():
            logger.error("no such queue root: %s", root)
            return 2

    deadline = time.monotonic() + args.max_hours * 3600
    logger.info("watching %d root(s) for up to %.1fh (pid %d, poll %ds)",
                len(roots), args.max_hours, os.getpid(), args.interval)
    for root in roots:
        logger.info("  %s: %s", root.name, queue_counts(root))
    logger.info("stop with: touch %s", STOP_FILE)

    recent: dict[str, tuple[str, float]] = {}
    polls = 0
    last_status: dict[str, str] = {}
    while time.monotonic() < deadline:
        if STOP_FILE.exists():
            logger.info("STOP_WATCH present — exiting")
            STOP_FILE.unlink(missing_ok=True)
            return 0

        polls += 1
        now = time.time()
        allocs = running_allocations()
        p = plan(roots, allocs, recent, now=now,
                 stale_seconds=args.stale_seconds, grace_s=args.startup_grace)
        if p["status"] != last_status:
            for rel, s in p["status"].items():
                logger.info("poll %d: %s — %s", polls, rel, s)
            last_status = p["status"]

        if p["settled"]:
            logger.info("every root is drained or has a live worker — watcher done")
            logger.info("check with: python scripts/dispatch/queue.py status --root <root>")
            return 0

        if not p["needy"] or not p["free"]:
            if p["needy"] and polls % 15 == 1:   # ~ every half hour at the default
                logger.info("poll %d: %d root(s) need a worker, no free allocation "
                            "(%d RUNNING)", polls, len(p["needy"]), len(allocs))
            time.sleep(args.interval)
            continue

        logger.info("poll %d: %d root(s) need a worker, free allocation(s): %s",
                    polls, len(p["needy"]), [a["node"] for a in p["free"]])

        ok, detail = checkout_is_clean()
        if not ok:
            # Keep waiting rather than exiting. This watcher is meant to sit
            # through a multi-day queue, and the checkout going briefly behind
            # upstream is routine -- any push from the laptop does it. Dying on
            # that would mean the one process whose whole job is to be present
            # when a node lands is reliably absent right after a push. A `git
            # pull` on the cluster fixes it and the next poll proceeds.
            logger.error("not dispatching this poll — %s", detail)
            logger.error("fix the checkout (git pull); the watcher keeps waiting")
            time.sleep(args.interval)
            continue
        logger.info("checkout clean at %s", detail)

        # nodes.json goes stale as allocations come and go; dispatch only
        # reaches nodes registered there.
        rc, out = sh([str(VENV_PY), "scripts/gpu_dispatch.py", "sync-jupyter"])
        logger.info("sync-jupyter rc=%s\n%s", rc, out)
        if rc != 0:
            logger.error("sync-jupyter failed — waiting rather than guessing")
            time.sleep(args.interval)
            continue

        for root, alloc in zip(p["needy"], p["free"]):    # one worker per node
            rel = str(root.relative_to(REPO))
            if dispatch(root, alloc["node"]):
                recent[rel] = (alloc["node"], time.time())
                logger.info("dispatched %s onto %s", root.name, alloc["node"])
            else:
                logger.error("dispatch failed for %s on %s — will retry next poll",
                             root.name, alloc["node"])
        time.sleep(args.interval)

    logger.info("deadline reached after %d polls; last state: %s", polls, last_status)
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
