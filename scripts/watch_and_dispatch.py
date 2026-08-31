#!/usr/bin/env python3
"""
watch_and_dispatch.py — wait for a Jupyter allocation to land, then put one
worker on each queued capture root. One-shot: it dispatches once and exits.

Why this exists: allocations on `alpha` sit PENDING for days (the v3 queues
were staged 2026-08-30 and the oldest request had waited five days by 08-31).
Nobody wants to poll `squeue` by hand across that, and a node that lands
unattended is a node burning its 3-day TimeLimit doing nothing.

Run it on the login node, detached:

    cd ~/LLM_research/thinking-gating
    setsid nohup .venv/bin/python scripts/watch_and_dispatch.py \
        --roots shared/dispatch/capture_qwen3v3 shared/dispatch/capture_nemotronv3 \
        > shared/logs/watch_dispatch.out 2>&1 &

    tail -f shared/logs/watch_dispatch.log      # what it is doing
    touch shared/dispatch/STOP_WATCH            # make it exit at the next poll

It only ever calls `gpu_dispatch.py sync-jupyter` (read-only) and
`gpu_dispatch.py run` (dispatch). It never submits, cancels, or kills a SLURM
job — see agent.md, "what needs approval". Launching an allocation is still a
separate, deliberate act.

Guards, in the order they are checked:

  * **Deadline.** Stops after --max-hours no matter what. A watcher that
    outlives its reason is a liability, not a convenience.
  * **Clean tree.** Refuses to dispatch if the checkout is dirty or behind its
    upstream. "Commit before dispatching" is a repo rule, and a watcher firing
    days later is exactly when uncommitted code gets run and forgotten.
  * **Empty queue.** A root with no pending cells is skipped, so this cannot
    resurrect retired work.
  * **Already busy.** A root with claimed cells, or with a live worker in
    gpu_jobs.json, is skipped. Two workers on one root is survivable (claims
    are atomic) but it double-books a GPU.
  * **One worker per node**, and never more workers than roots.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV_PY = REPO / ".venv" / "bin" / "python"
STOP_FILE = REPO / "shared" / "dispatch" / "STOP_WATCH"

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


def checkout_is_clean() -> tuple[bool, str]:
    """Dispatching code that is not committed and pushed is not reproducible."""
    rc, dirty = sh(["git", "status", "--porcelain"])
    if rc != 0:
        return False, f"git status failed: {dirty}"
    if dirty:
        return False, "working tree is dirty:\n" + dirty
    rc, head = sh(["git", "rev-parse", "HEAD"])
    rc2, upstream = sh(["git", "rev-parse", "@{u}"])
    if rc == 0 and rc2 == 0 and head != upstream:
        return False, f"HEAD {head[:8]} != upstream {upstream[:8]} — pull first"
    return True, head[:8]


def queue_counts(root: Path) -> dict[str, int]:
    return {state: len(list((root / state).glob("*__*.json")))
            if (root / state).is_dir() else 0
            for state in ("pending", "claimed", "done", "failed")}


def live_worker_roots() -> set[str]:
    """Roots that a currently-running dispatched worker is already serving."""
    manifest = REPO / "shared" / "gpu_jobs.json"
    if not manifest.exists():
        return set()
    try:
        jobs = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        logger.warning("gpu_jobs.json is unparseable — assuming no live workers")
        return set()
    busy = set()
    for job in jobs:
        if job.get("status") != "running":
            continue
        cmd = job.get("command", "")
        if "--root" in cmd:
            busy.add(cmd.split("--root", 1)[1].split()[0].rstrip("/"))
    return busy


def running_allocations() -> list[dict]:
    """RUNNING jupyter_* allocations, as gpu_dispatch's sync sees them."""
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
            allocs.append({"name": name, "node": node})
    return allocs


def dispatch(root: Path, node: str) -> bool:
    """One worker onto one node. Quoted per agent.md: `run` takes nargs='+'."""
    cmd = [str(VENV_PY), "scripts/gpu_dispatch.py", "run",
           "--node", node,
           "--desc", f"v3 capture worker ({root.name}) [watch_and_dispatch]",
           ".venv/bin/python", "scripts/dispatch/worker.py",
           "--root", str(root.relative_to(REPO)), "--wait", "900"]
    logger.info("dispatching: %s", " ".join(cmd))
    rc, out = sh(cmd, timeout=600)
    logger.info("dispatch rc=%s\n%s", rc, out)
    return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Queue roots to serve, in priority order")
    ap.add_argument("--interval", type=int, default=120, help="Poll seconds")
    ap.add_argument("--max-hours", type=float, default=96.0,
                    help="Give up after this long (allocations expire too)")
    ap.add_argument("--log-file", default="shared/logs/watch_dispatch.log")
    args = ap.parse_args()

    log_path = REPO / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )

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

    polls = 0
    while time.monotonic() < deadline:
        if STOP_FILE.exists():
            logger.info("STOP_WATCH present — exiting without dispatching")
            STOP_FILE.unlink(missing_ok=True)
            return 0

        polls += 1
        allocs = running_allocations()
        if not allocs:
            if polls % 15 == 1:      # ~ every half hour at the default interval
                logger.info("poll %d: no RUNNING allocation yet", polls)
            time.sleep(args.interval)
            continue

        logger.info("poll %d: %d RUNNING allocation(s): %s",
                    polls, len(allocs), [a["node"] for a in allocs])

        ok, detail = checkout_is_clean()
        if not ok:
            logger.error("refusing to dispatch — %s", detail)
            logger.error("fix the checkout and restart the watcher; not retrying")
            return 3
        logger.info("checkout clean at %s", detail)

        # nodes.json goes stale as allocations come and go; dispatch only
        # reaches nodes registered there.
        rc, out = sh([str(VENV_PY), "scripts/gpu_dispatch.py", "sync-jupyter"])
        logger.info("sync-jupyter rc=%s\n%s", rc, out)
        if rc != 0:
            logger.error("sync-jupyter failed — waiting rather than guessing")
            time.sleep(args.interval)
            continue

        busy = live_worker_roots()
        servable = []
        for root in roots:
            counts = queue_counts(root)
            rel = str(root.relative_to(REPO))
            if counts["pending"] == 0:
                logger.info("skip %s: nothing pending (%s)", root.name, counts)
            elif counts["claimed"]:
                logger.info("skip %s: %d cell(s) already claimed",
                            root.name, counts["claimed"])
            elif rel in busy:
                logger.info("skip %s: a live worker is already on it", root.name)
            else:
                servable.append(root)

        if not servable:
            logger.info("nothing to serve — exiting")
            return 0

        dispatched = 0
        for root, alloc in zip(servable, allocs):    # one worker per node
            if dispatch(root, alloc["node"]):
                dispatched += 1
            else:
                logger.error("dispatch failed for %s on %s — leaving it queued",
                             root.name, alloc["node"])

        logger.info("dispatched %d/%d root(s); watcher done",
                    dispatched, len(servable))
        if dispatched < len(servable):
            logger.warning("still queued: %s",
                           [r.name for r in servable[dispatched:]])
        logger.info("check with: python scripts/dispatch/queue.py status --root <root>")
        return 0 if dispatched else 4

    logger.info("deadline reached after %d polls without an allocation", polls)
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
