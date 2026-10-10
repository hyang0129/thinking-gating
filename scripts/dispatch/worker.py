#!/usr/bin/env python3
"""The generic dispatch worker — one script, every kind of work.

Claims cells from a queue and runs them until the queue drains. It knows
nothing about probes, captures, or datasets: a cell says what to run, and the
worker runs it, watches it, and records what happened. Adding a new experiment
means writing a manifest (see cells.py), never editing this file.

Run it (on a GPU node, via gpu_dispatch.py):

    python scripts/gpu_dispatch.py run --desc "probe sweep worker" \
        .venv/bin/python scripts/dispatch/worker.py --root shared/dispatch/gsm8k_probe

Locally, to drain a CPU queue:

    python scripts/dispatch/worker.py --root shared/dispatch/labels

Behavior worth knowing:

  * **Isolation.** Every cell runs in its own subprocess with its own process
    group. A cell that segfaults, OOMs, or calls sys.exit() kills itself, not
    the worker, so the failure gets recorded instead of losing the queue.
  * **Resume.** A cell whose `output_check` paths already exist is marked done
    without running. Re-launching a worker over a partly-finished queue is the
    normal way to recover.
  * **Phantom completions.** Exit code 0 with missing `output_check` paths is
    recorded as a FAILURE. Scripts that swallow an internal error and exit 0
    are common enough that trusting the exit code alone loses real work.
  * **Retries.** A cell with `max_attempts > 1` goes back to pending on
    failure, with `attempts` incremented, so another worker (or node) can try.
  * **Shutdown.** On SIGTERM/SIGINT the current cell's process group is killed
    and its cell is released back to pending immediately, rather than sitting
    in claimed/ until the stale-claim GC notices.
  * **Deterministic failures are not retried.** If a failed attempt left a
    terminal marker (`terminal_markers`, default `TRUNCATION_FAILURE.*.json`)
    in an `output_check` directory, the cell is failed for good whatever its
    `max_attempts`. A retry at the same settings would reproduce the failure
    and burn the GPU time again. The same applies to a cell re-queued by an
    older worker whose previous attempt left such a marker; it is failed
    without running.
  * **Lost claims.** If this worker's claim is reclaimed while it runs (its
    heartbeat looked stale to someone's GC) and the cell turns up elsewhere,
    the worker kills its copy, leaves the cell to its new owner, and exits
    with code 3. It never writes to a cell it no longer holds. The cell may
    have run twice, which is why cells must be idempotent (see claim.py).
  * **Provenance.** Each attempt records the repo commit (`git_commit`,
    `git_dirty`) in its result and log header. A `git pull` between cells
    changes the code the next cell runs, and this is how to tell.

Stdlib only — the worker must start on any interpreter, even if the project's
own dependencies are broken.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch import claim  # noqa: E402
from scripts.dispatch import cells as cells_mod  # noqa: E402

# Keep BLAS from grabbing every core: several workers may share a node, and
# torch DataLoader workers fan out on top of this.
_THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
    "NUMEXPR_NUM_THREADS": "8",
}

_GRACE_SECONDS = 10  # SIGTERM → SIGKILL window for a timed-out or cancelled cell
_ERROR_TAIL_LINES = 60
# A marker counts as written by an attempt if its mtime is no earlier than the
# attempt's start minus this. Covers clock skew between the worker host and
# the filesystem server; markers from an earlier capture are days older.
_MARKER_CLOCK_SLACK_S = 120
EXIT_CLAIM_LOST = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(value: object) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def git_state(project_root: Path) -> dict:
    """Best-effort `{"git_commit": sha|None, "git_dirty": bool|None}`.

    Never raises and never blocks for long: a worker must run in a tree that
    is not a git checkout, or on a host without git.
    """
    def _git(*args: str) -> Optional[str]:
        try:
            p = subprocess.run(["git", "-C", str(project_root), *args],
                               capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return None
        return p.stdout.strip() if p.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"git_commit": None, "git_dirty": None}
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit, "git_dirty": None if status is None else bool(status)}


def _log(msg: str) -> None:
    print(f"[{_now_iso()}] {msg}", flush=True)


def _tail(path: Path, n: int = _ERROR_TAIL_LINES) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = Path(args.root)
        if not self.root.is_absolute():
            self.root = _PROJECT_ROOT / self.root
        self.project_root = Path(args.project_root or _PROJECT_ROOT)
        self.worker_id = args.worker_id or claim.worker_id_for()
        self.python = args.python or sys.executable
        self.skip_existing = not args.no_skip_existing
        self.stale_seconds = args.stale_seconds
        self.heartbeat_interval = args.heartbeat_interval
        self.poll_interval = args.poll_interval
        self.wait_seconds = args.wait
        self.max_cells = args.max_cells
        self.dry_run = args.dry_run
        self.extra_env = dict(pair.split("=", 1) for pair in (args.env or []))

        self._stop = threading.Event()
        self._current_cell: Optional[Path] = None
        self._current_proc: Optional[subprocess.Popen] = None
        # Set only while a cell subprocess runs; the heartbeat thread watches
        # it to notice a claim that was reclaimed out from under us.
        self._running_claim: Optional[Path] = None
        self._claim_lock = threading.Lock()
        self._cancelled = False
        self._claim_lost = False
        self.counts = {"done": 0, "skipped": 0, "failed": 0, "retried": 0, "lost": 0}

    # -- lifecycle ---------------------------------------------------------

    def install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            _log(f"received signal {signum} — finishing up")
            self._stop.set()
            self._cancelled = True
            self._kill_current("worker shutdown")
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, handler)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            try:
                claim.touch_heartbeat(self.root, self.worker_id)
            except OSError as exc:
                _log(f"heartbeat failed (continuing): {exc}")
            self._check_claim_while_running()

    def _claim_moved_elsewhere(self, claim_path: Path) -> bool:
        """True only on positive evidence: our claim file is gone AND the cell
        is now somewhere else (pending, another worker, done, failed).

        Requiring both keeps a transient filesystem glitch from killing a
        healthy multi-hour cell.
        """
        if claim_path.exists():
            return False
        try:
            found = claim.find_cell(self.root, claim.cell_id_from_path(claim_path))
        except OSError:
            return False
        return found is not None and found[1] != claim_path

    def _check_claim_while_running(self) -> None:
        # Under the lock, so the main thread cannot clear _running_claim and
        # start its own finishing move (which also makes the claim file
        # vanish) halfway through this check.
        with self._claim_lock:
            claim_path = self._running_claim
            if claim_path is None or self._claim_lost:
                return
            if not self._claim_moved_elsewhere(claim_path):
                return
            self._claim_lost = True
            self._stop.set()
        _log(f"LOST CLAIM on {claim_path.name}: it was reclaimed (this "
             "worker's heartbeat looked stale) and now belongs elsewhere. "
             "Killing this copy; the new owner's run stands.")
        self._kill_current("claim lost to another worker")

    def _kill_current(self, why: str) -> None:
        proc = self._current_proc
        if proc is None or proc.poll() is not None:
            return
        _log(f"killing cell process group ({why})")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.terminate()
        try:
            proc.wait(timeout=_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()

    # -- cell execution ----------------------------------------------------

    def _environment(self, cell: dict) -> dict:
        env = dict(os.environ)
        for key, value in _THREAD_DEFAULTS.items():
            env.setdefault(key, value)
        env.update({str(k): str(v) for k, v in (cell.get("env") or {}).items()})
        env.update(self.extra_env)
        # Let cell code identify itself (log naming, checkpoint tagging).
        env.update({
            "DISPATCH_CELL_ID": cell["cell_id"],
            "DISPATCH_ROOT": str(self.root),
            "DISPATCH_WORKER_ID": self.worker_id,
            "PYTHONUNBUFFERED": "1",
        })
        return env

    def _log_file_for(self, cell_id: str, attempt: int) -> Path:
        """`<id>.attempt<N>.log`, or `.attempt<N>.r<K>.log` if that exists.

        Attempt numbers repeat after a release or a GC reclaim, and opening
        the existing log with "w" would wipe the earlier run's output, or
        truncate a log that a still-running original worker is writing to.
        """
        log_file = claim.log_path(self.root, cell_id, attempt)
        k = 1
        while log_file.exists():
            log_file = log_file.with_name(f"{cell_id}.attempt{attempt}.r{k}.log")
            k += 1
        return log_file

    def _run_cell(self, cell: dict, attempt: int,
                  cell_path: Optional[Path] = None) -> dict:
        """Execute one cell in a subprocess. Returns the result record."""
        cell_id = cell["cell_id"]
        cwd = cells_mod.resolve_cwd(cell, self.project_root)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        log_file = self._log_file_for(cell_id, attempt)
        result_file = claim.result_path(self.root, cell_id) if cell["kind"] == "call" else None

        argv = cells_mod.build_command(
            cell,
            python=self.python,
            project_root=self.project_root,
            scratch_dir=self.root / "scratch",
            result_file=result_file,
        )

        record = {
            "worker_id": self.worker_id,
            "host": socket.gethostname(),
            "attempt": attempt,
            "command": " ".join(shlex.quote(a) for a in argv),
            "cwd": str(cwd),
            "log": str(log_file.relative_to(self.root)),
            "started_at": _now_iso(),
            # The code the cell subprocess loads is whatever is on disk now,
            # not what this worker started with.
            **git_state(self.project_root),
        }

        cwd.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        timeout = cell.get("timeout_s")
        timed_out = False

        # Last cheap check before spending GPU time: is the claim still ours?
        if cell_path is not None and self._claim_moved_elsewhere(cell_path):
            self._claim_lost = True
            record.update(status="lost", exit_code=None, duration_s=0.0,
                          ended_at=_now_iso(),
                          error="claim was reclaimed before the cell started")
            return record

        dirty = record["git_dirty"]
        with open(log_file, "w", encoding="utf-8") as handle:
            handle.write(f"# cell     {cell_id}\n")
            handle.write(f"# attempt  {attempt}\n")
            handle.write(f"# worker   {self.worker_id} on {record['host']}\n")
            handle.write(f"# commit   {record['git_commit'] or 'unknown'}"
                         f"{' (dirty)' if dirty else ''}\n")
            handle.write(f"# command  {record['command']}\n")
            handle.write(f"# cwd      {cwd}\n")
            handle.write(f"# started  {record['started_at']}\n\n")
            handle.flush()
            # start_new_session: the cell gets its own process group, so a
            # timeout or shutdown kills its children too instead of orphaning
            # them onto the GPU.
            self._current_proc = subprocess.Popen(
                argv, cwd=str(cwd), env=self._environment(cell),
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
            self._running_claim = cell_path
            try:
                exit_code = self._current_proc.wait(
                    timeout=float(timeout) if timeout else None
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill_current(f"timeout after {timeout}s")
                exit_code = self._current_proc.poll()
            finally:
                with self._claim_lock:
                    self._running_claim = None
                proc, self._current_proc = self._current_proc, None
                if proc.poll() is None:
                    exit_code = proc.wait()

        record.update({
            "exit_code": exit_code,
            "duration_s": round(time.monotonic() - started, 2),
            "ended_at": _now_iso(),
        })

        expected = cells_mod.output_paths(cell, self.project_root)
        missing = [str(p) for p in expected if not p.exists()]

        if self._claim_lost:
            record.update(status="lost",
                          error="claim was reclaimed by another worker mid-run")
        elif timed_out:
            record.update(status="timeout",
                          error=f"exceeded timeout_s={timeout}\n\n{_tail(log_file)}")
        elif self._cancelled:
            record.update(status="cancelled", error="worker shut down mid-cell")
        elif exit_code != 0:
            record.update(status="failed", error=_tail(log_file))
        elif missing:
            # The phantom-completion guard: a script can exit 0 having written
            # nothing. Without this the cell would be filed as done and the
            # gap only surface much later, in analysis.
            record.update(
                status="failed",
                error=("exit code 0 but output_check paths are missing:\n  "
                       + "\n  ".join(missing) + f"\n\n{_tail(log_file)}"),
                missing_outputs=missing,
            )
        else:
            record["status"] = "ok"
            if expected:
                record["outputs"] = [str(p) for p in expected]
            if result_file and result_file.exists():
                record["result_file"] = str(result_file.relative_to(self.root))

        return record

    # -- main loop ---------------------------------------------------------

    def run(self) -> int:
        claim.init_dispatch_dirs(self.root)
        _log(f"worker {self.worker_id} starting — root={self.root}")
        _log(f"python={self.python}  project_root={self.project_root}")
        state = git_state(self.project_root)
        _log(f"commit={state['git_commit'] or 'unknown'}"
             f"{' (dirty)' if state['git_dirty'] else ''}")

        if self.dry_run:
            return self._dry_run()

        reclaimed = claim.gc_stale_claims(self.root, self.stale_seconds)
        if reclaimed:
            _log(f"reclaimed {len(reclaimed)} stale cell(s) from dead workers")

        claim.touch_heartbeat(self.root, self.worker_id)
        heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat.start()

        idle_since: Optional[float] = None
        processed = 0
        try:
            while not self._stop.is_set():
                if self.max_cells and processed >= self.max_cells:
                    _log(f"reached --max-cells={self.max_cells}")
                    break

                cell_path = claim.claim_next_cell(self.root, self.worker_id)
                if cell_path is None:
                    if not self.wait_seconds:
                        _log("queue empty — exiting")
                        break
                    idle_since = idle_since or time.monotonic()
                    if time.monotonic() - idle_since >= self.wait_seconds:
                        _log(f"queue empty for {self.wait_seconds}s — exiting")
                        break
                    self._stop.wait(self.poll_interval)
                    continue

                idle_since = None
                self._current_cell = cell_path
                processed += 1
                self._process(cell_path)
                self._current_cell = None
        finally:
            self._stop.set()
            self._release_current()
            self._cleanup_worker_dir()

        _log(
            f"worker {self.worker_id} finished — "
            f"{self.counts['done']} done, {self.counts['skipped']} skipped, "
            f"{self.counts['retried']} re-queued, {self.counts['failed']} failed"
        )
        if self._claim_lost:
            _log("exiting after a lost claim (exit 3); the queue is intact")
            return EXIT_CLAIM_LOST
        return 1 if self.counts["failed"] else 0

    def _dry_run(self) -> int:
        """Print what would run, touching nothing.

        Deliberately does not claim: a preview that consumed the queue — or
        that raced a live worker for cells — would be worse than no preview.
        """
        pending = list(claim.iter_cells(self.root, "pending"))
        if not pending:
            _log("queue is empty — nothing to preview")
            return 0
        _log(f"[dry-run] {len(pending)} pending cell(s), in claim order:")
        for _, cell in pending:
            try:
                cells_mod.validate_cell(cell)
                argv = cells_mod.build_command(
                    cell, python=self.python, project_root=self.project_root,
                    scratch_dir=self.root / "scratch",
                    result_file=(claim.result_path(self.root, cell["cell_id"])
                                 if cell.get("kind") == "call" else None),
                )
                rendered = " ".join(shlex.quote(a) for a in argv)
            except Exception as exc:  # a bad cell should show up here, not at run time
                rendered = f"INVALID: {exc}"
            print(f"  {cell['cell_id']}")
            print(f"      {rendered}")
            for path in cells_mod.output_paths(cell, self.project_root):
                mark = "exists (would skip)" if path.exists() else "missing"
                print(f"      check: {path}  [{mark}]")
        _log("[dry-run] nothing claimed, nothing run")
        return 0

    # -- terminal (do-not-retry) failures ------------------------------------

    def _terminal_markers(self, cell: dict, since: Optional[float]) -> list[Path]:
        """Marker files that say this cell's failure is deterministic.

        Looked for in the directories of the cell's `output_check` paths. A
        marker named `<PREFIX>.<tag>.json` applies only to cells whose output
        file names contain `.<tag>.`, so shard01's marker does not condemn
        shard00. Only markers modified at or after `since` count, so a marker
        left by an older capture at a different budget is ignored.
        """
        patterns = cell.get("terminal_markers")
        if patterns is None:
            patterns = list(cells_mod.TERMINAL_MARKERS_DEFAULT)
        expected = cells_mod.output_paths(cell, self.project_root)
        if not patterns or not expected:
            return []
        names = [p.name for p in expected]
        found: list[Path] = []
        for directory in dict.fromkeys(p.parent for p in expected):
            if not directory.is_dir():
                continue
            for pattern in patterns:
                for marker in sorted(directory.glob(pattern)):
                    parts = marker.name.split(".")
                    tag = ".".join(parts[1:-1])
                    if tag and not any(f".{tag}." in n for n in names):
                        continue
                    try:
                        mtime = marker.stat().st_mtime
                    except OSError:
                        continue
                    if since is not None and mtime < since - _MARKER_CLOCK_SLACK_S:
                        continue
                    found.append(marker)
        return found

    @staticmethod
    def _terminal_message(markers: list[Path]) -> str:
        lines = ["TERMINAL FAILURE, not re-queued: the attempt left a marker saying "
                 "the failure is deterministic, so a retry at the same settings "
                 "would reproduce it."]
        for marker in markers:
            lines.append(f"  marker: {marker}")
            try:
                body = marker.read_text(encoding="utf-8", errors="replace").strip()
                lines.append("    " + body[:800].replace("\n", "\n    "))
            except OSError:
                pass
        lines.append("Fix the cause (e.g. raise --max-response-len) in a new manifest, "
                     "move the output dir aside, then `queue.py retry --cell <id>` "
                     "if this cell is still the right one.")
        return "\n".join(lines)

    # -- per-cell flow ---------------------------------------------------------

    def _finish(self, move, cell_path: Path, record: dict) -> bool:
        """Run one claim transition. False if the claim turned out to be lost."""
        try:
            move(self.root, cell_path, record)
            return True
        except claim.ClaimLost as exc:
            self._on_lost(cell_path, str(exc))
            return False

    def _on_lost(self, cell_path: Path, why: str) -> None:
        self._claim_lost = True
        self._stop.set()
        self.counts["lost"] += 1
        _log(f"{cell_path.name}: claim lost, leaving the cell to its new owner "
             f"and stopping this worker ({why})")

    def _process(self, cell_path: Path) -> None:
        try:
            cell = claim.load_cell(cell_path)
            cells_mod.validate_cell(cell)
        except FileNotFoundError as exc:
            self._on_lost(cell_path, str(exc))
            return
        except Exception as exc:  # malformed cell — fail it, keep the worker up
            _log(f"invalid cell {cell_path.name}: {exc}")
            if self._finish(claim.fail_cell, cell_path, {
                "status": "failed", "worker_id": self.worker_id,
                "error": f"invalid cell: {exc}", "ended_at": _now_iso(),
            }):
                self.counts["failed"] += 1
            return

        cell_id = cell["cell_id"]
        attempt = int(cell.get("attempts", 0)) + 1
        _log(f"claimed {cell_id} (attempt {attempt}) — {cells_mod.describe_command(cell)}")

        expected = cells_mod.output_paths(cell, self.project_root)
        if self.skip_existing and expected and all(p.exists() for p in expected):
            _log(f"{cell_id}: outputs already present — skipping")
            if self._finish(claim.complete_cell, cell_path, {
                "status": "skipped", "worker_id": self.worker_id,
                "attempt": attempt, "ended_at": _now_iso(),
                "note": "output_check satisfied before run",
                "outputs": [str(p) for p in expected],
            }):
                self.counts["skipped"] += 1
            return

        # A cell re-queued after a failure (possibly by an older worker that
        # did not know about markers) whose previous attempt left a terminal
        # marker: fail it now rather than spend another run reproducing it.
        previous = cell.get("result") or {}
        if previous.get("status") in ("failed", "timeout"):
            prev_start = _parse_iso(previous.get("started_at"))
            markers = self._terminal_markers(cell, prev_start) if prev_start else []
            if markers:
                error = self._terminal_message(markers)
                _log(f"{cell_id}: previous attempt left a terminal marker — "
                     f"failing without running ({markers[0].name})")
                if self._finish(claim.fail_cell, cell_path, {
                    "status": "failed", "terminal": True,
                    "worker_id": self.worker_id, "attempt": attempt,
                    "ended_at": _now_iso(), "error": error,
                    "terminal_markers": [str(m) for m in markers],
                    "previous_result": previous,
                }):
                    self.counts["failed"] += 1
                return

        attempt_wall_start = time.time()
        record = self._run_cell(cell, attempt, cell_path)

        if record["status"] == "lost":
            self._on_lost(cell_path, record.get("error", ""))
            return

        if record["status"] in ("ok", "skipped"):
            if self._finish(claim.complete_cell, cell_path, record):
                self.counts["done" if record["status"] == "ok" else "skipped"] += 1
                _log(f"{cell_id}: {record['status']} in {record.get('duration_s', 0)}s")
            return

        if record["status"] == "cancelled":
            if self._finish(claim.release_cell, cell_path, record):
                _log(f"{cell_id}: released back to pending (worker shutting down)")
            return

        max_attempts = int(cell.get("max_attempts", 1))
        markers = self._terminal_markers(cell, attempt_wall_start)
        if markers:
            record.update(terminal=True,
                          terminal_markers=[str(m) for m in markers],
                          error=self._terminal_message(markers) + "\n\n"
                                + (record.get("error") or ""))
            if self._finish(claim.fail_cell, cell_path, record):
                self.counts["failed"] += 1
                _log(f"{cell_id}: {record['status']} with terminal marker "
                     f"{markers[0].name}: failed for good, not re-queued "
                     f"(attempt {attempt}/{max_attempts})")
        elif attempt < max_attempts:
            if self._finish(claim.retry_cell, cell_path, record):
                self.counts["retried"] += 1
                _log(f"{cell_id}: {record['status']} — re-queued "
                     f"(attempt {attempt}/{max_attempts})")
        else:
            if self._finish(claim.fail_cell, cell_path, record):
                self.counts["failed"] += 1
                _log(f"{cell_id}: {record['status']} after {attempt} attempt(s) — "
                     f"see {record.get('log')}")

    def _release_current(self) -> None:
        """Put a half-run cell back on the queue when shutting down."""
        if (self._claim_lost or self._current_cell is None
                or not self._current_cell.exists()):
            return
        try:
            claim.release_cell(self.root, self._current_cell, {
                "status": "cancelled", "worker_id": self.worker_id,
                "ended_at": _now_iso(), "error": "worker shut down mid-cell",
            })
            _log(f"released {self._current_cell.name} back to pending")
        except OSError as exc:
            _log(f"could not release {self._current_cell.name}: {exc} "
                 "(the stale-claim GC will reclaim it)")

    def _cleanup_worker_dir(self) -> None:
        worker_dir = self.root / "claimed" / self.worker_id
        heartbeat = worker_dir / "heartbeat"
        try:
            if heartbeat.exists():
                heartbeat.unlink()
            worker_dir.rmdir()
        except OSError:
            pass  # still holds cells, or already gone — GC will handle it


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generic dispatch worker: claim cells and run them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", required=True,
                   help="Dispatch root (e.g. shared/dispatch/gsm8k_probe)")
    p.add_argument("--worker-id", default=None,
                   help="Override the generated <node>_<pid> worker id")
    p.add_argument("--python", default=None,
                   help="Interpreter for cells (default: this worker's own)")
    p.add_argument("--project-root", default=None,
                   help="Repo root that cells resolve paths against")
    p.add_argument("--env", action="append", metavar="KEY=VALUE",
                   help="Extra environment for every cell (repeatable)")
    p.add_argument("--max-cells", type=int, default=0,
                   help="Stop after this many cells (0 = until the queue drains)")
    p.add_argument("--wait", type=float, default=0.0,
                   help="Keep polling this many seconds after the queue empties, "
                        "for queues still being filled (0 = exit immediately)")
    p.add_argument("--poll-interval", type=float, default=10.0,
                   help="Seconds between polls while waiting (default: 10)")
    p.add_argument("--heartbeat-interval", type=float,
                   default=claim.DEFAULT_HEARTBEAT_INTERVAL,
                   help="Seconds between heartbeat touches (default: 60)")
    p.add_argument("--stale-seconds", type=float, default=claim.DEFAULT_STALE_SECONDS,
                   help="Reclaim claims whose heartbeat is older than this (default: 300)")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-run cells even when their output_check paths exist")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what the pending cells would run, then exit. "
                        "Claims nothing and changes nothing.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    worker = Worker(args)
    worker.install_signal_handlers()
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(main())
