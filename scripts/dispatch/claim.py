"""Filesystem work queue — coordinator-free, NFS-safe.

Multiple workers on multiple nodes race to atomically `rename(2)` a cell file
out of `pending/` into `claimed/<worker_id>/`. POSIX rename is atomic on a
shared filesystem, so exactly one worker wins each cell. There is no
coordinator process, no database, and no lock server to keep alive.

Layout, anchored at the dispatch root:

    pending/<prio>__<cell_id>.json
    claimed/<worker_id>/<prio>__<cell_id>.json
    claimed/<worker_id>/heartbeat              # mtime touched while alive
    done/<prio>__<cell_id>.json                # cell + "result" block
    failed/<prio>__<cell_id>.json              # cell + "result" block
    logs/<cell_id>.attempt<N>.log              # stdout+stderr of each attempt
    results/<cell_id>.json                     # return value of `call` cells

Every state transition is write-temp-then-rename, so a worker killed mid-move
leaves either the old state or the new one — never a half-written cell. A cell
stranded in `claimed/` by a dead worker is reclaimed by `gc_stale_claims()`
once its heartbeat goes stale, and re-run; cells are expected to be idempotent
(see `output_check` in cells.py, which also makes re-runs cheap).

Stdlib only, and pure functions over the filesystem — no shared in-memory
state. That is what lets a worker, a CLI, and a test all drive the same queue.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Iterator, Optional

STATES = ("pending", "claimed", "done", "failed")
SUBDIRS = STATES + ("logs", "results")

# Workers touch their heartbeat every HEARTBEAT_INTERVAL seconds; a claim whose
# heartbeat is older than this is treated as abandoned and returned to pending.
DEFAULT_STALE_SECONDS = 300
DEFAULT_HEARTBEAT_INTERVAL = 60

_CELL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FILENAME_RE = re.compile(r"^(?P<prio>\d{3})__(?P<cell_id>.+)\.json$")


class QueueError(RuntimeError):
    """Raised for malformed cell ids, filenames, or impossible transitions."""


# ---------------------------------------------------------------------------
# Paths and naming
# ---------------------------------------------------------------------------

def init_dispatch_dirs(root: Path) -> Path:
    """Create the queue directories. Idempotent."""
    root = Path(root)
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def validate_cell_id(cell_id: str) -> str:
    """Cell ids become filenames, so keep them boring and path-safe."""
    if not isinstance(cell_id, str) or not _CELL_ID_RE.match(cell_id):
        raise QueueError(
            f"invalid cell_id {cell_id!r}: use letters, digits, dot, dash, "
            "underscore; must start alphanumeric"
        )
    return cell_id


def cell_filename(cell: dict) -> str:
    """`<priority>__<cell_id>.json`.

    Priority rides in the filename so that claiming — a sorted scan of
    `pending/` — hands out low-priority-number cells first without reading
    every file. Ties break on cell_id, which keeps ordering deterministic.
    """
    cell_id = validate_cell_id(cell.get("cell_id", ""))
    priority = int(cell.get("priority", 100))
    if not 0 <= priority <= 999:
        raise QueueError(f"priority must be 0-999, got {priority}")
    return f"{priority:03d}__{cell_id}.json"


def cell_id_from_path(path: Path) -> str:
    m = _FILENAME_RE.match(Path(path).name)
    if not m:
        raise QueueError(f"not a cell filename: {path}")
    return m.group("cell_id")


def log_path(root: Path, cell_id: str, attempt: int) -> Path:
    return Path(root) / "logs" / f"{cell_id}.attempt{attempt}.log"


def result_path(root: Path, cell_id: str) -> Path:
    return Path(root) / "results" / f"{cell_id}.json"


# ---------------------------------------------------------------------------
# Atomic read/write
# ---------------------------------------------------------------------------

def load_cell(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_cell_atomic(path: Path, cell: dict) -> Path:
    """Write JSON via a temp file in the same directory, then rename over."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cell, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def find_cell(root: Path, cell_id: str) -> Optional[tuple[str, Path]]:
    """Locate a cell in any state. Returns (state, path) or None."""
    root = Path(root)
    for state in ("pending", "done", "failed"):
        for path in (root / state).glob(f"*__{cell_id}.json"):
            if cell_id_from_path(path) == cell_id:
                return state, path
    claimed = root / "claimed"
    if claimed.is_dir():
        for worker_dir in claimed.iterdir():
            if not worker_dir.is_dir():
                continue
            for path in worker_dir.glob(f"*__{cell_id}.json"):
                if cell_id_from_path(path) == cell_id:
                    return "claimed", path
    return None


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------

def add_cell(root: Path, cell: dict, *, replace: bool = False) -> tuple[Path, str]:
    """Enqueue one cell. Returns (path, "added" | "skipped" | "replaced").

    A cell already present in *any* state is skipped unless `replace=True`, so
    re-expanding a manifest to append new work never re-runs finished cells and
    never steals a cell out from under a running worker.
    """
    root = init_dispatch_dirs(Path(root))
    cell_id = validate_cell_id(cell.get("cell_id", ""))
    existing = find_cell(root, cell_id)
    if existing is not None:
        state, path = existing
        if not replace:
            return path, "skipped"
        if state == "claimed":
            raise QueueError(
                f"cell {cell_id!r} is claimed by a live worker; refusing to replace"
            )
        path.unlink()
    target = root / "pending" / cell_filename(cell)
    write_cell_atomic(target, cell)
    return target, "replaced" if existing else "added"


def claim_next_cell(root: Path, worker_id: str) -> Optional[Path]:
    """Atomically claim the next pending cell, or return None if none remain.

    Sorted scan of `pending/` (priority-ordered by filename), attempting a
    rename on each candidate. Losing a race raises FileNotFoundError from
    rename, which just means another worker got there first — try the next one.
    """
    root = Path(root)
    pending = root / "pending"
    mine = root / "claimed" / worker_id
    mine.mkdir(parents=True, exist_ok=True)

    if not pending.is_dir():
        return None
    for candidate in sorted(pending.glob("*__*.json")):
        target = mine / candidate.name
        try:
            os.rename(candidate, target)
        except (FileNotFoundError, OSError):
            continue
        return target
    return None


def _move_with_result(cell_path: Path, dest_dir: Path, result: Optional[dict]) -> Path:
    """Attach `result` to the cell, then move it to `dest_dir` atomically.

    The result is written into the claimed file first and only then renamed, so
    an observer of the destination directory never sees a cell without its
    outcome recorded.
    """
    cell_path = Path(cell_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if result is not None:
        cell = load_cell(cell_path)
        cell["result"] = result
        write_cell_atomic(cell_path, cell)
    target = dest_dir / cell_path.name
    os.replace(cell_path, target)
    return target


def complete_cell(root: Path, cell_path: Path, result: Optional[dict] = None) -> Path:
    """claimed → done."""
    return _move_with_result(cell_path, Path(root) / "done", result)


def fail_cell(root: Path, cell_path: Path, result: Optional[dict] = None) -> Path:
    """claimed → failed (terminal; `retry_failed` puts it back)."""
    return _move_with_result(cell_path, Path(root) / "failed", result)


def release_cell(root: Path, cell_path: Path, result: Optional[dict] = None) -> Path:
    """claimed → pending, unchanged attempt count.

    Used when a worker is shut down mid-cell: the work goes straight back on
    the queue instead of waiting out the stale-claim timeout.
    """
    return _move_with_result(cell_path, Path(root) / "pending", result)


def retry_cell(root: Path, cell_path: Path, result: Optional[dict] = None) -> Path:
    """claimed → pending with `attempts` incremented (transient-failure path)."""
    cell_path = Path(cell_path)
    cell = load_cell(cell_path)
    cell["attempts"] = int(cell.get("attempts", 0)) + 1
    if result is not None:
        cell["result"] = result
    write_cell_atomic(cell_path, cell)
    target = Path(root) / "pending" / cell_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(cell_path, target)
    return target


def requeue(root: Path, path: Path, *, reset_attempts: bool = True) -> Path:
    """Move a done/failed cell back to pending (operator action)."""
    path = Path(path)
    cell = load_cell(path)
    if reset_attempts:
        cell["attempts"] = 0
    cell.pop("result", None)
    write_cell_atomic(path, cell)
    target = Path(root) / "pending" / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, target)
    return target


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------

def worker_id_for(pid: Optional[int] = None) -> str:
    """Stable-ish worker id: node slice + pid.

    `DISPATCH_NODE` is injected by gpu_dispatch.py and names the Jupyter slice
    (`alphagpu17-8881`). $HOSTNAME alone collides when two slices of the same
    physical node each run a worker.
    """
    node = os.environ.get("DISPATCH_NODE") or socket.gethostname()
    node = re.sub(r"[^A-Za-z0-9._-]", "-", node)
    return f"{node}_{pid or os.getpid()}"


def touch_heartbeat(root: Path, worker_id: str) -> Path:
    hb = Path(root) / "claimed" / worker_id / "heartbeat"
    hb.parent.mkdir(parents=True, exist_ok=True)
    hb.touch(exist_ok=True)
    now = time.time()
    os.utime(hb, (now, now))  # explicit: touch() may not bump mtime on some FS
    return hb


def gc_stale_claims(
    root: Path,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    now: Optional[float] = None,
) -> list[Path]:
    """Return cells held by dead workers to pending. Returns re-pended paths."""
    root = Path(root)
    now = time.time() if now is None else now
    claimed_root = root / "claimed"
    pending = root / "pending"
    pending.mkdir(parents=True, exist_ok=True)

    reclaimed: list[Path] = []
    if not claimed_root.is_dir():
        return reclaimed

    for worker_dir in sorted(claimed_root.iterdir()):
        if not worker_dir.is_dir():
            continue
        hb = worker_dir / "heartbeat"
        last_seen = hb.stat().st_mtime if hb.exists() else worker_dir.stat().st_mtime
        if now - last_seen < stale_seconds:
            continue
        for cell in sorted(worker_dir.glob("*__*.json")):
            target = pending / cell.name
            try:
                os.rename(cell, target)
                reclaimed.append(target)
            except OSError:
                continue
        try:
            if hb.exists():
                hb.unlink()
            worker_dir.rmdir()
        except OSError:
            pass  # another worker may have just claimed into this dir
    return reclaimed


def live_workers(
    root: Path,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    now: Optional[float] = None,
) -> list[dict]:
    """One record per worker directory: id, liveness, age, and held cells."""
    root = Path(root)
    now = time.time() if now is None else now
    claimed_root = root / "claimed"
    if not claimed_root.is_dir():
        return []

    out = []
    for worker_dir in sorted(claimed_root.iterdir()):
        if not worker_dir.is_dir():
            continue
        hb = worker_dir / "heartbeat"
        last_seen = hb.stat().st_mtime if hb.exists() else worker_dir.stat().st_mtime
        cells = [cell_id_from_path(p) for p in sorted(worker_dir.glob("*__*.json"))]
        out.append({
            "worker_id": worker_dir.name,
            "seconds_since_heartbeat": round(now - last_seen, 1),
            "alive": (now - last_seen) < stale_seconds,
            "cells": cells,
        })
    return out


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def iter_cells(root: Path, state: str) -> Iterator[tuple[Path, dict]]:
    """Yield (path, cell) for every cell in one state."""
    root = Path(root)
    if state not in STATES:
        raise QueueError(f"unknown state {state!r}; expected one of {STATES}")
    if state == "claimed":
        claimed_root = root / "claimed"
        if not claimed_root.is_dir():
            return
        for worker_dir in sorted(claimed_root.iterdir()):
            if worker_dir.is_dir():
                for path in sorted(worker_dir.glob("*__*.json")):
                    yield path, load_cell(path)
        return
    state_dir = root / state
    if not state_dir.is_dir():
        return
    for path in sorted(state_dir.glob("*__*.json")):
        yield path, load_cell(path)


def count_status(root: Path) -> dict[str, int]:
    """Cell counts per state, plus the total."""
    counts = {state: sum(1 for _ in iter_cells(root, state)) for state in STATES}
    counts["total"] = sum(counts.values())
    return counts
