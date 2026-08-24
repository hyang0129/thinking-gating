#!/usr/bin/env python3
"""Queue CLI — build, inspect, and repair a dispatch queue.

    # 1. turn a sweep manifest into cells
    python scripts/dispatch/queue.py expand configs/dispatch/gsm8k_probe.json

    # 2. dispatch workers (needs user approval — it submits jobs)
    python scripts/gpu_dispatch.py run --desc "probe sweep" \
        .venv/bin/python scripts/dispatch/worker.py --root shared/dispatch/gsm8k_probe

    # 3. watch it drain
    python scripts/dispatch/queue.py status --root shared/dispatch/gsm8k_probe

    # 4. inspect and re-run whatever failed
    python scripts/dispatch/queue.py logs --root shared/dispatch/gsm8k_probe --cell <id>
    python scripts/dispatch/queue.py retry --root shared/dispatch/gsm8k_probe --all

Every subcommand is read-only except `expand`, `add`, `retry`, `gc`, and
`clear`. None of them submit jobs — launching workers goes through
gpu_dispatch.py, which keeps job submission on one approval path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch import claim  # noqa: E402
from scripts.dispatch import cells as cells_mod  # noqa: E402

_DEFAULT_ROOT_BASE = "shared/dispatch"


def _resolve_root(root: str) -> Path:
    p = Path(root)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _default_root_for(manifest: dict) -> Path:
    return _PROJECT_ROOT / _DEFAULT_ROOT_BASE / manifest["name"]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_expand(args: argparse.Namespace) -> int:
    manifest = cells_mod.load_manifest(Path(args.manifest))
    cells = cells_mod.expand_manifest(manifest)
    if args.limit:
        cells = cells[: args.limit]

    root = _resolve_root(args.root) if args.root else _default_root_for(manifest)
    print(f"manifest {manifest['name']!r} → {len(cells)} cell(s)")
    print(f"root: {root}")

    if args.dry_run:
        for cell in cells:
            print(f"  {cell['cell_id']}")
            print(f"      {cells_mod.describe_command(cell)}")
            for path in cell.get("output_check", []) or []:
                print(f"      check: {path}")
        print("\n(dry run — nothing written)")
        return 0

    claim.init_dispatch_dirs(root)
    tally = {"added": 0, "skipped": 0, "replaced": 0}
    for cell in cells:
        _, action = claim.add_cell(root, cell, replace=args.replace)
        tally[action] += 1
    print(f"added {tally['added']}, replaced {tally['replaced']}, "
          f"skipped {tally['skipped']} (already queued or finished)")
    print(f"\nnext: dispatch a worker with\n"
          f"  python scripts/gpu_dispatch.py run --desc {manifest['name']!r} \\\n"
          f"      .venv/bin/python scripts/dispatch/worker.py --root {root}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    """Add cells from a JSON list, JSONL file, or stdin."""
    raw = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if stripped.startswith("["):
        cells = json.loads(raw)
    elif stripped.startswith("{") and "\n{" not in stripped.rstrip():
        cells = [json.loads(raw)]
    else:
        cells = [json.loads(line) for line in raw.splitlines() if line.strip()]

    root = _resolve_root(args.root)
    claim.init_dispatch_dirs(root)
    tally = {"added": 0, "skipped": 0, "replaced": 0}
    for cell in cells:
        cells_mod.validate_cell(cell)
        _, action = claim.add_cell(root, cell, replace=args.replace)
        tally[action] += 1
    print(f"added {tally['added']}, replaced {tally['replaced']}, skipped {tally['skipped']}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    counts = claim.count_status(root)
    workers = claim.live_workers(root, args.stale_seconds)

    if args.json:
        failures = [
            {"cell_id": cell["cell_id"],
             "status": (cell.get("result") or {}).get("status"),
             "error": ((cell.get("result") or {}).get("error") or "").splitlines()[-1:] }
            for _, cell in claim.iter_cells(root, "failed")
        ]
        print(json.dumps({"root": str(root), "counts": counts,
                          "workers": workers, "failed": failures}, indent=2))
        return 0

    total = counts["total"]
    finished = counts["done"] + counts["failed"]
    pct = (100 * finished / total) if total else 0.0
    print(f"queue: {root}")
    print(f"  pending {counts['pending']:>5}")
    print(f"  claimed {counts['claimed']:>5}")
    print(f"  done    {counts['done']:>5}")
    print(f"  failed  {counts['failed']:>5}")
    print(f"  total   {total:>5}   ({pct:.0f}% finished)")

    if workers:
        print("\nworkers:")
        for w in workers:
            state = "alive" if w["alive"] else "STALE"
            running = ", ".join(w["cells"]) or "-"
            print(f"  {w['worker_id']:<28} {state:<6} "
                  f"hb {w['seconds_since_heartbeat']:>6.0f}s ago   {running}")
        if any(not w["alive"] for w in workers):
            print("\n  stale workers hold cells that `queue.py gc` will re-queue.")

    failed = list(claim.iter_cells(root, "failed"))
    if failed:
        print(f"\nfailed cells ({len(failed)}):")
        for _, cell in failed[: args.max_failed]:
            result = cell.get("result") or {}
            first = (result.get("error") or "").strip().splitlines()
            summary = first[-1][:100] if first else result.get("status", "?")
            print(f"  {cell['cell_id']}")
            print(f"      {result.get('status', '?')}: {summary}")
        if len(failed) > args.max_failed:
            print(f"  ... and {len(failed) - args.max_failed} more")
        print("\n  `queue.py logs --cell <id>` for the full log, "
              "`queue.py retry --all` to re-queue them.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    rows = []
    states = claim.STATES if args.state == "all" else (args.state,)
    for state in states:
        for _, cell in claim.iter_cells(root, state):
            if args.tag and args.tag not in (cell.get("tags") or []):
                continue
            rows.append((state, cell))

    if args.json:
        print(json.dumps([{"state": s, **c} for s, c in rows], indent=2))
        return 0
    if not rows:
        print("no cells")
        return 0
    for state, cell in rows:
        result = cell.get("result") or {}
        extra = f"  [{result['status']} {result.get('duration_s', '')}s]" if result else ""
        print(f"{state:<8} {cell['cell_id']}{extra}")
        if args.verbose:
            print(f"         {cells_mod.describe_command(cell)}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    logs = sorted((root / "logs").glob(f"{args.cell}.attempt*.log"))
    if not logs:
        print(f"no logs for cell {args.cell!r} under {root / 'logs'}", file=sys.stderr)
        return 1
    target = logs[-1] if not args.attempt else root / "logs" / f"{args.cell}.attempt{args.attempt}.log"
    if not target.exists():
        print(f"no such attempt: {target}", file=sys.stderr)
        return 1
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if args.tail:
        lines = lines[-args.tail:]
    print(f"# {target}  ({len(logs)} attempt(s) recorded)")
    print("\n".join(lines))
    return 0


def cmd_retry(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    if not args.all and not args.cell:
        print("pass --cell <id> or --all", file=sys.stderr)
        return 2
    moved = 0
    for path, cell in list(claim.iter_cells(root, "failed")):
        if args.cell and cell["cell_id"] != args.cell:
            continue
        claim.requeue(root, path, reset_attempts=True)
        moved += 1
        print(f"re-queued {cell['cell_id']}")
    if not moved:
        print("nothing to retry")
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    reclaimed = claim.gc_stale_claims(root, args.stale_seconds)
    print(f"reclaimed {len(reclaimed)} stale cell(s)")
    for path in reclaimed:
        print(f"  {claim.cell_id_from_path(path)}")
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    if not args.confirm:
        print("refusing to delete without --confirm", file=sys.stderr)
        return 2
    states = claim.STATES if args.state == "all" else (args.state,)
    removed = 0
    for state in states:
        for path, _ in list(claim.iter_cells(root, state)):
            path.unlink()
            removed += 1
    print(f"deleted {removed} cell(s) from {', '.join(states)}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build and inspect a cell dispatch queue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("expand", help="expand a sweep manifest into pending cells")
    e.add_argument("manifest")
    e.add_argument("--root", default=None,
                   help=f"default: {_DEFAULT_ROOT_BASE}/<manifest name>")
    e.add_argument("--limit", type=int, default=0, help="only queue the first N cells")
    e.add_argument("--replace", action="store_true",
                   help="overwrite cells that already exist (not claimed ones)")
    e.add_argument("--dry-run", action="store_true", help="print cells, write nothing")
    e.set_defaults(func=cmd_expand)

    a = sub.add_parser("add", help="add cells from JSON / JSONL / stdin")
    a.add_argument("--root", required=True)
    a.add_argument("--file", default="-", help="JSON list, JSONL, or '-' for stdin")
    a.add_argument("--replace", action="store_true")
    a.set_defaults(func=cmd_add)

    s = sub.add_parser("status", help="counts, live workers, and failure summary")
    s.add_argument("--root", required=True)
    s.add_argument("--json", action="store_true")
    s.add_argument("--max-failed", type=int, default=10)
    s.add_argument("--stale-seconds", type=float, default=claim.DEFAULT_STALE_SECONDS)
    s.set_defaults(func=cmd_status)

    ls = sub.add_parser("list", help="list cells in one state")
    ls.add_argument("--root", required=True)
    ls.add_argument("--state", default="all",
                    choices=(*claim.STATES, "all"))
    ls.add_argument("--tag", default=None, help="only cells carrying this tag")
    ls.add_argument("--json", action="store_true")
    ls.add_argument("--verbose", "-v", action="store_true")
    ls.set_defaults(func=cmd_list)

    lg = sub.add_parser("logs", help="print a cell's log")
    lg.add_argument("--root", required=True)
    lg.add_argument("--cell", required=True)
    lg.add_argument("--attempt", type=int, default=0, help="default: latest")
    lg.add_argument("--tail", type=int, default=0)
    lg.set_defaults(func=cmd_logs)

    r = sub.add_parser("retry", help="move failed cells back to pending")
    r.add_argument("--root", required=True)
    r.add_argument("--cell", default=None)
    r.add_argument("--all", action="store_true")
    r.set_defaults(func=cmd_retry)

    g = sub.add_parser("gc", help="re-queue cells held by dead workers")
    g.add_argument("--root", required=True)
    g.add_argument("--stale-seconds", type=float, default=claim.DEFAULT_STALE_SECONDS)
    g.set_defaults(func=cmd_gc)

    c = sub.add_parser("clear", help="delete cells (destructive)")
    c.add_argument("--root", required=True)
    c.add_argument("--state", default="failed", choices=(*claim.STATES, "all"))
    c.add_argument("--confirm", action="store_true")
    c.set_defaults(func=cmd_clear)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (cells_mod.CellError, claim.QueueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
