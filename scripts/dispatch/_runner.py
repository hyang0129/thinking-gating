#!/usr/bin/env python3
"""Subprocess shim for `call` cells.

Imports `module.path:function`, calls it with the cell's args/kwargs, and — if
the return value is JSON-serializable — writes it to the result file. Run as a
separate process by the worker, so anything the callee does to the interpreter
(exit, segfault, CUDA context) stays contained.

Not meant to be invoked by hand; see scripts/dispatch/cells.py.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path


def resolve_target(target: str):
    """'package.module:function' → the callable."""
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path)
    obj = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"target {target!r} resolved to a non-callable {type(obj).__name__}")
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one `call` cell.")
    parser.add_argument("--target", required=True, help="module.path:function")
    parser.add_argument("--payload", required=True, help='JSON {"args": [], "kwargs": {}}')
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--result-file", default=None)
    args = parser.parse_args()

    if args.project_root:
        root = str(Path(args.project_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    payload = json.loads(args.payload)
    call_args = payload.get("args", [])
    call_kwargs = payload.get("kwargs", {})

    try:
        func = resolve_target(args.target)
    except Exception:
        traceback.print_exc()
        print(f"\nFailed to resolve target {args.target!r}", file=sys.stderr)
        return 2

    print(f"[runner] calling {args.target} "
          f"with {len(call_args)} arg(s), {len(call_kwargs)} kwarg(s)", flush=True)
    value = func(*call_args, **call_kwargs)

    if args.result_file:
        out = Path(args.result_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            serialized = json.dumps({"target": args.target, "value": value}, default=str)
        except (TypeError, ValueError) as exc:
            # A non-serializable return is not a failure — the cell's real
            # output is whatever it wrote to disk. Record the type instead.
            serialized = json.dumps({
                "target": args.target,
                "value": None,
                "note": f"return value not JSON-serializable ({type(value).__name__}): {exc}",
            })
        out.write_text(serialized + "\n", encoding="utf-8")
        print(f"[runner] wrote result to {out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
