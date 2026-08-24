"""Cell + worker dispatch: a coordinator-free work queue on the filesystem.

The design goal is that **the worker never changes**. A cell fully describes
the work — an arbitrary Python script, an inline snippet, an importable
function, or a shell command — so new training, inference, or data-generation
sweeps mean writing a manifest, not another worker script.

Modules:
    claim.py    queue primitives (atomic claim, heartbeat, GC, state moves)
    cells.py    cell schema, validation, command building, manifest expansion
    worker.py   the one generic worker
    queue.py    CLI: init / expand / add / status / list / logs / retry / gc
    _runner.py  in-subprocess shim for `call` cells
"""
