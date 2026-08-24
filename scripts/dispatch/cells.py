"""Cell schema, command building, and manifest expansion.

A **cell** is one unit of work, fully self-describing. The worker reads a cell
and runs it; it never knows what experiment it is part of. That is the whole
point — a new sweep is a new manifest, not a new worker.

Four kinds cover everything this repo will need:

    python_script   {"script": "scripts/run_experiment.py", "args": [...]}
    python_code     {"code": "import torch; print(torch.cuda.is_available())"}
    call            {"target": "tasks.gsm8k:load_gsm8k", "kwargs": {...}}
    shell           {"command": "nvidia-smi > gpu.txt"}

`call` is the most useful for new work: point at any importable function, pass
JSON kwargs, and its return value is captured to `results/<cell_id>.json`. No
CLI wrapper needed to put a function on the cluster.

Common fields (all optional unless noted):

    cell_id       str   required, unique, filename-safe
    kind          str   required, one of the four above
    cwd           str   working directory, relative to project root (default: root)
    env           dict  extra environment for this cell only
    output_check  list  paths that must exist after a successful run. Present
                        before the run  → the cell is skipped (resume). Absent
                        after exit 0    → the cell is FAILED, not silently
                        completed, which is how a script that exits 0 on an
                        internal error gets caught.
    timeout_s     num   kill the cell (process group) after this long
    max_attempts  int   re-queue on failure until this many attempts (default 1)
    priority      int   0-999, lower runs first (default 100)
    tags          list  free-form labels, for filtering in the CLI
    meta          dict  free-form, carried into the result record

A **manifest** describes a whole sweep and expands into cells:

    {
      "name": "gsm8k_probe",
      "kind": "python_script",
      "script": "scripts/run_experiment.py",
      "args": ["--method", "{method}", "--seed", "{seed}", "--out-dir", "{out}"],
      "constants": {"out": "output/gsm8k_probe/{method}_seed{seed}"},
      "grid": {"method": ["mlp", "contrastive"], "seed": [42, 1, 2, 3, 4]},
      "output_check": ["{out}/metrics.json"]
    }

`grid` takes the cartesian product; `zip` varies several keys in lockstep;
`exclude` drops combinations. Every string field is `{name}`-substituted from
the combination, and substitution runs to a fixed point, so constants may
reference grid variables (as `out` does above).
"""

from __future__ import annotations

import itertools
import json
import re
import shlex
from pathlib import Path
from typing import Any, Iterable, Optional

KINDS = ("python_script", "python_code", "call", "shell")

_FIELD_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_MAX_SUBST_PASSES = 10


class CellError(ValueError):
    """Raised for a malformed cell or manifest."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_cell(cell: dict) -> dict:
    """Check a cell is runnable. Returns it unchanged, or raises CellError."""
    if not isinstance(cell, dict):
        raise CellError(f"cell must be a dict, got {type(cell).__name__}")
    cid = cell.get("cell_id")
    if not cid:
        raise CellError("cell is missing 'cell_id'")
    kind = cell.get("kind")
    if kind not in KINDS:
        raise CellError(f"{cid}: kind must be one of {KINDS}, got {kind!r}")

    required = {
        "python_script": ("script",),
        "python_code": ("code",),
        "call": ("target",),
        "shell": ("command",),
    }[kind]
    for field in required:
        if not cell.get(field):
            raise CellError(f"{cid}: kind {kind!r} requires '{field}'")

    if kind == "call":
        target = cell["target"]
        if ":" not in target:
            raise CellError(
                f"{cid}: target must be 'module.path:function', got {target!r}"
            )
        if not isinstance(cell.get("kwargs", {}), dict):
            raise CellError(f"{cid}: 'kwargs' must be an object")
        if not isinstance(cell.get("args", []), list):
            raise CellError(f"{cid}: 'args' must be a list")
    if kind == "python_script" and not isinstance(cell.get("args", []), list):
        raise CellError(f"{cid}: 'args' must be a list")

    for field in ("output_check", "tags"):
        if field in cell and not isinstance(cell[field], list):
            raise CellError(f"{cid}: '{field}' must be a list")
    for field in ("env", "meta"):
        if field in cell and not isinstance(cell[field], dict):
            raise CellError(f"{cid}: '{field}' must be an object")
    if int(cell.get("max_attempts", 1)) < 1:
        raise CellError(f"{cid}: max_attempts must be >= 1")
    timeout = cell.get("timeout_s")
    if timeout is not None and float(timeout) <= 0:
        raise CellError(f"{cid}: timeout_s must be positive")
    return cell


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------

def resolve_cwd(cell: dict, project_root: Path) -> Path:
    cwd = cell.get("cwd")
    if not cwd:
        return Path(project_root)
    cwd = Path(cwd)
    return cwd if cwd.is_absolute() else Path(project_root) / cwd


def output_paths(cell: dict, project_root: Path) -> list[Path]:
    """`output_check` entries resolved against the cell's working directory."""
    base = resolve_cwd(cell, project_root)
    out = []
    for raw in cell.get("output_check", []) or []:
        p = Path(raw)
        out.append(p if p.is_absolute() else base / p)
    return out


def build_command(
    cell: dict,
    *,
    python: str,
    project_root: Path,
    scratch_dir: Path,
    result_file: Optional[Path] = None,
) -> list[str]:
    """Turn a cell into an argv list for subprocess execution.

    Every kind runs as a *subprocess*, never inside the worker: a segfault,
    OOM-kill, or `sys.exit()` in cell code takes down the cell, not the worker
    that has to report on it.
    """
    validate_cell(cell)
    kind = cell["kind"]
    project_root = Path(project_root)

    if kind == "python_script":
        script = Path(cell["script"])
        if not script.is_absolute():
            script = project_root / script
        return [python, str(script), *[str(a) for a in cell.get("args", [])]]

    if kind == "python_code":
        scratch_dir.mkdir(parents=True, exist_ok=True)
        snippet = scratch_dir / f"{cell['cell_id']}.py"
        # Put the project root on sys.path so inline code can import tasks/,
        # utils/, scripts/ the same way a script in the repo would.
        preamble = (
            "import sys\n"
            f"sys.path.insert(0, {str(project_root)!r})\n"
        )
        snippet.write_text(preamble + cell["code"], encoding="utf-8")
        return [python, str(snippet)]

    if kind == "call":
        runner = Path(__file__).resolve().parent / "_runner.py"
        argv = [
            python, str(runner),
            "--target", cell["target"],
            "--project-root", str(project_root),
            "--payload", json.dumps({
                "args": cell.get("args", []),
                "kwargs": cell.get("kwargs", {}),
            }),
        ]
        if result_file is not None:
            argv += ["--result-file", str(result_file)]
        return argv

    # shell — an escape hatch. `bash -c`, not `-lc`: no profile, no surprises.
    return ["bash", "-c", cell["command"]]


def describe_command(cell: dict) -> str:
    """One-line human summary of what a cell runs, for logs and status output."""
    kind = cell.get("kind")
    if kind == "python_script":
        args = " ".join(shlex.quote(str(a)) for a in cell.get("args", []))
        return f"{cell['script']} {args}".strip()
    if kind == "python_code":
        first = next((ln for ln in cell["code"].splitlines() if ln.strip()), "")
        return f"<inline> {first[:70]}"
    if kind == "call":
        return f"{cell['target']}({_kwargs_summary(cell)})"
    if kind == "shell":
        return f"$ {cell['command'][:80]}"
    return "<unknown kind>"


def _kwargs_summary(cell: dict) -> str:
    parts = [repr(a) for a in cell.get("args", [])]
    parts += [f"{k}={v!r}" for k, v in (cell.get("kwargs") or {}).items()]
    joined = ", ".join(parts)
    return joined if len(joined) <= 60 else joined[:57] + "..."


# ---------------------------------------------------------------------------
# Template substitution
# ---------------------------------------------------------------------------

def substitute(value: Any, namespace: dict) -> Any:
    """Recursively replace `{name}` in strings, using values from `namespace`.

    Non-string leaves pass through untouched, so an int stays an int. A string
    that is *exactly* one placeholder adopts the referenced value's type —
    `"{seed}"` with seed=42 yields the integer 42, not "42".
    """
    if isinstance(value, str):
        whole = _FIELD_RE.fullmatch(value)
        if whole and whole.group(1) in namespace:
            return namespace[whole.group(1)]

        def repl(m: re.Match) -> str:
            key = m.group(1)
            if key not in namespace:
                raise CellError(
                    f"template refers to unknown variable {{{key}}} in {value!r}; "
                    f"known: {sorted(namespace)}"
                )
            return str(namespace[key])

        return _FIELD_RE.sub(repl, value)
    if isinstance(value, list):
        return [substitute(v, namespace) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, namespace) for k, v in value.items()}
    return value


def _resolve_namespace(namespace: dict) -> dict:
    """Let namespace entries reference each other, to a fixed point.

    `{"out": "output/{name}/seed{seed}"}` resolves once `name` and `seed` are
    known. Repeats until nothing changes so chains resolve in any order, and
    stops rather than looping forever on a self-reference.
    """
    resolved = dict(namespace)
    for _ in range(_MAX_SUBST_PASSES):
        updated = {
            k: substitute(v, resolved) if isinstance(v, (str, list, dict)) else v
            for k, v in resolved.items()
        }
        if updated == resolved:
            _reject_cycles(resolved)
            return resolved
        resolved = updated
    raise CellError(
        "template substitution did not converge after "
        f"{_MAX_SUBST_PASSES} passes — check for a self-referencing constant "
        f"in {sorted(namespace)}"
    )


def _reject_cycles(resolved: dict) -> None:
    """Catch constants that reference each other in a loop.

    A cycle like {"a": "{b}", "b": "{a}"} reaches a fixed point where each
    value is still a literal placeholder — stable, but not resolved. Left
    alone it would silently produce a cell containing the string "{a}", so
    treat a surviving placeholder that names a known key as the error it is.
    """
    for key, value in resolved.items():
        if not isinstance(value, str):
            continue
        for match in _FIELD_RE.finditer(value):
            if match.group(1) in resolved:
                raise CellError(
                    f"template substitution did not converge: constant "
                    f"{key!r} still resolves to {value!r} — "
                    f"{key!r} and {match.group(1)!r} reference each other"
                )


# ---------------------------------------------------------------------------
# Manifest expansion
# ---------------------------------------------------------------------------

_MANIFEST_ONLY = {"name", "grid", "zip", "exclude", "constants", "cell_id_template"}


def _combinations(manifest: dict) -> list[dict]:
    """Cartesian product of `grid`, crossed with lockstep tuples from `zip`."""
    grid = manifest.get("grid") or {}
    zipped = manifest.get("zip") or {}

    for name, values in {**grid, **zipped}.items():
        if not isinstance(values, list) or not values:
            raise CellError(f"grid/zip entry {name!r} must be a non-empty list")

    grid_combos: Iterable[dict]
    if grid:
        keys = list(grid)
        grid_combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    else:
        grid_combos = [{}]

    if zipped:
        lengths = {len(v) for v in zipped.values()}
        if len(lengths) > 1:
            raise CellError(
                f"'zip' lists must be the same length, got { {k: len(v) for k, v in zipped.items()} }"
            )
        zip_combos = [
            {k: v[i] for k, v in zipped.items()} for i in range(next(iter(lengths)))
        ]
    else:
        zip_combos = [{}]

    return [{**g, **z} for g in grid_combos for z in zip_combos]


def _excluded(combo: dict, exclude: list[dict]) -> bool:
    """True if `combo` matches any exclusion rule (a partial dict match)."""
    for rule in exclude or []:
        if all(combo.get(k) == v for k, v in rule.items()):
            return True
    return False


def _default_cell_id(name: str, combo: dict) -> str:
    if not combo:
        return name
    parts = [f"{k}-{combo[k]}" for k in sorted(combo)]
    raw = f"{name}__" + "_".join(parts)
    return re.sub(r"[^A-Za-z0-9._-]", "-", raw)


def expand_manifest(manifest: dict) -> list[dict]:
    """Expand a sweep manifest into a list of validated cells."""
    if not isinstance(manifest, dict):
        raise CellError("manifest must be a JSON object")
    name = manifest.get("name")
    if not name:
        raise CellError("manifest is missing 'name'")

    template = {k: v for k, v in manifest.items() if k not in _MANIFEST_ONLY}
    constants = manifest.get("constants") or {}
    id_template = manifest.get("cell_id_template")

    cells: list[dict] = []
    seen: set[str] = set()
    for combo in _combinations(manifest):
        if _excluded(combo, manifest.get("exclude") or []):
            continue
        namespace = _resolve_namespace({"name": name, **constants, **combo})
        cell = substitute(dict(template), namespace)
        cell["cell_id"] = (
            substitute(id_template, namespace) if id_template
            else _default_cell_id(name, combo)
        )
        cell.setdefault("meta", {})
        cell["meta"] = {**cell["meta"], "manifest": name, **combo}
        validate_cell(cell)
        if cell["cell_id"] in seen:
            raise CellError(
                f"duplicate cell_id {cell['cell_id']!r} — set 'cell_id_template' "
                "to something that varies across the grid"
            )
        seen.add(cell["cell_id"])
        cells.append(cell)

    if not cells:
        raise CellError(f"manifest {name!r} expanded to zero cells")
    return cells


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
