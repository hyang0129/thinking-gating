"""Reading capture directories written by capture_inference_thinking.py.

A capture is sharded — one `meta.shardNN.jsonl` and one activations `.npz` per
shard — so a run can be fanned across nodes. Everything downstream wants a
single aligned (meta, activations) pair, which is what `load_capture` returns.

The alignment between a meta row and its activation row is positional within a
shard, so shards are always concatenated in the same sorted order and their
lengths are checked against each other. A silent misalignment here would train
a probe on the wrong labels and still look plausible, so it is an error, not a
warning.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_SHARD_RE = re.compile(r"\.shard(\d+)\.")


def _shard_key(path: Path) -> int:
    m = _SHARD_RE.search(path.name)
    return int(m.group(1)) if m else -1


def shard_paths(capture_dir: Path, pattern: str) -> list[Path]:
    return sorted(Path(capture_dir).glob(pattern), key=_shard_key)


def load_meta(capture_dir: Path) -> list[dict]:
    """All meta rows, shards concatenated in shard order."""
    capture_dir = Path(capture_dir)
    paths = shard_paths(capture_dir, "meta.shard*.jsonl")
    if not paths:
        raise FileNotFoundError(f"no meta.shard*.jsonl under {capture_dir}")
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def load_activations(capture_dir: Path, mode: str = "off") -> np.ndarray:
    """Prefill activations, (N, num_layers+1, hidden_dim), shards concatenated."""
    if mode not in ("off", "on"):
        raise ValueError(f"mode must be 'off' or 'on', got {mode!r}")
    paths = shard_paths(Path(capture_dir), f"activations_thinking_{mode}.shard*.npz")
    if not paths:
        raise FileNotFoundError(
            f"no activations_thinking_{mode}.shard*.npz under {capture_dir}")
    chunks = []
    for path in paths:
        with np.load(path) as data:
            key = "activations" if "activations" in data else data.files[0]
            chunks.append(data[key])
    return np.concatenate(chunks, axis=0)


def load_capture(capture_dir: Path, mode: str = "off") -> tuple[list[dict], np.ndarray]:
    """Return (meta rows, activations) with per-shard alignment verified."""
    capture_dir = Path(capture_dir)
    meta_paths = shard_paths(capture_dir, "meta.shard*.jsonl")
    act_paths = shard_paths(capture_dir, f"activations_thinking_{mode}.shard*.npz")
    if len(meta_paths) != len(act_paths):
        raise ValueError(
            f"{capture_dir}: {len(meta_paths)} meta shard(s) but "
            f"{len(act_paths)} activation shard(s) for mode {mode!r} — "
            "a shard failed or is still running")
    for m_path, a_path in zip(meta_paths, act_paths):
        if _shard_key(m_path) != _shard_key(a_path):
            raise ValueError(f"shard mismatch: {m_path.name} vs {a_path.name}")

    meta = load_meta(capture_dir)
    acts = load_activations(capture_dir, mode)
    if len(meta) != acts.shape[0]:
        raise ValueError(
            f"{capture_dir}: {len(meta)} meta rows but {acts.shape[0]} activation "
            "rows — refusing to guess the alignment")
    return meta, acts


# ---------------------------------------------------------------------------
# Thinking-off confidence: stored (meta) vs re-scored sidecar (B1)
# ---------------------------------------------------------------------------

# scripts/rescore_confidence.py writes <prefix>.shardNN.jsonl beside each meta
# shard and <prefix>.summary.json once every shard is done.
CONFIDENCE_SIDECAR_PREFIX = "confidence_off_v2"
CONFIDENCE_SOURCES = ("auto", "v2", "stored")


def load_confidence_sidecar(capture_dir: Path) -> dict[str, dict] | None:
    """sample_id -> sidecar row, or None when the capture has no sidecar."""
    paths = shard_paths(Path(capture_dir), f"{CONFIDENCE_SIDECAR_PREFIX}.shard*.jsonl")
    if not paths:
        return None
    out: dict[str, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if rec["sample_id"] in out:
                        raise ValueError(f"{path}: duplicate sample_id "
                                         f"{rec['sample_id']!r} in the sidecar")
                    out[rec["sample_id"]] = rec
    return out


def resolve_confidence(meta: list[dict], capture_dir: Path,
                       source: str = "auto") -> tuple[list[dict | None], dict]:
    """Thinking-off confidence for each meta row, and a record of where it came
    from.

    source:
      auto    the re-scored sidecar when the capture has one, else stored
      v2      the sidecar; refuse if it is missing
      stored  `confidence_off` as written by the capture, whatever its version

    The sidecar must cover every meta row; a partial one is an error, never a
    silent mix of re-scored and corrupted values. Stored rows without
    `confidence_version` are v1 (B1: scored with pad tokens attended), and the
    returned info says how many there are.
    """
    if source not in CONFIDENCE_SOURCES:
        raise ValueError(f"source must be one of {CONFIDENCE_SOURCES}, got {source!r}")
    sidecar = None if source == "stored" else load_confidence_sidecar(capture_dir)
    if source == "v2" and sidecar is None:
        raise FileNotFoundError(
            f"{capture_dir}: no {CONFIDENCE_SIDECAR_PREFIX}.shard*.jsonl -- run "
            "scripts/rescore_confidence.py, or pass --confidence-source stored")

    info: dict = {"capture_dir": str(capture_dir), "requested": source}
    if sidecar is not None:
        missing = [r["sample_id"] for r in meta if r["sample_id"] not in sidecar]
        if missing:
            raise ValueError(
                f"{capture_dir}: {len(missing)} meta row(s) have no re-scored "
                f"confidence in the sidecar (first: {missing[0]!r}) -- the "
                "rescore is incomplete; finish it rather than mix versions")
        recs = [sidecar[r["sample_id"]] for r in meta]
        flags: dict[str, int] = {}
        for rec in recs:
            for f in rec.get("flags", []):
                flags[f] = flags.get(f, 0) + 1
        info.update(source=f"{CONFIDENCE_SIDECAR_PREFIX} sidecar",
                    confidence_version=sorted({rec.get("confidence_version")
                                               for rec in recs}),
                    n_rows=len(recs), flags=flags)
        return [rec.get("confidence_off") for rec in recs], info

    versions: dict[str, int] = {}
    for r in meta:
        v = str(r.get("confidence_version", 1))
        versions[v] = versions.get(v, 0) + 1
    info.update(source="stored confidence_off", n_rows=len(meta),
                confidence_version_counts=versions)
    return [r.get("confidence_off") for r in meta], info


def load_config(capture_dir: Path) -> dict:
    path = Path(capture_dir) / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_labels(labels_file: Path) -> list[dict]:
    with open(labels_file, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def align_labels(meta: list[dict], labels: list[dict]) -> np.ndarray:
    """Row indices into `meta` for each label, matched on sample_id.

    Labels are matched by id rather than position: a re-run that reorders or
    drops rows would otherwise pair activations with the wrong outcome.
    """
    index = {row["sample_id"]: i for i, row in enumerate(meta)}
    missing = [lab["sample_id"] for lab in labels if lab["sample_id"] not in index]
    if missing:
        raise ValueError(
            f"{len(missing)} label(s) have no matching capture row "
            f"(first: {missing[0]!r})")
    return np.array([index[lab["sample_id"]] for lab in labels], dtype=np.int64)
