"""
s1_pools.py — the Phase 0 train / eval item pools (issue #3, §4).

Membership is a pure function of each item's key (sha256 order within a
group), so it never depends on load order, dataset shuffles or shard count.

  eval   MMLU-Pro 3,000 (stratified by category) | BBH half of every subtask
         | GSM8K test | MATH-500 | LSAT-AR (all 230)
  train  MMLU-Pro the rest | BBH the other half | GSM8K train | MATH train

LSAT has only a 230-row test split, so it is eval-only: a family the flag never
trains on (leave-one-family-out for free). MATH train is Hendrycks MATH train,
disjoint from MATH-500 (a subset of MATH test).

The eval pool is part of the pre-registration: changing it after any eval
result exists needs a new POOL_VERSION and an erratum.
"""

from __future__ import annotations

import hashlib
import importlib

POOL_VERSION = 1
MMLU_PRO_EVAL = 3000
BBH_EVAL_FRACTION = 0.5


def _h(key: str) -> str:
    return hashlib.sha256(f"s1pool-v{POOL_VERSION}:{key}".encode()).hexdigest()


def family(item: dict) -> str | None:
    return (item.get("family") or item.get("category") or item.get("subtask")
            or item.get("subject"))


def _split(items: list[dict], n_eval_per_group) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(family(it) or "", []).append(it)
    ev, tr = [], []
    for g, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: _h(r["key"]))
        k = n_eval_per_group(g, len(rows))
        ev += rows[:k]
        tr += rows[k:]
    return ev, tr


def _mmlu_pro():
    from tasks.mmlu_pro import load_mmlu_pro
    items = load_mmlu_pro("test")
    n = len(items)
    # Proportional by category, largest-remainder so the total is exact.
    sizes: dict[str, int] = {}
    for it in items:
        sizes[family(it)] = sizes.get(family(it), 0) + 1
    quota = {g: MMLU_PRO_EVAL * c / n for g, c in sizes.items()}
    alloc = {g: int(q) for g, q in quota.items()}
    for g in sorted(quota, key=lambda g: (alloc[g] - quota[g], g))[:MMLU_PRO_EVAL - sum(alloc.values())]:
        alloc[g] += 1
    return _split(items, lambda g, _: alloc[g])


def _bbh():
    from tasks.bbh import load_bbh
    return _split(load_bbh("test", per_subtask=10 ** 9),
                  lambda g, c: round(c * BBH_EVAL_FRACTION))


def load_pool(pool: str, task: str) -> list[dict]:
    """Items of ``task`` in ``pool`` ('eval' or 'train'), in a fixed order."""
    if pool not in ("eval", "train"):
        raise ValueError(pool)
    if task == "mmlu_pro":
        ev, tr = _mmlu_pro()
    elif task == "bbh":
        ev, tr = _bbh()
    elif task == "gsm8k":
        from tasks.gsm8k import load_gsm8k
        ev, tr = load_gsm8k("test"), load_gsm8k("train")
    elif task == "math500":
        from tasks.math500 import load_math500, load_math_train
        ev, tr = load_math500("test"), load_math_train()
    elif task == "lsat":
        ev, tr = importlib.import_module("tasks.lsat").load_lsat_logic("test"), []
    else:
        raise ValueError(f"no pool for task {task!r}")
    return sorted(ev if pool == "eval" else tr, key=lambda r: _h(r["key"]))
