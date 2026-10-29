"""utils/s1_pools split logic on fake items (no dataset download)."""

import random

from utils import s1_pools


def _items(n_groups=5, per=37):
    return [{"key": f"k{g}-{i}", "category": f"g{g}"} for g in range(n_groups) for i in range(per)]


def test_split_is_disjoint_exact_and_order_free():
    items = _items()
    ev, tr = s1_pools._split(items, lambda g, c: c // 3)
    assert {x["key"] for x in ev}.isdisjoint({x["key"] for x in tr})
    assert len(ev) + len(tr) == len(items)
    assert all(sum(x["category"] == f"g{g}" for x in ev) == 37 // 3 for g in range(5))
    shuffled = items[:]
    random.Random(1).shuffle(shuffled)
    ev2, _ = s1_pools._split(shuffled, lambda g, c: c // 3)
    assert sorted(x["key"] for x in ev) == sorted(x["key"] for x in ev2)
