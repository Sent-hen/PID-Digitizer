"""Split selection helpers."""

from __future__ import annotations

import itertools
import random
from collections.abc import Mapping, Sequence


def rare_class_aware_pick(counts: Mapping[str, Mapping[str, int]], n_train: int,
                          rare: Sequence[str], seed: int) -> list[str]:
    """Choose `n_train` drawings for training from a small subset.

    `counts[uid][cls]` is the number of instances of `cls` in drawing `uid`.
    Constraints: every class present in the subset keeps >= 1 instance in the test
    remainder, and every rare class gets >= 1 instance in training. Among feasible
    choices, minimize sum over rare classes of (train share - n_train/N)^2, so each
    rare class is split roughly in proportion; ties are broken by a seeded shuffle.
    """
    uids = sorted(counts)
    classes = sorted({c for v in counts.values() for c, k in v.items() if k > 0})
    total = {c: sum(counts[u].get(c, 0) for u in uids) for c in classes}
    target = n_train / len(uids)
    order = list(itertools.combinations(uids, n_train))
    random.Random(seed).shuffle(order)  # seeded tie-break: first minimum wins
    best: tuple[float, tuple[str, ...]] | None = None
    for combo in order:
        train = {c: sum(counts[u].get(c, 0) for u in combo) for c in classes}
        if any(total[c] - train[c] < 1 for c in classes):
            continue
        if any(train.get(c, 0) < 1 for c in rare if total.get(c, 0) > 0):
            continue
        score = sum((train[c] / total[c] - target) ** 2 for c in rare if total.get(c, 0) > 0)
        if best is None or score < best[0]:
            best = (score, combo)
    if best is None:
        raise ValueError("no feasible split satisfies the class constraints")
    return sorted(best[1], key=lambda u: int(u.split("/")[1]) if u.split("/")[1].isdigit()
                  else 0)
