"""Ranking metrics. ESCI gains follow the KDD Cup 2022 Task 1 convention."""

from __future__ import annotations

import numpy as np

ESCI_GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def dcg(gains: list[float], k: int) -> float:
    g = np.asarray(gains[:k], dtype=float)
    if g.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, g.size + 2))
    return float(np.sum(g * discounts))


def ndcg_at_k(ranked_ids: list[str], gains: dict[str, float], k: int = 10) -> float:
    """nDCG@k. Unjudged ids count as gain 0. Ideal ranking uses all judged gains."""
    ideal = dcg(sorted(gains.values(), reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg([gains.get(pid, 0.0) for pid in ranked_ids], k) / ideal


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int = 10) -> float:
    """Share of the attainable relevant items found in the top-k: divides by min(|relevant|, k)."""
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / min(len(relevant), k)


def judged_at_k(ranked_ids: list[str], judged: set[str], k: int = 10) -> float:
    """Share of the top-k that has a human judgement. Low values mean nDCG is a lower bound."""
    top = ranked_ids[:k]
    return len([p for p in top if p in judged]) / k if top else 0.0


def alpha_ndcg_at_k(ranked_ids: list[str], subtopics: dict[str, str], alpha: float = 0.5, k: int = 10) -> float:
    """alpha-nDCG@k (Clarke et al., 2008) with one subtopic per relevant item.

    `subtopics` maps each relevant item to its subtopic (here: its brand). Items not in the map
    are non-relevant. The n-th item of an already-seen subtopic earns (1 - alpha)^n, so repeating a
    brand pays less. The ideal ranking is built greedily, the usual approximation.
    """

    def gains(ids: list[str]) -> list[float]:
        seen: dict[str, int] = {}
        out = []
        for pid in ids:
            s = subtopics.get(pid)
            if s is None:
                out.append(0.0)
                continue
            out.append((1 - alpha) ** seen.get(s, 0))
            seen[s] = seen.get(s, 0) + 1
        return out

    # Greedy ideal: round-robin across subtopics yields the max gain at each step for single-subtopic items.
    by_topic: dict[str, list[str]] = {}
    for pid, s in subtopics.items():
        by_topic.setdefault(s, []).append(pid)
    ideal_ids: list[str] = []
    depth = 0
    while len(ideal_ids) < min(k, len(subtopics)):
        for items in by_topic.values():
            if depth < len(items):
                ideal_ids.append(items[depth])
        depth += 1
    ideal = dcg(gains(ideal_ids), k)
    return dcg(gains(ranked_ids), k) / ideal if ideal > 0 else 0.0


def distinct_at_k(ranked_ids: list[str], attribute: dict[str, str], k: int = 10) -> int:
    """Number of distinct attribute values (e.g. brands) in the top-k; items without the attribute are ignored."""
    return len({attribute[p] for p in ranked_ids[:k] if p in attribute})


def bootstrap_ci(values, n_boot: int = 10_000, seed: int = 42, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and percentile CI over queries. Pass paired differences to compare two systems."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(v.mean()), float(lo), float(hi)


def intra_list_diversity(embeddings: np.ndarray) -> float:
    """1 - mean pairwise cosine similarity of L2-normalised embeddings (higher = more diverse)."""
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    sims = embeddings @ embeddings.T
    upper = sims[np.triu_indices(n, k=1)]
    return float(1.0 - upper.mean())
