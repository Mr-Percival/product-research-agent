"""Lexical (BM25) and dense retrieval plus Maximal Marginal Relevance re-ranking."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int = 10, candidates: list[int] | None = None) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.asarray(candidates) if candidates is not None else np.arange(len(self.ids))
        order = idx[np.argsort(-scores[idx], kind="stable")][:k]
        return [(self.ids[i], float(scores[i])) for i in order]


def cosine_top_k(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectors must be L2-normalised, so the dot product equals cosine similarity."""
    sims = doc_vecs @ query_vec
    order = np.argsort(-sims, kind="stable")[:k]
    return order, sims[order]


def rrf(rankings: list[list[str]], k: int = 60, top: int | None = None) -> list[str]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum over lists of 1 / (k + rank)."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, pid in enumerate(ranking, start=1):
            score[pid] = score.get(pid, 0.0) + 1.0 / (k + r)
    fused = sorted(score, key=lambda p: (-score[p], p))
    return fused[:top] if top else fused


def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 10, lam: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

    Picks, at each step, argmax_d [ lam * sim(q, d) - (1 - lam) * max_{s in selected} sim(d, s) ].
    lam = 1 reproduces plain cosine ranking; lower lam trades relevance for diversity.
    Returns positions into doc_vecs.
    """
    n = doc_vecs.shape[0]
    if n == 0:
        return []
    rel = doc_vecs @ query_vec
    selected: list[int] = []
    max_sim_to_sel = np.full(n, -np.inf)
    remaining = np.ones(n, dtype=bool)
    for _ in range(min(k, n)):
        redundancy = np.where(np.isfinite(max_sim_to_sel), max_sim_to_sel, 0.0)
        score = lam * rel - (1 - lam) * redundancy
        score[~remaining] = -np.inf
        best = int(np.argmax(score))
        selected.append(best)
        remaining[best] = False
        max_sim_to_sel = np.maximum(max_sim_to_sel, doc_vecs @ doc_vecs[best])
    return selected
