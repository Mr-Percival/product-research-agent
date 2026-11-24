"""Transparent product score: relevance x review evidence for the features the query asks for."""

from __future__ import annotations


def evidence_score(feature_counts: dict[str, int], prior: float = 1.0) -> float:
    """Smoothed share of positive mentions: (pos + prior) / (pos + neg + 2 * prior).

    The Laplace prior keeps products with few reviews near 0.5 instead of 0 or 1.
    """
    pos = feature_counts.get("positive", 0)
    neg = feature_counts.get("negative", 0)
    return (pos + prior) / (pos + neg + 2 * prior)


def product_score(relevance: float, evidence: float, alpha: float = 1.0) -> float:
    """score = relevance * evidence**alpha. alpha = 0 ignores reviews (ablation)."""
    return relevance * evidence**alpha
