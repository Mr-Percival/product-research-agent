import math

import numpy as np

from pra.metrics import (
    alpha_ndcg_at_k,
    bootstrap_ci,
    distinct_at_k,
    intra_list_diversity,
    judged_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_ndcg_perfect_and_reversed():
    gains = {"a": 1.0, "b": 0.1, "c": 0.0}
    assert ndcg_at_k(["a", "b", "c"], gains) == 1.0
    worst = ndcg_at_k(["c", "b", "a"], gains)
    expected = (0.1 / math.log2(3) + 1.0 / math.log2(4)) / (1.0 + 0.1 / math.log2(3))
    assert math.isclose(worst, expected)


def test_ndcg_unjudged_counts_zero():
    assert ndcg_at_k(["x", "a"], {"a": 1.0}) == 1.0 / math.log2(3)


def test_recall_divides_by_attainable():
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, k=2) == 1 / 2
    assert recall_at_k(["a", "b"], {"a", "b", "c"}, k=2) == 1.0
    assert recall_at_k(["a", "x"], {"a", "b"}, k=10) == 0.5


def test_judged():
    assert judged_at_k(["a", "x", "b", "y"], {"a", "b"}, k=4) == 0.5


def test_alpha_ndcg_rewards_brand_variety():
    sub = {"a1": "A", "a2": "A", "b1": "B"}
    assert math.isclose(alpha_ndcg_at_k(["a1", "b1", "a2"], sub), 1.0)
    assert alpha_ndcg_at_k(["a1", "a2", "b1"], sub) < 1.0
    assert alpha_ndcg_at_k(["x", "y"], sub) == 0.0


def test_distinct():
    assert distinct_at_k(["a", "b", "c", "z"], {"a": "X", "b": "X", "c": "Y"}) == 2


def test_bootstrap_ci_contains_mean():
    m, lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert lo < m < hi and math.isclose(m, 0.5)


def test_diversity_identical_is_zero():
    v = np.tile(np.array([[1.0, 0.0]]), (3, 1))
    assert math.isclose(intra_list_diversity(v), 0.0)
