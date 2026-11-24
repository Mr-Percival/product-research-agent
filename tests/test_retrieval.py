import numpy as np

from pra.retrieval import BM25Index, mmr


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_mmr_lambda_one_equals_cosine_order():
    q = _unit([1, 0])
    docs = np.stack([_unit([1, 0.1]), _unit([1, 0.5]), _unit([0, 1])])
    assert mmr(q, docs, k=3, lam=1.0) == [0, 1, 2]


def test_mmr_low_lambda_skips_near_duplicate():
    q = _unit([1, 0])
    docs = np.stack([_unit([1, 0.1]), _unit([1, 0.11]), _unit([0.6, 0.8])])
    assert mmr(q, docs, k=2, lam=0.3) == [0, 2]


def test_bm25_ranks_matching_doc_first():
    idx = BM25Index(["a", "b", "c"], ["quiet blender glass jar", "toaster oven", "coffee grinder"])
    assert idx.search("quiet blender", k=1)[0][0] == "a"


def test_rrf_rewards_agreement():
    from pra.retrieval import rrf

    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert set(fused[:2]) == {"a", "b"} and fused[-1] in {"c", "d"}
