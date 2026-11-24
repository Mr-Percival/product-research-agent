from pra.scoring import evidence_score, product_score


def test_evidence_prior_with_no_reviews_is_half():
    assert evidence_score({}) == 0.5


def test_evidence_and_score_by_hand():
    ev = evidence_score({"positive": 8, "negative": 2})  # (8+1)/(10+2) = 0.75
    assert ev == 0.75
    assert product_score(0.8, ev) == 0.8 * 0.75
    assert product_score(0.8, ev, alpha=0) == 0.8
