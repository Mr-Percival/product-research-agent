from pra.reviews import AspectMention, ReviewAspects, aggregate_by_review, is_verbatim, keep_verbatim


def m(aspect, pol, ev="x"):
    return AspectMention(aspect=aspect, detail="d", polarity=pol, evidence=ev)


def test_verbatim_tolerates_case_space_quotes():
    assert is_verbatim("It’s  QUIET", "Honestly it's quiet at night")
    assert not is_verbatim("very quiet", "Honestly it's quiet at night")
    assert not is_verbatim("", "anything")


def test_keep_verbatim_drops_paraphrase():
    ra = ReviewAspects(mentions=[m("noise", "positive", "so quiet"), m("price_value", "negative", "too pricey")])
    kept, dropped = keep_verbatim(ra, "So quiet! Worth every penny.")
    assert [k.aspect for k in kept] == ["noise"] and dropped == 1


def test_aggregate_counts_once_per_review():
    t = aggregate_by_review([[m("noise", "negative"), m("noise", "negative")], [m("noise", "positive")]])
    assert t["noise"] == {"positive": 1, "negative": 1, "neutral": 0}


def test_aggregate_conflicting_polarity_is_neutral():
    t = aggregate_by_review([[m("noise", "negative"), m("noise", "positive")]])
    assert t["noise"]["neutral"] == 1
