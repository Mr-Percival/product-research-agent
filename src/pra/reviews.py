"""Feature-mention extraction from product reviews with Claude structured outputs."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from pra.taxonomy import ASPECTS

DEFAULT_MODEL = "claude-haiku-5-5"


class FeatureMention(BaseModel):
    feature: str = Field(description="Product feature, lowercase noun phrase, e.g. 'noise', 'ice crushing'")
    polarity: Literal["positive", "negative", "neutral"]
    evidence: str = Field(description="Short verbatim quote from the review supporting the mention")


class ReviewFeatures(BaseModel):
    mentions: list[FeatureMention]


SYSTEM_PROMPT = (
    "You extract product feature mentions from a customer review. "
    "Only include features the review actually talks about. "
    "Normalise feature names to short lowercase noun phrases (e.g. 'noise', 'blade quality', 'price'). "
    "The evidence must be a verbatim substring of the review."
)


def extract_features(client, review_text: str, model: str = DEFAULT_MODEL) -> tuple[ReviewFeatures, dict]:
    """Returns parsed features and token usage. `client` is an anthropic.Anthropic instance."""
    resp = client.messages.parse(
        model=model,
        max_tokens=1024,
        output_config={"effort": "low"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": review_text}],
        output_format=ReviewFeatures,
    )
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return resp.parsed_output, usage


def aggregate(per_review: list[ReviewFeatures]) -> dict[str, dict[str, int]]:
    """feature -> {positive, negative, neutral} counts."""
    table: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
    for rf in per_review:
        for m in rf.mentions:
            table[m.feature][m.polarity] += 1
    return dict(table)


# ---------------------------------------------------------------- closed-taxonomy extraction (story 3)

Aspect = Literal[tuple(ASPECTS)]  # type: ignore[valid-type]


class AspectMention(BaseModel):
    aspect: Aspect = Field(description="One aspect from the taxonomy")
    detail: str = Field(description="What exactly is said, 2-6 lowercase words, e.g. 'loud compressor hum'")
    polarity: Literal["positive", "negative", "neutral"]
    evidence: str = Field(description="Shortest verbatim span of the review that supports the mention")


class ReviewAspects(BaseModel):
    mentions: list[AspectMention]


_ASPECT_LIST = "\n".join(f"- {k}: {v}" for k, v in ASPECTS.items())
ASPECT_SYSTEM_PROMPT = f"""You annotate customer reviews of home appliances with the aspects they discuss.
Aspects (use exactly these ids):
{_ASPECT_LIST}

Rules:
- Only aspects the review actually talks about; an empty list is fine.
- At most one mention per aspect; if the review is mixed on an aspect, use the dominant polarity.
- polarity is the reviewer's judgement of that aspect; neutral only for factual statements with no judgement.
- evidence is copied character for character from the review (no paraphrase, no ellipsis).
- Use 'overall' only for general satisfaction that names no specific aspect."""


def extract_aspects(
    client, review_text: str, model: str = DEFAULT_MODEL, max_tokens: int = 1024
) -> tuple[ReviewAspects, dict]:
    resp = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        output_config={"effort": "low"},
        system=ASPECT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": review_text}],
        output_format=ReviewAspects,
    )
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return resp.parsed_output, usage


_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.translate(_QUOTES).lower()).strip()


def is_verbatim(evidence: str, review_text: str) -> bool:
    """Evidence must appear in the review up to case, whitespace and curly quotes."""
    e = _norm(evidence)
    return bool(e) and e in _norm(review_text)


def keep_verbatim(ra: ReviewAspects, review_text: str) -> tuple[list[AspectMention], int]:
    """Drops mentions whose evidence is not in the review. Returns kept mentions and the number dropped."""
    kept = [m for m in ra.mentions if is_verbatim(m.evidence, review_text)]
    return kept, len(ra.mentions) - len(kept)


def aggregate_by_review(per_review: list[list[AspectMention]]) -> dict[str, dict[str, int]]:
    """aspect -> {positive, negative, neutral} counted once per review (repeats inside a review do not inflate)."""
    table: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
    for mentions in per_review:
        seen: dict[str, set[str]] = defaultdict(set)
        for m in mentions:
            seen[m.aspect].add(m.polarity)
        for aspect, pols in seen.items():
            pol = next(iter(pols)) if len(pols) == 1 else "neutral"  # contradictory polarities -> neutral
            table[aspect][pol] += 1
    return dict(table)
