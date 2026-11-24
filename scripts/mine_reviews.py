"""Story 3: closed-taxonomy aspect extraction over a set of reviews.

usage: python scripts/mine_reviews.py <reviews.parquet> <out_mentions.parquet> [--limit N]

Each review is sent once (results are cached in the output file, so re-runs only process new reviews).
Mentions whose evidence is not verbatim in the review are dropped and counted; the drop rate is reported.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import pandas as pd
import pydantic
from dotenv import load_dotenv

from pra.reviews import extract_aspects, keep_verbatim

PRICE_IN, PRICE_OUT = 0.10, 0.50  # Haiku 5.5, USD per million tokens (Batch API halves both)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reviews")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    load_dotenv()

    reviews = pd.read_parquet(args.reviews)
    if args.limit:
        reviews = reviews.head(args.limit)
    out = Path(args.out)
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["review_id"])
    todo = reviews[~reviews.review_id.isin(set(done.review_id))]
    client = anthropic.Anthropic(max_retries=6)

    failed: list[tuple[str, str]] = []

    def run(r):
        text = f"{r.title}\n{r.text}"
        for max_tokens in (1024, 4096):  # a truncated JSON is retried once with more room
            try:
                ra, u = extract_aspects(client, text, max_tokens=max_tokens)
                kept, dropped = keep_verbatim(ra, text)
                return r.review_id, kept, dropped, u
            except (pydantic.ValidationError, anthropic.APIError) as e:
                err = f"{type(e).__name__}: {str(e)[:120]}"
        failed.append((r.review_id, err))
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        res = [x for x in pool.map(run, todo.itertuples()) if x is not None]  # failures are retried on re-run

    rows = []
    for rid, kept, dropped, _ in res:
        rows += [{"review_id": rid, **m.model_dump(), "dropped": 0} for m in kept]
        # a sentinel row keeps reviews with no surviving mention in the cache, and records the drops
        rows.append(
            {"review_id": rid, "aspect": None, "detail": None, "polarity": None, "evidence": None, "dropped": dropped}
        )
    new = pd.DataFrame(rows)
    allm = pd.concat([done, new], ignore_index=True) if len(done) else new
    allm.to_parquet(out, index=False)

    usage = {k: sum(u[k] for *_, u in res) for k in ("input_tokens", "output_tokens")}
    cost = usage["input_tokens"] / 1e6 * PRICE_IN + usage["output_tokens"] / 1e6 * PRICE_OUT
    m = allm[allm.aspect.notna()]
    report = {
        "reviews_processed_now": len(res),
        "failed_now": len(failed),
        "usd_now": round(cost, 4),
        "usd_per_1000_reviews": round(cost / max(len(res), 1) * 1000, 4),
        "mentions_kept": int(len(m)),
        "mentions_dropped_not_verbatim": int(allm.dropped.sum()),
        "drop_rate": round(float(allm.dropped.sum() / max(len(m) + allm.dropped.sum(), 1)), 4),
        "reviews_with_any_aspect": round(m.review_id.nunique() / allm.review_id.nunique(), 3),
        "aspect_share": m.aspect.value_counts(normalize=True).round(3).head(20).to_dict(),
    }
    print(json.dumps(report, indent=1))
    for rid, err in failed[:5]:
        print("FAILED", rid, err)
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
