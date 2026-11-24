"""Story 3, step 1: discover the aspect taxonomy from data instead of inventing it.

1. Open-vocabulary extraction (Claude Haiku, structured outputs) on the 1,000-review taxonomy sample.
2. Embed every distinct feature string (bge-small) and cluster them (agglomerative, cosine).
3. Write the clusters, by review coverage, to results/taxonomy_clusters.md for a human to name.

The named taxonomy then becomes a closed enum in the extraction schema, which merges synonyms
("noise" / "loudness") by construction. The 200 hand-labelled reviews are not touched here.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering

from pra.embeddings import BgeEncoder
from pra.reviews import extract_features

C = Path("data/catalog")
R = Path("results")
PRICE_IN, PRICE_OUT = 0.10, 0.50  # Haiku 5.5, USD per million tokens
DISTANCE = 0.25  # cosine distance threshold for merging feature strings


def main() -> None:
    load_dotenv()
    sample = pd.read_parquet(C / "taxonomy_sample.parquet")
    cache = C / "taxonomy_mentions.parquet"
    if cache.exists():
        mentions = pd.read_parquet(cache)
    else:
        client = anthropic.Anthropic(max_retries=6)

        def run(r):
            rf, u = extract_features(client, f"{r.title}\n{r.text}")
            return r.review_id, rf, u

        with ThreadPoolExecutor(max_workers=8) as pool:
            out = list(pool.map(run, sample.itertuples()))
        usage = {k: sum(u[k] for _, _, u in out) for k in ("input_tokens", "output_tokens")}
        cost = usage["input_tokens"] / 1e6 * PRICE_IN + usage["output_tokens"] / 1e6 * PRICE_OUT
        print(usage, f"${cost:.3f} for {len(sample)} reviews")
        mentions = pd.DataFrame(
            [
                {"review_id": rid, "feature": m.feature.strip().lower(), "polarity": m.polarity, "evidence": m.evidence}
                for rid, rf, _ in out
                for m in rf.mentions
            ]
        )
        mentions.to_parquet(cache, index=False)
        (C / "taxonomy_usage.json").write_text(json.dumps({**usage, "usd": cost, "reviews": len(sample)}))

    feats = mentions.feature.value_counts()
    enc = BgeEncoder()
    vecs = enc.docs(feats.index.tolist())
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=DISTANCE, metric="cosine", linkage="average"
    ).fit_predict(vecs)
    fmap = dict(zip(feats.index, labels, strict=True))
    mentions["cluster"] = mentions.feature.map(fmap)
    cov = mentions.groupby("cluster").review_id.nunique().sort_values(ascending=False)

    lines = [
        "# Feature clusters from 1,000 reviews (open extraction)",
        "",
        f"{len(mentions)} mentions, {len(feats)} distinct strings, {len(cov)} clusters (cosine distance < {DISTANCE}).",
        "",
        "| Cluster | Reviews | Top strings |",
        "|---|---|---|",
    ]
    for c, n in cov.head(80).items():
        top = mentions[mentions.cluster == c].feature.value_counts().head(8)
        lines.append(f"| {c} | {n} | {', '.join(f'{k} ({v})' for k, v in top.items())} |")
    covered = mentions[mentions.cluster.isin(cov.head(40).index)].review_id.nunique()
    lines += ["", f"Top-40 clusters cover {covered / mentions.review_id.nunique():.1%} of reviews with a mention."]
    R.mkdir(exist_ok=True)
    (R / "taxonomy_clusters.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:60]))
    np.save(C / "taxonomy_feature_vecs.npy", vecs)


if __name__ == "__main__":
    main()
