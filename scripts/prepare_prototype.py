"""Build the frozen prototype subset from raw ESCI + Amazon Reviews 2023 (Appliances).

Outputs (data/proto/):
  queries.parquet   20 eval queries (seed 42) drawn from the 500-query pool
  qrels.parquet     ESCI judgements for the 500-query pool
  corpus.parquet    every product judged for the pool (pooled retrieval corpus)
  overlap.json      ESCI <-> Reviews 2023 ASIN overlap stats
  demo.json         one query + products that do have Appliances reviews, for the scoring demo
  demo_reviews.parquet  reviews of the demo products
  demo_products.parquet catalog text of the demo products
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

RAW = Path("data")
OUT = RAW / "proto"
SEED = 42
POOL_QUERIES = 500
EVAL_QUERIES = 20
MAX_REVIEWS_PER_PRODUCT = 20


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ex = pd.read_parquet(RAW / "shopping_queries_dataset_examples.parquet")
    ex = ex[(ex.product_locale == "us") & (ex.small_version == 1) & (ex.split == "test")]

    pool_ids = ex.query_id.drop_duplicates().sample(POOL_QUERIES, random_state=SEED)
    qrels = ex[ex.query_id.isin(pool_ids)][["query_id", "query", "product_id", "esci_label"]]
    eval_ids = pool_ids.sample(EVAL_QUERIES, random_state=SEED)
    queries = qrels[qrels.query_id.isin(eval_ids)][["query_id", "query"]].drop_duplicates()

    products = pd.read_parquet(
        RAW / "shopping_queries_dataset_products.parquet",
        columns=["product_id", "product_locale", "product_title", "product_bullet_point", "product_brand"],
    )
    products = products[products.product_locale == "us"].drop(columns="product_locale")
    corpus = products[products.product_id.isin(qrels.product_id)].drop_duplicates("product_id")

    # ESCI <-> Reviews 2023 overlap
    meta = {}
    with gzip.open(RAW / "meta_Appliances.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            meta[m["parent_asin"]] = {"title": m.get("title"), "rating_number": m.get("rating_number") or 0}
    us_all = set(ex.product_id)
    overlap_all = us_all & meta.keys()

    # Demo query: the ESCI test query with most Exact products that have Appliances reviews
    ex_ov = ex[ex.product_id.isin(overlap_all) & (ex.esci_label == "E")]
    demo_qid = ex_ov.groupby("query_id").size().sort_values(ascending=False).index[0]
    demo_rows = ex[ex.query_id == demo_qid]
    demo_products = [p for p in demo_rows.product_id if p in meta]

    reviews = []
    counts = dict.fromkeys(demo_products, 0)
    with gzip.open(RAW / "Appliances.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pa = r["parent_asin"]
            if pa in counts and counts[pa] < MAX_REVIEWS_PER_PRODUCT:
                counts[pa] += 1
                reviews.append({"product_id": pa, "rating": r["rating"], "title": r["title"], "text": r["text"]})

    overlap = {
        "esci_us_test_products": len(us_all),
        "appliances_items": len(meta),
        "overlap_products": len(overlap_all),
        "overlap_share": round(len(overlap_all) / len(us_all), 4),
        "pool_products_with_reviews": int(corpus.product_id.isin(meta.keys()).sum()),
        "pool_products": len(corpus),
    }
    demo = {
        "query_id": int(demo_qid),
        "query": demo_rows["query"].iloc[0],
        "products": demo_products,
        "labels": dict(zip(demo_rows.product_id, demo_rows.esci_label, strict=True)),
    }

    queries.to_parquet(OUT / "queries.parquet", index=False)
    qrels.to_parquet(OUT / "qrels.parquet", index=False)
    corpus.to_parquet(OUT / "corpus.parquet", index=False)
    pd.DataFrame(reviews).to_parquet(OUT / "demo_reviews.parquet", index=False)
    products[products.product_id.isin(demo_products)].drop_duplicates("product_id").to_parquet(
        OUT / "demo_products.parquet", index=False
    )
    (OUT / "overlap.json").write_text(json.dumps(overlap, indent=2))
    (OUT / "demo.json").write_text(json.dumps(demo, indent=2))
    print(json.dumps(overlap, indent=2))
    print("demo:", demo["query"], len(demo_products), "products,", len(reviews), "reviews")


if __name__ == "__main__":
    main()
