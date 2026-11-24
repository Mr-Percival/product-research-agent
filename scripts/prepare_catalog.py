"""Freeze the review catalog for stories 3-6 from Amazon Reviews 2023 (Appliances).

Scope: whole appliances (not spare parts) in a few leaf categories where reviews talk about
comparable aspects (noise, capacity, ease of cleaning, durability, ...), with >= MIN_REVIEWS reviews.

Per product, reviews are split by time: the older half is the *mining* half (what the system may read),
the newer half is *held out* (its ratings and text are ground truth for the scoring evaluation, never mined).

Outputs data/catalog/:
  products.parquet         parent_asin, title, brand, category, price, features, n_reviews
  reviews.parquet          all reviews of those products with a `half` column (mine | holdout)
  label_sample.parquet     200 mining-half reviews for hand labelling (40 per star rating), frozen test set
  taxonomy_sample.parquet  1,000 other mining-half reviews used to discover the aspect taxonomy
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

RAW = Path("data")
OUT = RAW / "catalog"
SEED = 42
MIN_REVIEWS = 30
LEAVES = {
    "Ice Makers",
    "Washers & Dryers",
    "Range Hoods",
    "Refrigerators",
    "Freezers",
    "Beverage Refrigerators",
    "Countertop Dishwashers",
    "Built-In Dishwashers",
}


def _price(x) -> float | None:
    try:
        return float(str(x).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    with gzip.open(RAW / "meta_Appliances.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            cats = m.get("categories") or []
            if len(cats) >= 3 and cats[0] == "Appliances" and cats[2] in LEAVES:
                meta[m["parent_asin"]] = {
                    "parent_asin": m["parent_asin"],
                    "title": m.get("title"),
                    "brand": (m.get("details") or {}).get("Brand") or m.get("store"),
                    "category": cats[2],
                    "price": _price(m.get("price")),
                    "features": " | ".join(m.get("features") or [])[:1500],
                }

    rows = []
    with gzip.open(RAW / "Appliances.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["parent_asin"] in meta:
                rows.append(
                    {
                        "review_id": f"{r['parent_asin']}:{r['user_id']}:{r['timestamp']}",
                        "parent_asin": r["parent_asin"],
                        "rating": r["rating"],
                        "title": r["title"],
                        "text": r["text"],
                        "timestamp": r["timestamp"],
                        "verified": r["verified_purchase"],
                        "helpful_vote": r["helpful_vote"],
                    }
                )
    reviews = pd.DataFrame(rows).drop_duplicates("review_id")
    counts = reviews.parent_asin.value_counts()
    keep = counts[counts >= MIN_REVIEWS].index
    reviews = reviews[reviews.parent_asin.isin(keep)].sort_values(["parent_asin", "timestamp", "review_id"])
    rank = reviews.groupby("parent_asin").cumcount()
    size = reviews.parent_asin.map(counts)
    reviews["half"] = (rank < size // 2).map({True: "mine", False: "holdout"})

    products = pd.DataFrame([meta[p] for p in keep])
    products["n_reviews"] = products.parent_asin.map(counts)

    mine = reviews[(reviews.half == "mine") & reviews.text.str.len().between(40, 1500)]
    label = mine.groupby(mine.rating.round().astype(int), group_keys=False).sample(40, random_state=SEED)
    taxonomy = mine.drop(label.index).sample(1000, random_state=SEED)

    products.to_parquet(OUT / "products.parquet", index=False)
    reviews.to_parquet(OUT / "reviews.parquet", index=False)
    label.to_parquet(OUT / "label_sample.parquet", index=False)
    taxonomy.to_parquet(OUT / "taxonomy_sample.parquet", index=False)
    print(len(products), "products |", len(reviews), "reviews |", (reviews.half == "mine").sum(), "mining half")
    print(products.category.value_counts().to_dict())
    print("with price:", products.price.notna().mean().round(3), "| label", len(label), "| taxonomy", len(taxonomy))


if __name__ == "__main__":
    main()
