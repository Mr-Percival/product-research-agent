"""Freeze the retrieval benchmark (story 1-2) from raw ESCI.

* test: 500 queries from the ESCI `test` split (US, small_version), seed 42 -- the same pool as the prototype.
* dev:  200 queries from the ESCI `train` split, seed 42 -- used only to pick hyper-parameters (MMR lambda),
        so nothing is tuned on test queries.
* corpus: every US product judged for any dev or test query (pooled corpus, ~14K products).

Outputs data/retrieval/{queries,qrels,corpus}.parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path("data")
OUT = RAW / "retrieval"
SEED = 42
N_TEST = 500
N_DEV = 200


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ex = pd.read_parquet(RAW / "shopping_queries_dataset_examples.parquet")
    ex = ex[(ex.product_locale == "us") & (ex.small_version == 1)]

    test_ids = ex[ex.split == "test"].query_id.drop_duplicates().sample(N_TEST, random_state=SEED)
    dev_ids = ex[ex.split == "train"].query_id.drop_duplicates().sample(N_DEV, random_state=SEED)
    split = {**dict.fromkeys(test_ids, "test"), **dict.fromkeys(dev_ids, "dev")}

    qrels = ex[ex.query_id.isin(split)][["query_id", "query", "product_id", "esci_label"]].copy()
    qrels["split"] = qrels.query_id.map(split)
    queries = qrels[["query_id", "query", "split"]].drop_duplicates("query_id").sort_values(["split", "query_id"])

    products = pd.read_parquet(
        RAW / "shopping_queries_dataset_products.parquet",
        columns=["product_id", "product_locale", "product_title", "product_bullet_point", "product_brand"],
    )
    products = products[products.product_locale == "us"].drop(columns="product_locale")
    corpus = products[products.product_id.isin(qrels.product_id)].drop_duplicates("product_id")

    queries.to_parquet(OUT / "queries.parquet", index=False)
    qrels.to_parquet(OUT / "qrels.parquet", index=False)
    corpus.to_parquet(OUT / "corpus.parquet", index=False)
    print(queries.split.value_counts().to_dict(), "| qrels", len(qrels), "| corpus", len(corpus))
    print(qrels.groupby("split").esci_label.value_counts(normalize=True).round(3).unstack())


if __name__ == "__main__":
    main()
