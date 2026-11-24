# product-research-agent

> **Work in progress.** Retrieval and diversity are measured and reproducible. Review mining, transparent scoring, the MCP server and the agent are under construction.

A product-research agent backed by an MCP server. It ranks products for a shopping query with dense retrieval and a cross-encoder. It mines feature mentions and their polarity from customer reviews with Claude's structured outputs. It combines both into a transparent score that you can recompute by hand. Every component is evaluated against public ground truth, with confidence intervals.

## Results so far: retrieval (Amazon ESCI)

The benchmark uses the [Shopping Queries Dataset](https://github.com/amazon-science/esci-data) (ESCI), US locale, `small_version`:

- **Test set:** 500 queries from the `test` split, fixed with seed 42.
- **Dev set:** 200 queries from the `train` split. It is used only to tune hyper-parameters, never the test queries.
- **Gains:** follow KDD Cup 2022: Exact = 1, Substitute = 0.1, Complement = 0.01, Irrelevant = 0.
- **Confidence intervals:** 95% paired bootstrap over queries.

**Setting B (retrieval):** each query searches a pooled corpus of 13,956 products. Unjudged hits count as 0, so these numbers are lower bounds (judged@10 is about 0.75).

| System | nDCG@10 | Δ vs BM25 [95% CI] |
|---|---|---|
| BM25 | 0.521 | — |
| BLaIR-base (domain model, Amazon reviews) | 0.478 | −0.043 [−0.073, −0.013] |
| Hybrid RRF (BM25 + bge-small) | 0.580 | +0.059 [+0.047, +0.071] |
| bge-small-en-v1.5 | 0.597 | +0.077 [+0.057, +0.096] |
| RRF top-50 → ms-marco-MiniLM cross-encoder | 0.627 | +0.106 [+0.089, +0.124] |
| RRF top-50 → bge-reranker-base | 0.639 | +0.118 [+0.099, +0.137] |
| **bge-small top-50 → bge-reranker-base** | **0.641** | **+0.121 [+0.102, +0.140]** |

Both second-stage variants are statistically tied (Δ +0.002, CI [−0.003, +0.007]). The served pipeline uses the simpler one: bge-small top-50 followed by bge-reranker-base, with no BM25 index needed. Re-running the benchmark reproduces every number exactly.

**Setting A (re-rank the judged products, ESCI Task 1):** bge-reranker-base scores 0.758 and BM25 scores 0.692, a gain of +0.066 [+0.052, +0.081].

Findings worth knowing:

- **A domain-pretrained encoder is not automatically better.** BLaIR is pretrained on Amazon Reviews 2023, yet it trails the general-purpose bge-small by 0.119 nDCG@10 on these short queries.
- **Naive hybrid fusion hurts here.** Fusing BM25 into bge-small with RRF lowers nDCG, and it adds nothing once a cross-encoder re-ranks the list. The whole gain comes from the second-stage cross-encoder.
- **MMR does not pay off on ESCI.** λ was tuned on dev with α-nDCG@10, using brands as subtopics, and the selected value was λ = 1, which means no MMR. Lower λ raises the number of distinct brands in the top 10 from 6.6 to 8.9, but it costs relevance. ESCI queries are mostly specific ("x brand y model"), so they do not reward diversity. MMR is kept as an option for exploratory agent questions, and it is not claimed as a metric win.

Full tables: [`results/retrieval.md`](results/retrieval.md). Per-query scores: [`results/retrieval_per_query.csv`](results/retrieval_per_query.csv).

## Roadmap

1. ✅ Dense ranking vs BM25 with nDCG@10, recall@10 and judged@10, plus bootstrap CIs.
2. ✅ MMR λ sweep, with α-nDCG, brand coverage and intra-list diversity, λ chosen on dev.
3. ⏳ Review mining with Claude structured outputs on Amazon Reviews 2023. It will report precision and recall against 200 hand-labelled reviews, and the cost per 1,000 reviews (about $0.15 measured on the prototype).
4. ⏳ Transparent scoring (relevance × review evidence), validated against held-out ratings and human pairwise preferences, with the method written up in `METHODOLOGY.md`.
5. ⏳ MCP server with `search_products`, `feature_mentions` and `rank_for_intent`.
6. ⏳ LangGraph agent that answers multi-step shopping questions with cited review evidence.

## Reproduce

Requires Python 3.11+. A CUDA GPU is optional (the benchmark took about 40 minutes on an RTX 3050 with 4 GB).

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -e ".[dev]"
# Download from esci-data into data/:
#   shopping_queries_dataset_examples.parquet
#   shopping_queries_dataset_products.parquet
python scripts/prepare_retrieval.py
python scripts/eval_retrieval.py
pytest
```

`notebooks/01_prototype.ipynb` is the first prototype: 20 queries plus a single-query review-mining demo. Its MMR and scoring numbers are superseded by the results above.

## Data and model licenses

- **Data:** ESCI is Apache-2.0. Amazon Reviews 2023 (McAuley Lab) is for research use.
- **Models:** bge-small-en-v1.5 and bge-reranker-base are MIT. BLaIR is MIT. ms-marco-MiniLM-L-6-v2 is Apache-2.0.
- **This code:** Apache-2.0.
