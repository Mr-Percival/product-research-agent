"""Story 1-2: retrieval and MMR benchmark on the frozen ESCI subset (run prepare_retrieval.py first).

Settings
  A  re-rank: order only the products judged for each query (ESCI Task 1 setup).
  B  retrieve: search the pooled corpus of ~14K products; unjudged hits count as gain 0 (judged@10 reported).
Methods: BM25, bge-small, BLaIR, hybrid RRF(BM25, bge-small), and two cross-encoders re-ranking hybrid's top-50.
MMR: lambda grid on bge-small top-50; lambda is chosen on the 200 dev queries by alpha-nDCG@10 (brand subtopics),
then reported once on the 500 test queries.

Writes results/retrieval_per_query.csv, results/retrieval_summary.json and results/retrieval.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pra.embeddings import BgeEncoder, BlairEncoder, default_device, product_text
from pra.metrics import (
    ESCI_GAIN,
    alpha_ndcg_at_k,
    bootstrap_ci,
    distinct_at_k,
    intra_list_diversity,
    judged_at_k,
    ndcg_at_k,
    recall_at_k,
)
from pra.retrieval import BM25Index, mmr, rrf

D = Path("data/retrieval")
R = Path("results")
SEED = 42
FIRST_STAGE_K = 100
RERANK_K = 50
CROSS_ENCODERS = {
    "ce-minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ce-bge-reranker": "BAAI/bge-reranker-base",
}
LAMBDAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def encode_cached(enc, texts: list[str], tag: str) -> np.ndarray:
    path = D / f"emb_{enc.name}_{tag}_{len(texts)}.npy"
    if path.exists():
        return np.load(path)
    v = enc.docs(texts) if tag == "docs" else enc.queries(texts)
    np.save(path, v)
    return v


def main() -> None:
    torch.manual_seed(SEED)
    R.mkdir(exist_ok=True)
    queries = pd.read_parquet(D / "queries.parquet").reset_index(drop=True)
    qrels = pd.read_parquet(D / "qrels.parquet")
    corpus = pd.read_parquet(D / "corpus.parquet").reset_index(drop=True)
    corpus["text"] = [
        product_text(r.product_title, r.product_brand, r.product_bullet_point) for r in corpus.itertuples()
    ]
    ids = corpus.product_id.tolist()
    pos = {p: i for i, p in enumerate(ids)}
    text = dict(zip(ids, corpus.text, strict=True))
    brand = {
        p: b.strip().lower() for p, b in zip(ids, corpus.product_brand, strict=True) if isinstance(b, str) and b.strip()
    }
    g = {q: dict(zip(x.product_id, x.esci_label.map(ESCI_GAIN), strict=True)) for q, x in qrels.groupby("query_id")}
    exact = {q: {p for p, v in gq.items() if v == 1.0} for q, gq in g.items()}
    # alpha-nDCG subtopics: each Exact product covers its brand (unbranded products are their own subtopic)
    subtopics = {q: {p: brand.get(p, p) for p in ex} for q, ex in exact.items()}

    t0 = time.time()
    q_texts = queries["query"].tolist()
    dense = {}
    for enc in (BgeEncoder(), BlairEncoder()):
        dense[enc.name] = (encode_cached(enc, corpus.text.tolist(), "docs"), encode_cached(enc, q_texts, "queries"))
        del enc
        torch.cuda.empty_cache()
    bm25 = BM25Index(ids, corpus.text.tolist())
    print(f"encoded + indexed in {time.time() - t0:.0f}s on {default_device()}")

    def dense_rank(name: str, qi: int, cand: np.ndarray, k: int) -> list[str]:
        dv, qv = dense[name]
        sims = dv[cand] @ qv[qi]
        return [ids[i] for i in cand[np.argsort(-sims, kind="stable")[:k]]]

    # ---- first stage, both settings
    rankings: dict[tuple[str, str], dict[int, list[str]]] = {}
    all_idx = np.arange(len(ids))
    for qi, (qid, q) in enumerate(zip(queries.query_id, q_texts, strict=True)):
        judged_idx = np.array([pos[p] for p in g[qid] if p in pos])
        for setting, cand, k in (("A", judged_idx, len(judged_idx)), ("B", all_idx, FIRST_STAGE_K)):
            bm = [p for p, _ in bm25.search(q, k=k, candidates=cand)]
            bg = dense_rank("bge-small", qi, cand, k)
            rankings.setdefault((setting, "bm25"), {})[qid] = bm
            rankings.setdefault((setting, "bge-small"), {})[qid] = bg
            rankings.setdefault((setting, "blair-base"), {})[qid] = dense_rank("blair-base", qi, cand, k)
            rankings.setdefault((setting, "hybrid-rrf"), {})[qid] = rrf([bm, bg], top=k)
    print(f"first stage done {time.time() - t0:.0f}s")

    # ---- cross-encoders re-rank hybrid (A: all judged; B: top-50)
    from sentence_transformers import CrossEncoder

    # (setting, first-stage source, suffix): A re-ranks all judged products; B re-ranks the top-50 of a first stage
    ce_runs = [("A", "hybrid-rrf", ""), ("B", "hybrid-rrf", ""), ("B", "bge-small", "@bge")]
    for name, model_id in CROSS_ENCODERS.items():
        # Scores are cached per (query, product) so re-runs and new first stages only score unseen pairs.
        cache_path = D / f"ce_{name}.parquet"
        cache: dict[tuple[int, str], float] = {}
        if cache_path.exists():
            c = pd.read_parquet(cache_path)
            cache = dict(zip(zip(c.query_id, c.product_id, strict=True), c.score, strict=True))
        qtext = dict(zip(queries.query_id, q_texts, strict=True))
        needed = {
            (qid, p)
            for setting, src, _ in ce_runs
            for qid, r in rankings[(setting, src)].items()
            for p in (r if setting == "A" else r[:RERANK_K])
        } - cache.keys()
        if needed:
            ce = CrossEncoder(model_id, max_length=256, device=default_device())
            todo = sorted(needed)
            scores = ce.predict([(qtext[q], text[p]) for q, p in todo], batch_size=64, show_progress_bar=False)
            cache.update({k: float(s) for k, s in zip(todo, scores, strict=True)})
            del ce
            torch.cuda.empty_cache()
            pd.DataFrame(
                [(q, p, s) for (q, p), s in cache.items()], columns=["query_id", "product_id", "score"]
            ).to_parquet(cache_path, index=False)
        for setting, src, suffix in ce_runs:
            rankings[(setting, name + suffix)] = {
                qid: sorted(r if setting == "A" else r[:RERANK_K], key=lambda p, q=qid: (-cache[(q, p)], p))
                for qid, r in rankings[(setting, src)].items()
            }
        print(f"{name} done {time.time() - t0:.0f}s ({len(needed)} new pairs scored)")

    # ---- MMR on bge-small top-50 in setting B
    dv, qv = dense["bge-small"]
    for lam in LAMBDAS:
        out = {}
        for qi, qid in enumerate(queries.query_id):
            top = np.argsort(-(dv @ qv[qi]), kind="stable")[:RERANK_K]
            out[qid] = [ids[i] for i in top[mmr(qv[qi], dv[top], k=10, lam=lam)]]
        rankings[("B", f"mmr-{lam}")] = out

    # ---- per-query metrics
    split = dict(zip(queries.query_id, queries.split, strict=True))
    rows = []
    for (setting, method), per_q in rankings.items():
        for qid, ranked in per_q.items():
            top10 = ranked[:10]
            rows.append(
                {
                    "query_id": qid,
                    "split": split[qid],
                    "setting": setting,
                    "method": method,
                    "ndcg10": ndcg_at_k(ranked, g[qid]),
                    "recall10": recall_at_k(ranked, exact[qid]),
                    "judged10": judged_at_k(ranked, set(g[qid])),
                    "alpha_ndcg10": alpha_ndcg_at_k(ranked, subtopics[qid]),
                    "brands10": distinct_at_k(ranked, brand),
                    "ild10": intra_list_diversity(dv[[pos[p] for p in top10]]),
                }
            )
    pq = pd.DataFrame(rows)
    pq.round(5).to_csv(R / "retrieval_per_query.csv", index=False)

    # ---- choose lambda on dev, report on test
    mmr_dev = (
        pq[(pq.split == "dev") & pq.method.str.startswith("mmr-")]
        .groupby("method")[["ndcg10", "alpha_ndcg10", "brands10", "ild10"]]
        .mean()
    )
    chosen = mmr_dev.alpha_ndcg10.idxmax()

    test = pq[pq.split == "test"]
    summary: dict = {"n_test_queries": int(test.query_id.nunique()), "n_dev_queries": int(len(queries) - 500)}
    table = []
    for (setting, method), x in test.groupby(["setting", "method"], sort=False):
        row = {"setting": setting, "method": method}
        for m in ["ndcg10", "recall10", "judged10", "alpha_ndcg10", "brands10", "ild10"]:
            mean, lo, hi = bootstrap_ci(x[m].to_numpy(), seed=SEED)
            row[m] = mean
            row[m + "_ci"] = [lo, hi]
        table.append(row)
    summary["test"] = table

    def paired(setting: str, a: str, b: str, metric: str = "ndcg10") -> dict:
        xa = test[(test.setting == setting) & (test.method == a)].set_index("query_id")[metric]
        xb = test[(test.setting == setting) & (test.method == b)].set_index("query_id")[metric]
        diff = (xa - xb.loc[xa.index]).to_numpy()
        mean, lo, hi = bootstrap_ci(diff, seed=SEED)
        return {"a": a, "b": b, "setting": setting, "metric": metric, "diff": mean, "ci": [lo, hi]}

    comps = []
    for s in ("A", "B"):
        for m in ["bge-small", "blair-base", "hybrid-rrf", "ce-minilm", "ce-bge-reranker"]:
            comps.append(paired(s, m, "bm25"))
        comps.append(paired(s, "ce-bge-reranker", "hybrid-rrf"))
        comps.append(paired(s, "blair-base", "bge-small"))
    for m in ["ce-minilm@bge", "ce-bge-reranker@bge"]:
        comps.append(paired("B", m, "bm25"))
    comps.append(paired("B", "ce-bge-reranker@bge", "ce-bge-reranker"))
    for metric in ["alpha_ndcg10", "ndcg10", "brands10"]:
        comps.append(paired("B", chosen, "mmr-1.0", metric))
    summary["paired_vs"] = comps
    summary["mmr_dev"] = mmr_dev.round(4).reset_index().to_dict(orient="records")
    summary["mmr_chosen"] = chosen
    (R / "retrieval_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- markdown report
    def fmt(r, m, d=3):
        lo, hi = r[m + "_ci"]
        return f"{r[m]:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"

    md = ["# Retrieval benchmark (ESCI, US, small_version)", ""]
    md += [f"{summary['n_test_queries']} test queries, 95% bootstrap CIs over queries (10,000 resamples, seed 42).", ""]
    md += ["| Setting | Method | nDCG@10 | Recall@10 (E) | Judged@10 |", "|---|---|---|---|---|"]
    for r in table:
        if not r["method"].startswith("mmr-"):
            md.append(
                f"| {r['setting']} | {r['method']} | {fmt(r, 'ndcg10')} | {fmt(r, 'recall10')} | {r['judged10']:.2f} |"
            )
    md += ["", "Paired differences in nDCG@10 (a - b):", ""]
    md += ["| Setting | a | b | Δ | 95% CI |", "|---|---|---|---|---|"]
    for c in comps:
        if c["metric"] == "ndcg10" and not c["a"].startswith("mmr"):
            ci = f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
            md.append(f"| {c['setting']} | {c['a']} | {c['b']} | {c['diff']:+.3f} | {ci} |")
    md += ["", f"## MMR (bge-small top-{RERANK_K}, setting B); λ chosen on dev by α-nDCG@10: **{chosen}**", ""]
    md += ["Dev (200 queries):", "", mmr_dev.round(3).to_markdown(), "", "Test:", ""]
    md += ["| λ | nDCG@10 | α-nDCG@10 | Brands@10 | ILD@10 |", "|---|---|---|---|---|"]
    for r in table:
        if r["method"].startswith("mmr-"):
            md.append(
                f"| {r['method'][4:]} | {fmt(r, 'ndcg10')} | {fmt(r, 'alpha_ndcg10')} | "
                f"{r['brands10']:.2f} | {r['ild10']:.3f} |"
            )
    (R / "retrieval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
