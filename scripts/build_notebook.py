"""Generates notebooks/01_prototype.ipynb (kept as code so the notebook diff stays reviewable)."""

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md("""# Prototype: dense retrieval, MMR and review evidence on Amazon ESCI

Goal: decide whether the scoring design (relevance x review evidence) is worth building out.

* **Data**: frozen subset of the Shopping Queries Dataset (ESCI, US, `small_version`, test split): 500-query pool, 20 eval queries, seed 42 (`scripts/prepare_prototype.py`).
* **Gains**: KDD Cup 2022 convention, E=1, S=0.1, C=0.01, I=0. Recall@10 counts only Exact products.
* **Two settings**: (A) *re-rank* the products judged for each query, the standard ESCI Task 1 setup; (B) *retrieve* from the pooled corpus of every product judged for the 500 queries (unjudged hits count as 0, so this underestimates absolute quality).
* **Models**: BM25, `BAAI/bge-small-en-v1.5` (general MTEB model) and `hyp1231/blair-roberta-base` (BLaIR, pretrained on Amazon Reviews 2023).""")

code("""import json, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from pra.metrics import ESCI_GAIN, ndcg_at_k, recall_at_k, intra_list_diversity
from pra.retrieval import BM25Index, mmr

load_dotenv(Path('..') / '.env')
P = Path('../data/proto')
np.random.seed(42); torch.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
queries = pd.read_parquet(P / 'queries.parquet')
qrels = pd.read_parquet(P / 'qrels.parquet')
corpus = pd.read_parquet(P / 'corpus.parquet').reset_index(drop=True)
print(DEVICE, len(queries), 'eval queries |', qrels.query_id.nunique(), 'pool queries |', len(corpus), 'products')
print(json.loads((P / 'overlap.json').read_text()))""")

code("""def _s(x):
    return x if isinstance(x, str) else ''

def product_text(r):
    return f"{_s(r.product_title)}. Brand: {_s(r.product_brand)}. {_s(r.product_bullet_point)[:600]}"

corpus['text'] = [product_text(r) for r in corpus.itertuples()]
pid_to_pos = {p: i for i, p in enumerate(corpus.product_id)}
gains = {q: {p: ESCI_GAIN[l] for p, l in zip(g.product_id, g.esci_label)} for q, g in qrels.groupby('query_id')}
exact = {q: {p for p, l in zip(g.product_id, g.esci_label) if l == 'E'} for q, g in qrels.groupby('query_id')}""")

code("""# Embeddings
bge = SentenceTransformer('BAAI/bge-small-en-v1.5', device=DEVICE)
BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

blair_tok = AutoTokenizer.from_pretrained('hyp1231/blair-roberta-base')
blair = AutoModel.from_pretrained('hyp1231/blair-roberta-base').to(DEVICE).eval()

@torch.no_grad()
def blair_encode(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = blair_tok(texts[i:i+bs], padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
        v = blair(**enc).last_hidden_state[:, 0]  # first-token embedding, as in the model card
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(out)

t = time.time()
doc_vecs = {
    'bge-small': bge.encode(corpus.text.tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False),
    'blair-base': blair_encode(corpus.text.tolist()),
}
q_texts = queries['query'].tolist()
q_vecs = {
    'bge-small': bge.encode([BGE_QUERY_PREFIX + q for q in q_texts], normalize_embeddings=True),
    'blair-base': blair_encode(q_texts),
}
print(f'encoded in {time.time()-t:.0f}s')
bm25 = BM25Index(corpus.product_id.tolist(), corpus.text.tolist())""")

code("""def rank(method, qi, query, candidates=None, k=10):
    if method == 'bm25':
        return [p for p, _ in bm25.search(query, k=k, candidates=candidates)]
    D = doc_vecs[method]
    idx = np.asarray(candidates) if candidates is not None else np.arange(len(D))
    sims = D[idx] @ q_vecs[method][qi]
    return [corpus.product_id[i] for i in idx[np.argsort(-sims, kind='stable')[:k]]]

rows = []
for setting in ['A: re-rank judged', 'B: pooled retrieval']:
    for method in ['bm25', 'bge-small', 'blair-base']:
        nd, rc = [], []
        for qi, (qid, q) in enumerate(zip(queries.query_id, queries['query'])):
            cands = [pid_to_pos[p] for p in gains[qid] if p in pid_to_pos] if setting.startswith('A') else None
            r = rank(method, qi, q, cands)
            nd.append(ndcg_at_k(r, gains[qid])); rc.append(recall_at_k(r, exact[qid]))
        rows.append({'setting': setting, 'method': method, 'nDCG@10': np.mean(nd), 'recall@10 (E)': np.mean(rc)})
results = pd.DataFrame(rows).round(3)
results""")

md("""## MMR re-ranking
Take the top-50 by cosine from the best dense model in setting B, then re-rank with MMR. `lambda = 1` is plain cosine. Diversity = 1 - mean pairwise cosine of the top-10.""")

code("""best = results[results.setting.str.startswith('B') & (results.method != 'bm25')].sort_values('nDCG@10').method.iloc[-1]
D = doc_vecs[best]
rows = []
for lam in [1.0, 0.85, 0.7, 0.5]:
    nd, div = [], []
    for qi, qid in enumerate(queries.query_id):
        qv = q_vecs[best][qi]
        top = np.argsort(-(D @ qv), kind='stable')[:50]
        order = top[mmr(qv, D[top], k=10, lam=lam)]
        nd.append(ndcg_at_k([corpus.product_id[i] for i in order], gains[qid]))
        div.append(intra_list_diversity(D[order]))
    rows.append({'model': best, 'lambda': lam, 'nDCG@10': np.mean(nd), 'diversity@10': np.mean(div)})
mmr_table = pd.DataFrame(rows).round(3)
mmr_table""")

md("""## Review evidence (Claude Haiku 5.5, structured outputs)
Demo query: the ESCI test query with the most Exact products that also have reviews in Amazon Reviews 2023 (Appliances). Up to 20 reviews per product. Requires `ANTHROPIC_API_KEY` in `.env`; otherwise these cells are skipped.""")

code("""from pra.reviews import extract_features, aggregate
demo = json.loads((P / 'demo.json').read_text())
demo_reviews = pd.read_parquet(P / 'demo_reviews.parquet')
print(demo['query'], '|', len(demo['products']), 'products |', len(demo_reviews), 'reviews')
HAVE_KEY = bool(os.getenv('ANTHROPIC_API_KEY'))
mined = {}
usage = {'input_tokens': 0, 'output_tokens': 0}
if HAVE_KEY:
    import anthropic
    client = anthropic.Anthropic()
    for pid, g in demo_reviews.groupby('product_id'):
        per = []
        for r in g.itertuples():
            rf, u = extract_features(client, f"{r.title}\\n{r.text}")
            per.append(rf); usage = {k: usage[k] + u[k] for k in usage}
        mined[pid] = aggregate(per)
    # Haiku 5.5 list price: $0.10 / 1M input, $0.50 / 1M output (Batch API halves it)
    cost = usage['input_tokens'] / 1e6 * 0.10 + usage['output_tokens'] / 1e6 * 0.50
    print(usage, f'cost ${cost:.4f} -> ${cost / len(demo_reviews) * 1000:.3f} per 1,000 reviews')
else:
    print('No ANTHROPIC_API_KEY: skipping extraction')""")

code("""if mined:
    top_pid = max(mined, key=lambda p: sum(sum(c.values()) for c in mined[p].values()))
    tbl = pd.DataFrame(mined[top_pid]).T
    tbl['mentions'] = tbl.sum(axis=1)
    display(tbl.sort_values('mentions', ascending=False).head(15))""")

md("""## Transparent score and ablation
`evidence = (pos + 1) / (pos + neg + 2)` over mentions whose feature is semantically close (cosine >= 0.6, bge-small) to a word in the query; if none match, all mentions are used. `score = relevance x evidence**alpha`; `alpha = 0` is the no-reviews ablation.""")

code("""from pra.scoring import evidence_score, product_score
if mined:
    qv = bge.encode([BGE_QUERY_PREFIX + demo['query']], normalize_embeddings=True)[0]
    q_terms = [t for t in demo['query'].split() if len(t) > 2]
    term_vecs = bge.encode(q_terms, normalize_embeddings=True)
    dp = pd.read_parquet(P / 'demo_products.parquet')
    dp_vecs = dict(zip(dp.product_id, bge.encode([product_text(r) for r in dp.itertuples()], normalize_embeddings=True)))
    rows = []
    for pid in demo['products']:
        if pid not in dp_vecs or pid not in mined:
            continue
        rel = float(dp_vecs[pid] @ qv)
        feats = list(mined[pid])
        fv = bge.encode(feats, normalize_embeddings=True)
        close = [f for f, v in zip(feats, fv) if (term_vecs @ v).max() >= 0.6] or feats
        pos = sum(mined[pid][f]['positive'] for f in close); neg = sum(mined[pid][f]['negative'] for f in close)
        ev = evidence_score({'positive': pos, 'negative': neg})
        rows.append({'product_id': pid, 'label': demo['labels'][pid], 'relevance': rel, 'pos': pos, 'neg': neg,
                     'evidence': ev, 'score_a1': product_score(rel, ev, 1), 'score_a0': product_score(rel, ev, 0),
                     'matched_features': ', '.join(close[:5])})
    demo_tbl = pd.DataFrame(rows).round(3)
    g = {p: ESCI_GAIN[l] for p, l in demo['labels'].items()}
    for col in ['score_a0', 'score_a1']:
        print(col, 'nDCG@10 =', round(ndcg_at_k(demo_tbl.sort_values(col, ascending=False).product_id.tolist(), g), 3))
    r = demo_tbl.iloc[0]
    print(f"By hand, {r.product_id}: ({r.pos}+1)/({r.pos}+{r.neg}+2) = {(r.pos+1)/(r.pos+r.neg+2):.3f}; "
          f"{r.relevance:.3f} x {r.evidence:.3f} = {r.relevance*r.evidence:.3f}")
    display(demo_tbl)""")

md("""## Reading the results
* Setting A is comparable in kind to the ESCI paper's Task 1 baselines; setting B is a harder, pooled-retrieval view.
* 20 queries is a smoke test, not a benchmark: differences under ~0.03 nDCG are noise. Story 1 freezes 500 queries.
* The review demo rests on one query, so the ablation shows the mechanics, not a general effect.""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/01_prototype.ipynb")
print("written")
