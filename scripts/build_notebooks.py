"""Build the phase notebooks (02-07) from code, so they stay reproducible. Inspection only: $0, no API calls.

Run:  .venv/bin/python scripts/build_notebooks.py 02 03 04
      .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_indexing.ipynb
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = """import sys, json, pickle; sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', 70); pd.set_option('display.width', 200)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})"""


def md(s): return nbf.v4.new_markdown_cell(s.strip())
def code(s): return nbf.v4.new_code_cell(s.strip())


NOTEBOOKS = {
"02_indexing": [
    md("""# 02 · Indexing: turning 10,943 chunks into searchable vectors and keywords
Inspects the outputs of `ragtraining.index` (Phase 2). Two **dense** indexes and one **sparse** index:
- **Qwen3-Embedding 0.6B** (1024-d, normalized) → main dense retriever (**D13**)
- **nomic-embed-text raw** (768-d, NOT normalized) → only for the cosine-vs-Euclid experiment (**D1/D14**)
- **BM25** with Snowball stemming → keyword retriever (**D15**)"""),
    code(SETUP + """
chunks = pd.read_parquet('../data/chunks.parquet')
D = np.load('../data/emb_qwen3.npy'); R = np.load('../data/emb_nomic_raw.npy')
meta = json.load(open('../data/index_meta.json'))
print('chunks', len(chunks), '| Qwen3', D.shape, '| nomic raw', R.shape)"""),
    md("""## 1. Vector length (norm): does it carry information?
Cosine only looks at **direction**, Euclidean distance also at **length**. So combining them can only help if lengths
vary in a meaningful way. Ollama's `/api/embed` always returns unit vectors, and only the legacy endpoint gives raw ones (P-23)."""),
    code("""nd, nr = np.linalg.norm(D, axis=1), np.linalg.norm(R, axis=1)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
ax[0].hist(nd, bins=50, range=(0.99, 1.01), color='tab:blue'); ax[0].set_title(f'Qwen3 norms: all = 1 (CV {nd.std()/nd.mean():.4f})')
ax[1].hist(nr, bins=60, color='tab:orange'); ax[1].set_title(f'nomic raw norms (CV {nr.std()/nr.mean():.3f})')
ax[2].scatter(chunks.n_tokens, nr, s=2, alpha=.25, color='tab:orange')
ax[2].set(xlabel='chunk tokens', ylabel='nomic norm', title='length of text vs length of vector')
plt.tight_layout(); plt.show()
print('correlation(tokens, norm) =', round(float(np.corrcoef(chunks.n_tokens, nr)[0, 1]), 3))"""),
    md("""**Reading:** nomic lengths vary by only ~4% (CV). Shorter chunks get slightly longer vectors, which is a property
of the model, not of meaning. That's the first hint that Euclid won't beat cosine (confirmed in notebook 04)."""),
    md("""## 2. The cosine ↔ Euclid identity, checked numerically
For unit vectors: **‖a − b‖² = 2 − 2·cos θ**, so both give the *same ranking*. On raw vectors they can differ."""),
    code("""rng = np.random.default_rng(0); i, j = rng.integers(0, len(D), (2, 2000))
cos_q = (D[i] * D[j]).sum(1); d2_q = ((D[i] - D[j]) ** 2).sum(1)
print('Qwen3 (unit):  max |d² - (2 - 2cos)| =', f'{np.abs(d2_q - (2 - 2 * cos_q)).max():.2e}')
cos_n = (R[i] * R[j]).sum(1) / (nr[i] * nr[j]); dist_n = np.linalg.norm(R[i] - R[j], axis=1)
rho = pd.Series(cos_n).corr(pd.Series(-dist_n), method='spearman')
print(f'nomic (raw):   rank correlation(cos, -distance) = {rho:.3f}  (1.000 would mean identical rankings)')"""),
    md("""## 3. A map of the corpus
Each dot is a chunk. **PCA** reduces 1024 → 50 dimensions, then **t-SNE** squeezes them to 2-D while keeping
neighbours close. Courses form their own islands. Overlaps show topics shared across courses
(e.g. CAPM appears in Investments, Financial Modeling and Financial Intermediation)."""),
    code("""from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
s = chunks.sample(3000, random_state=0)
xy = TSNE(2, init='pca', perplexity=30, random_state=0).fit_transform(PCA(50, random_state=0).fit_transform(D[s.index]))
fig, ax = plt.subplots(figsize=(11, 7.5))
cmap = plt.get_cmap('tab20')
for k, (course, idx) in enumerate(s.groupby('course').groups.items()):
    m = s.index.get_indexer(idx)
    ax.scatter(xy[m, 0], xy[m, 1], s=4, color=cmap(k % 20), label=f'{course[:28]} ({len(m)})')
ax.legend(markerscale=4, fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left'); ax.set_xticks([]); ax.set_yticks([])
ax.set_title('3,000 random chunks, Qwen3 embeddings, t-SNE'); plt.tight_layout(); plt.show()"""),
    md("""## 4. The sparse side: what BM25 actually sees
Text is lower-cased, stop words dropped, words stemmed ("locking" → "lock"). **IDF** = how rare a term is in the corpus:
rare terms (2pl, serializ) decide the ranking, and common ones (use, data) barely count."""),
    code("""from ragtraining.index.sparse import tokenize
bm = pickle.load(open('../data/bm25.pkl', 'rb'))
q = 'How does two-phase locking (2PL) guarantee conflict serializability of a schedule?'
print('raw tokens :', tokenize(q, stem=False)); print('stemmed    :', tokenize(q))
idf = bm.bm25.idf
pd.DataFrame({'term': tokenize(q), 'IDF': [round(idf.get(t, 0), 2) for t in tokenize(q)]}).sort_values('IDF', ascending=False)"""),
    code("""print(f"BM25 parameters: k1={meta['sparse']['k1']} (term-frequency saturation), b={meta['sparse']['b']} (length penalty)")
print('vocabulary size:', len(idf), 'stemmed terms')"""),
],

"03_retrieval": [
    md("""# 03 · Retrieval: the same question, ranked by each method
Inspects `ragtraining.retrieve.search` (Phase 3). For a few questions, the top-5 chunks per method are shown side by side.
Everything runs locally (Ollama + local reranker): **$0**. Needs `ollama serve` running."""),
    code(SETUP + """
from ragtraining.retrieve.search import Retriever
r = Retriever()
METHODS = {'cosine (Qwen3)': dict(method='cosine'), 'euclid (nomic raw)': dict(method='euclid'),
           'BM25': dict(method='bm25'), 'hybrid RRF': dict(method='hybrid', how='rrf'),
           'hybrid + local rerank': dict(method='hybrid', how='rrf', rerank='local')}

def label(x):
    page = f'p.{int(x.page_start)}' if pd.notna(x.page_start) else (x.section if isinstance(x.section, str) else '')[:15]
    return f"{x.course[:14]} · {x.path.split('/')[-1][:28]} · {page}"

def side_by_side(q, k=5):
    res = {n: r.search(q, k=k, **cfg) for n, cfg in METHODS.items()}
    print('Q:', q)
    return pd.DataFrame({n: [label(x) for x in df.itertuples()] for n, df in res.items()}, index=range(1, k + 1)), res"""),
    md("""## 1. A normal exam question
All methods agree on the course. They differ on **which pages** rise to the top."""),
    code("""t1, res1 = side_by_side('How does two-phase locking guarantee that a schedule is serializable?'); t1"""),
    md("""## 2. A paraphrase with none of the slide's keywords → dense wins
The question describes the Feistel idea without saying "Feistel", "round" or "XOR". BM25 has nothing to match."""),
    code("""t2, _ = side_by_side("Why can a block cipher be decrypted even when its inner scrambling function can't be inverted?"); t2"""),
    md("""## 3. An exact technical term → BM25 shines
Rare names and formulas ("Sharpe ratio") are exactly what keyword search is built for."""),
    code("""t3, _ = side_by_side('Sharpe ratio formula'); t3"""),
    md("""## 4. How RRF fuses two rankings
**score = 1/(60 + rank_dense) + 1/(60 + rank_bm25)**. It uses only ranks, so the scales of the two scores never need
calibrating. A chunk ranked well by *both* beats one ranked #1 by only one."""),
    code("""q = 'How does two-phase locking guarantee that a schedule is serializable?'
rank = lambda s: pd.Series(s).rank(ascending=False, method='first').astype(int).to_numpy()
rd, rb = rank(r.cosine(q)), rank(r.bm25_scores(q))
top = r.search(q, k=8, method='hybrid', how='rrf', use_type_weights=False)
pos = {c: i for i, c in enumerate(r.chunks.chunk_id)}
idx = [pos[c] for c in top.chunk_id]
pd.DataFrame({'chunk': [label(x) for x in top.itertuples()], 'dense rank': rd[idx], 'BM25 rank': rb[idx],
              'RRF score': (1 / (60 + rd[idx]) + 1 / (60 + rb[idx])).round(4)}, index=range(1, 9))"""),
    md("""## 5. How different are the methods? (overlap of top-10)
**Jaccard** = shared chunks / all chunks in the two top-10 lists. Low overlap between dense and BM25 is *why* hybrid
fusion helps: they find different relevant pages."""),
    code("""tops = {n: set(r.search(q, k=10, **cfg).chunk_id) for n, cfg in METHODS.items()}
names = list(tops)
J = pd.DataFrame([[len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for b in names] for a in names], index=names, columns=names)
fig, ax = plt.subplots(figsize=(6.5, 5)); im = ax.imshow(J, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(len(names)), names, rotation=40, ha='right'); ax.set_yticks(range(len(names)), names)
for a in range(len(names)):
    for b in range(len(names)): ax.text(b, a, f'{J.iat[a, b]:.2f}', ha='center', va='center', fontsize=8)
ax.set_title('top-10 overlap (Jaccard), 2PL question'); plt.tight_layout(); plt.show()"""),
],

"04_retrieval_eval": [
    md("""# 04 · Retrieval evaluation: which method finds the right page?
Charts for LOGBOOK §7.2. **60 synthetic questions** (Opus wrote one per source passage). A hit = a retrieved chunk
overlaps the source page. Metrics @10: **Hit@k** (right page in top k?), **MRR** (1/rank of the first hit),
**nDCG** (rewards hits near the top).
⚠️ Synthetic questions share some wording with their source, so they **favour BM25** (P-34). Max's own questions are the reality check."""),
    code(SETUP + """
ev = pd.read_csv('../data/eval_retrieval.csv').sort_values('mrr')
sw = pd.read_csv('../data/eval_sweeps.csv')
perq = pd.read_csv('../data/eval_retrieval_perq.csv', index_col=0)
ev[['method', 'hit1', 'hit5', 'mrr', 'ndcg', 'ms_per_query']].sort_values('mrr', ascending=False)"""),
    md("## 1. Quality per method"),
    code("""fig, ax = plt.subplots(figsize=(10, 5.5)); y = np.arange(len(ev)); h = .27
for k, (m, c) in enumerate([('hit5', 'tab:blue'), ('mrr', 'tab:orange'), ('ndcg', 'tab:green')]):
    ax.barh(y + (k - 1) * h, ev[m], h, label=m, color=c)
ax.set_yticks(y, ev.method); ax.set_xlim(.4, 1.02); ax.legend(loc='lower right'); ax.set_title('Retrieval quality (60 synthetic questions)')
plt.tight_layout(); plt.show()"""),
    md("""**Reading:** the **embedding model** matters most (Qwen3 vs nomic: +15 pts Hit@5). **Euclid = cosine** (D1 answer).
Hybrid beats either side alone, and a **reranker** adds the last few points. Claude rerank is best (D19)."""),
    md("## 2. Quality vs. speed: what you pay for the last points"),
    code("""fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(ev.ms_per_query, ev.mrr, s=40)
for x in ev.itertuples(): ax.annotate(x.method, (x.ms_per_query, x.mrr), fontsize=7, xytext=(4, 2), textcoords='offset points')
ax.set_xscale('log'); ax.set(xlabel='ms per query (log)', ylabel='MRR', title='Rerankers: ~100× slower for +0.04–0.11 MRR')
plt.tight_layout(); plt.show()"""),
    md("""## 3. Sweeps: how much of each signal to mix
α = weight of the first signal in a weighted (min-max normalized) fusion."""),
    code("""fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for a, (name, g) in zip(ax, sw.groupby('sweep')):
    a.plot(g.alpha, g.hit5, 'o-', label='Hit@5'); a.plot(g.alpha, g.mrr, 's-', label='MRR'); a.set(title=name, xlabel='α')
ax[0].legend(); plt.tight_layout(); plt.show()"""),
    md("""**Reading:** cos↔Euclid is **flat** (length adds nothing). Dense↔BM25 peaks at α≈0.75: **both together beat either alone**."""),
    md("""## 4. Per question: which questions does everyone miss? (Hit@5)
Rows = questions, columns = methods. A row that's red for every method usually means a **bad gold label** or a
question that is genuinely hard. These are the rows worth checking in Max's spot-check."""),
    code("""order = perq.mean(axis=1).sort_values().index
fig, ax = plt.subplots(figsize=(11, 9)); ax.imshow(perq.loc[order].T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_yticks(range(perq.shape[1]), perq.columns, fontsize=8); ax.set_xlabel('question (sorted: hardest left)')
ax.set_title('Hit@5 per question and method (green = found)'); plt.tight_layout(); plt.show()
tests = pd.read_json('../testset/synthetic.jsonl', lines=True)
hard = perq.mean(axis=1).sort_values().head(5)
pd.DataFrame({'question': tests.question[hard.index], 'share of methods that hit': hard.round(2)})"""),
],

"05_generation_guardrails": [
    md("""# 05 · Generation + guardrails: when should the tutor refuse?
Inspects `ragtraining.guard.rules` and the results of `ragtraining.eval.guardrails` (LOGBOOK §7.4). Reads stored results: **$0**.
Defences, in order: **① input filter** (regex) → **② confidence gate** (best rerank score too low → "not covered", no Opus call)
→ **③ chunk sanitizer** (drops instruction-like lines in documents) → **④ system prompt** (sources are data, G1–G11)
→ **⑤ citation validator**."""),
    code(SETUP + """
from ragtraining.eval.guardrails import threshold_table
from ragtraining.generate.tutor import REFUSE_BELOW
sc = pd.read_csv('../data/guard_scores.csv')
suite = pd.DataFrame(json.load(open('../data/guard_suite.json')))
sc.groupby('kind')[['claude', 'local']].median()"""),
    md("""## 1. The confidence gate: two score distributions
Answerable questions should score high, not-covered ones low. The **threshold** sits in the gap. Where the two overlap,
you have to choose which error is worse."""),
    code("""fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for a, col, bins, cut in [(ax[0], 'claude', np.arange(-0.5, 11), [3, 6]), (ax[1], 'local', np.logspace(-3.2, 0, 30), [0.05])]:
    for kind, c in [('answerable', 'tab:green'), ('off_topic', 'tab:red'), ('unanswerable', 'tab:orange')]:
        a.hist(sc[sc.kind == kind][col], bins=bins, alpha=.6, color=c, label=kind)
    for t in cut: a.axvline(t, ls='--', color='k'); a.text(t, a.get_ylim()[1] * .9, f' {t}', fontsize=8)
    a.set(title=f'{col} reranker: best score per question', xlabel='top score'); a.legend(fontsize=8)
ax[1].set_xscale('log'); plt.tight_layout(); plt.show()
threshold_table(sc)"""),
    md(f"""**Reading:** on synthetic questions a threshold of 6 looks perfect. But the real, vague question *"What is a key?"*
scored **4** → it would be wrongly refused (**P-37**). The gate only saves money: whatever slips through still reaches Opus,
which refuses off-topic questions itself. So a **false refusal is the worse error** → final threshold **Claude 3, local 0.05**."""),
    md("## 2. The suite: 26 cases end to end"),
    code("""suite.groupby('kind', sort=False).passed.agg(passed='sum', cases='count')"""),
    code("""suite[['kind', 'question', 'passed', 'why']].style.hide(axis='index')"""),
    md("""## 3. Injection hidden inside a document
The payload is appended to the top real chunk. The obvious English one matches the regex and gets **removed**. The subtle
English and Italian ones get through to Opus, which **ignores** them because the system prompt says sources are data (G5)."""),
    code("""from ragtraining.guard.rules import sanitize_chunk
for p in suite[suite.kind == 'injection_doc'].payload:
    clean, n = sanitize_chunk('Feistel: L_i = R_(i-1) ...\\n' + p)
    print(f'removed {n} line(s) | payload: {p[:70]}')
print('\\nAnswer with the subtle payload (look for "24 December": absent):\\n')
print(suite[suite.kind == 'injection_doc'].answer.iloc[1][:600])"""),
    md("""## 4. Ambiguous questions: retrieval decides the breadth of the answer (P-38)
For *"What is a key?"* all 8 sources are crypto, so the tutor can't mention **primary keys** (Data Management).
That's a **retrieval** problem and the motivation for multi-query (notebook 07)."""),
    code("""for _, r in suite[suite.kind == 'ambiguous'].iterrows():
    print(f"{'PASS' if r.passed else 'FAIL'} {r.question:26s} {r.why}")
    print('     sources:', sorted({s.split(' · ')[0].split(' ', 1)[1] for s in r.sources}))"""),
],

"06_ragas": [
    md("""# 06 · RAGAS: grading the answers, not just the retrieval
Inspects `ragtraining.eval.ragas_lite` / `run_ragas` (LOGBOOK §7.3). Judge = **Haiku 5.5**, answers = **Opus 5.5**. Reads stored results: **$0**.
- **faithfulness**: share of the answer's claims supported by the sources → *does it make things up?*
- **answer relevancy**: does it answer *this* question? (questions regenerated from the answer vs. the real one)
- **context precision**: are the useful sources ranked high?
- **context recall**: did retrieval bring everything the reference answer needs?"""),
    code(SETUP + """
M = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
old = {p: json.load(open(f'../data/ragas_{p}.json')) for p in ['baseline', 'prod']}
new = {p: json.load(open(f'../data/ragas_{p}_rejudged.json')) for p in ['baseline', 'prod']}
pd.DataFrame({f'{p} · {w}': d[p]['summary'] for w, d in [('old judge', old), ('fixed judge', new)] for p in d}).loc[M]"""),
    md("""## 1. P-33: the judge must see what the tutor saw
The tutor ends with *"Review: Lecture 3 slides p.1-4"*. Those page numbers come from source **metadata**. The old judge only
got the chunk **text**, so it marked every such pointer as unsupported. Same answers, same sources, the judge just sees the headers now:"""),
    code("""fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for a, p in zip(ax, ['baseline', 'prod']):
    o = [r['faithfulness'] for r in old[p]['rows']]; n = [r['faithfulness'] for r in new[p]['rows']]
    x = np.arange(len(o)); a.bar(x - .2, o, .4, label='old judge'); a.bar(x + .2, n, .4, label='fixed judge')
    a.set(title=f'{p}: faithfulness per question', xlabel='question'); a.legend(loc='lower left')
plt.tight_layout(); plt.show()
for p in new:
    n_old = sum(len(r['unsupported']) for r in old[p]['rows']); n_new = sum(len(r['unsupported']) for r in new[p]['rows'])
    print(f'{p:9s} unsupported claims: {n_old} -> {n_new}')"""),
    md("""## 2. What is still "unsupported"?
Mostly the tutor's own **inferences** and mnemonics. They're reasonable, but not stated in the sources. G1 says *only from the
material*, so the judge is right to flag them."""),
    code("""pd.DataFrame([(p, u) for p in new for r in new[p]['rows'] for u in r['unsupported']], columns=['pipeline', 'claim']).head(15)"""),
    md("""## 3. Judge noise (P-36)
Context precision and recall don't depend on the answer, and the sources were *identical* in both runs. Any change is the
judge's own randomness. Rule of thumb: differences under ~0.03 on 20 questions are noise."""),
    code("""pd.DataFrame({p: {m: round(new[p]['summary'][m] - old[p]['summary'][m], 3) for m in ['context_precision', 'context_recall']}
              for p in new}).rename_axis('change with SAME sources')"""),
    md("""## 4. Final run (S3, D24): local engine on all 70 questions vs. Claude engine on Max's 12
Current concise tutor, final test set (58 synthetic after the spot-check + Max's 12). Local answers cost $0, only the judge is paid."""),
    code("""fin = {'local · all 70': json.load(open('../data/ragas_local_all.json')),
       'claude · Max 12': json.load(open('../data/ragas_prod_manual.json'))}
tab = {k: {m: v['summary'][m] for m in M} | {'n': v['summary']['n'], 'citation problems': v['summary']['citation_problems']}
       for k, v in fin.items()}
loc = fin['local · all 70']['summary']['by_source']
tab |= {f'local · {g}': {m: loc[g][m] for m in M} | {'n': loc[g]['n']} for g in loc}
pd.DataFrame(tab).T"""),
    md("""**Reading:** both engines are faithful (0.96 vs 0.99 on the same 12). The big gap is **context precision** (0.75 vs 0.95):
the Claude *reranker* puts more relevant chunks in the top 8. Answer relevancy is low on Max's vague questions for both
engines ("what is DS?"), which is a property of the metric, not a wrong answer."""),
    md("""## 5. P-53 → D25: a check that only logs is a measurement, not a guardrail
8/70 local answers had no inline `[S#]`. In 6 of them qwen3 named the sources only in the final *"Review: S1, p.4"* line.
Fix: if an answer has no citations, the tutor asks **once more** with a reminder (`tutor.CITE_REMINDER`). `run_ragas patch`
re-answered only those 8 rows on the **same evidence** and re-judged them (Haiku, ≈ $0.01):"""),
    code("""p = json.load(open('../data/ragas_local_all_patched.json'))
before = {r['question']: r for r in fin['local · all 70']['rows']}
pd.DataFrame([dict(question=r['question'][:55], cited_before='[S' in r['old_answer'], cited_after='[S' in (r['answer'] or ''),
                   faith_before=before[r['question']]['faithfulness'], faith_after=r['faithfulness'])
              for r in p['rows'] if r.get('patched')])"""),
    md("""## 6. P11 → D25: the regression gate
A threshold turns a score into pass/fail. Every bar sits at least one judge-noise step (±0.03) below today's score: a real
drop fails, noise doesn't. Answer relevancy is a *soft* bar (warns only). Run it after any component change:
`python -m ragtraining.eval.run_ragas gate <name>` (exit code 1 on FAIL)."""),
    code("""from ragtraining.eval.run_ragas import gate
for name in ['local_all', 'local_all_patched']:
    print(f'--- {name}'); gate(name)"""),
],

"07_advanced_agentic": [
    md("""# 07 · Advanced retrieval: multi-query, HyDE, HNSW, agentic loop
Inspects `ragtraining.retrieve.expand`, `ragtraining.eval.advanced` and `ragtraining.agent.loop` (LOGBOOK §7.5).
Haiku generations are **cached** (`data/expansions.json`) and HNSW is local, so this notebook costs **$0**. Needs `ollama serve`."""),
    code(SETUP + """
from ragtraining.retrieve.search import Retriever
from ragtraining.retrieve.expand import alt_queries, hypothetical_passage
r = Retriever(); courses = sorted(r.chunks.course.unique())
adv = pd.read_csv('../data/eval_advanced.csv'); senses = pd.read_csv('../data/eval_ambiguous_senses.csv')"""),
    md("""## 1. Multi-query: one question, several searches
Haiku rewrites the question 3 ways: other wording, the terms slides use, and **one query per meaning** if the term is ambiguous."""),
    code("""q = 'What is a key?'
for x in [q] + alt_queries(q, courses): print('•', x)"""),
    md("""## 2. How you fuse the rankings decides what you get
- **RRF** sums 1/(60+rank) over the queries → it rewards chunks *several* queries agree on, i.e. the **majority meaning**.
- **Round-robin** takes each query's next-best chunk in turn → **every meaning gets slots**."""),
    code("""def courses_top8(**cfg):
    return r.search(q, k=8, **cfg).course.value_counts().to_dict()
for name, cfg in [('plain hybrid', dict(method='hybrid', how='rrf')), ('multi-query, RRF fuse', dict(method='multi_query')),
                  ('multi-query, round-robin', dict(method='multi_query', fuse='rr')),
                  ('multi-query RR + local rerank', dict(method='multi_query', fuse='rr', rerank='local'))]:
    print(f'{name:32s}', courses_top8(**cfg))"""),
    md("## 3. Sense coverage on the 4 ambiguous questions (meanings found in the top-8 chunks)"),
    code("""senses.assign(score=senses.covered.astype(str) + '/' + senses.of.astype(str)).pivot(index='config', columns='question', values='score')"""),
    md("""## 4. HyDE: search with a fake answer instead of the question
The passage may contain wrong details. That doesn't matter, because it's only used to *find* real pages. It works when
questions and answers are worded very differently. It can hurt when the fake answer drifts off topic."""),
    code("""q2 = "Why can a block cipher be decrypted even when its inner scrambling function can't be inverted?"
print(hypothetical_passage(q2)[:500], '\\n')
for name, cfg in [('question → cosine', dict(method='cosine')), ('HyDE passage → cosine', dict(method='hyde', dense_only=True))]:
    h = r.search(q2, k=3, **cfg)
    print(name); [print('   ', x.course[:20], '·', x.path.split('/')[-1][:45], '·', x.page_start) for x in h.itertuples()]"""),
    md("## 5. Retrieval quality on the 60 synthetic questions"),
    code("""adv[['method', 'hit1', 'hit5', 'mrr', 'ndcg', 'ms_per_query']].sort_values('mrr', ascending=False)"""),
    md("""⚠️ The synthetic questions are specific and share wording with their page, so plain hybrid is already near the ceiling
(Hit@5 0.95 with rerank). Expansion is meant for **vague and ambiguous** questions, so Max's own questions are the real test.
Expansion also costs one Haiku call (~1–2 s) per question."""),
    md("""## 6. HNSW vs. exact search
**HNSW** builds a graph of near neighbours and walks it instead of comparing against every vector. At 10,943 chunks exact
search already takes < 1 ms, so it isn't needed here. At millions of chunks it would be."""),
    md("""FAISS runs in a **separate process** (`ragtraining/eval/hnsw.py`): PyTorch (loaded above by the reranker) and FAISS each
bundle their own OpenMP runtime, and both in one process crashes the kernel on macOS (P-48).
Timing is **one query at a time**, like the tutor. Timing a batch of 200 makes exact search look ~20× faster (P-49)."""),
    code("""import subprocess
res = json.loads(subprocess.run([sys.executable, '-m', 'ragtraining.eval.hnsw'], cwd='..', capture_output=True,
                                text=True, check=True).stdout)
print(f"exact: {res['exact_ms_per_query']} ms/query (batched: {res['exact_ms_per_query_batched']})")
pd.DataFrame(res['hnsw'])"""),
    md("""## 7. The agentic loop (`ragtraining/agent/loop.py`)
**PLAN** (Haiku: 1–3 queries + course filter) → **SEARCH** (hybrid + local rerank) → **GRADE** (Haiku: is the evidence
enough? if not, what's missing + new queries) → up to 2 more rounds, widening the course filter → **ANSWER** (Opus tutor).
The loop lives in our code, not in a hosted agent, because the data and the retriever are local.
In session 1 it rescued a vague question ("growing/shrinking phase" → two-phase locking). On RAGAS it isn't yet better
than one search (§7.3). Tuning is open: the grader's stopping rule (P-31), and multi-query inside the plan step."""),
],
}


def build(key: str):
    name = next(n for n in NOTEBOOKS if n.startswith(key))
    nb = nbf.v4.new_notebook(cells=NOTEBOOKS[name])
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = NB_DIR / f"{name}.ipynb"
    nbf.write(nb, path)
    print("wrote", path)


if __name__ == "__main__":
    for k in sys.argv[1:] or list(NOTEBOOKS):
        build(k)
