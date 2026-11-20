"""Phase 4: retrieval evaluation of every method on the test set.

Relevance (page-level, fair across chunkers): a retrieved chunk is relevant if it comes from the
gold file AND its pages overlap the gold pages (or, for .md/.docx, it has the same section).

Metrics @10:  Hit@1, Hit@5, Hit@10, MRR (1/rank of first relevant), nDCG (rewards relevant chunks
ranked high), Recall (share of all relevant chunks retrieved). Plus latency.

Caveat: each synthetic question has ONE gold source. If another course explains the same concept,
retrieving it counts as a miss -> absolute numbers are pessimistic; comparisons stay fair.

Run:  .venv/bin/python -m ragtraining.eval.retrieval [--claude]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from ragtraining.config import DATA_DIR, ROOT
from ragtraining.llm import cost_so_far
from ragtraining.retrieve.search import Retriever

K = 10


def load_testset(manual: bool = True, drop_bad: bool = True) -> pd.DataFrame:
    """Synthetic items (minus those marked BAD in the spot-check) + Max's answerable questions (kind answerable/vague).
    Max's not_covered questions have no gold source: they're used by the guardrail calibration, not here."""
    from ragtraining.eval.manual import load_manual, review_verdicts
    syn = pd.read_json(ROOT / "testset" / "synthetic.jsonl", lines=True)
    if drop_bad:
        verdict = review_verdicts()
        syn = syn[syn.question.map(verdict).fillna("") != "BAD"]
    if not manual:
        return syn.reset_index(drop=True)
    man = load_manual()
    man = man[man.kind != "not_covered"].assign(course=lambda d: d.courses.str.join("; "))
    return pd.concat([syn.assign(kind="synthetic"), man], ignore_index=True)


def relevance_matrix(chunks: pd.DataFrame, tests: pd.DataFrame) -> np.ndarray:
    """rel[q, i] = True if chunk i is relevant for question q."""
    from ragtraining.eval.manual import gold_mask
    rel = np.zeros((len(tests), len(chunks)), dtype=bool)
    for qi, t in enumerate(tests.itertuples()):
        if isinstance(getattr(t, "gold", None), list):          # Max's questions: many gold refs (eval/manual.py)
            rel[qi] = gold_mask(chunks, t.gold, t.courses)
            continue
        same = (chunks.path == t.path).to_numpy()
        if pd.notna(t.page_start):
            overlap = ((chunks.page_start <= t.page_end) & (chunks.page_end >= t.page_start)).to_numpy()
            rel[qi] = same & overlap
        else:
            rel[qi] = same & (chunks.section == t.section).to_numpy()
    return rel


def score_ranking(ranked: np.ndarray, rel_row: np.ndarray) -> dict:
    r = rel_row[ranked[:K]]
    first = int(np.argmax(r)) if r.any() else None
    dcg = float((r / np.log2(np.arange(2, len(r) + 2))).sum())
    ideal = min(int(rel_row.sum()), K)
    idcg = float((1 / np.log2(np.arange(2, ideal + 2))).sum()) if ideal else 1.0
    return dict(hit1=float(r[:1].any()), hit5=float(r[:5].any()), hit10=float(r.any()),
                mrr=1 / (first + 1) if first is not None else 0.0, ndcg=dcg / idcg,
                recall=float(r.sum()) / max(int(rel_row.sum()), 1))


def configs(with_claude: bool) -> dict[str, dict]:
    c = {
        "cosine (Qwen3)": dict(method="cosine"),
        "cosine (nomic)": dict(method="cos_nomic"),
        "euclid (nomic raw)": dict(method="euclid"),
        "cos+euclid weighted a=.5": dict(method="cos_euclid", alpha=0.5, how="weighted"),
        "cos+euclid RRF": dict(method="cos_euclid", how="rrf"),
        "BM25 (stemmed)": dict(method="bm25"),
        "hybrid RRF": dict(method="hybrid", how="rrf"),
        "hybrid weighted .5": dict(method="hybrid", how="weighted", w_dense=0.5),
        "hybrid RRF + cap 2/file": dict(method="hybrid", how="rrf", per_file_cap=2),
        "hybrid RRF + MMR .7": dict(method="hybrid", how="rrf", mmr_lambda=0.7),
        "hybrid RRF + local rerank": dict(method="hybrid", how="rrf", rerank="local"),
        "cosine + local rerank": dict(method="cosine", rerank="local"),
    }
    if with_claude:
        c["hybrid RRF + Claude rerank (Haiku)"] = dict(method="hybrid", how="rrf", rerank="claude",
                                                      rerank_model="claude-haiku-5-5")
    return c


def evaluate(ret: Retriever, tests: pd.DataFrame, cfgs: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rel = relevance_matrix(ret.chunks, tests)
    pos = {cid: i for i, cid in enumerate(ret.chunks.chunk_id)}
    for q in tests.question:                       # warm the query-embedding cache (timed separately)
        ret._qvec(q, "dense"), ret._qvec(q, "raw")
    rows, per_q = [], {}
    for name, cfg in cfgs.items():
        t0, scores = time.time(), []
        for qi, q in enumerate(tests.question):
            res = ret.search(q, k=K, use_type_weights=True, **cfg)
            ranked = np.array([pos[c] for c in res.chunk_id])
            scores.append(score_ranking(ranked, rel[qi]))
        ms = (time.time() - t0) * 1000 / len(tests)
        df = pd.DataFrame(scores)
        per_q[name] = df.hit5.to_numpy()
        # split synthetic vs Max's questions (D6: the manual ones are the reality check, P-34)
        src = tests["source"].fillna("synthetic").to_numpy() if "source" in tests else np.full(len(tests), "synthetic")
        extra = {}
        if len(set(src)) > 1:
            for g in sorted(set(src)):
                extra[f"hit5_{g}"] = round(float(df.hit5[src == g].mean()), 3)
                extra[f"mrr_{g}"] = round(float(df.mrr[src == g].mean()), 3)
        rows.append(dict(method=name, **df.mean().round(3).to_dict(), **extra, ms_per_query=round(ms)))
        split = "  ".join(f"{k}={v:.2f}" for k, v in extra.items())
        print(f"{name:38s} hit5={df.hit5.mean():.2f} mrr={df.mrr.mean():.2f} ndcg={df.ndcg.mean():.2f} {ms:.0f} ms  {split}",
              flush=True)
    return pd.DataFrame(rows), pd.DataFrame(per_q, index=tests.index)


def sweep(ret: Retriever, tests: pd.DataFrame) -> pd.DataFrame:
    """alpha sweeps: cos<->euclid (nomic) and dense<->BM25 (weighted fusion)."""
    rel = relevance_matrix(ret.chunks, tests)
    out = []
    for a in [0, 0.25, 0.5, 0.75, 1.0]:
        for label, fn in [("cos_euclid (1=cos, 0=euclid)", lambda q: ret.cos_euclid(q, alpha=a)),
                          ("hybrid weighted (1=dense, 0=BM25)", lambda q: ret.hybrid(q, how="weighted", w_dense=a))]:
            m = pd.DataFrame([score_ranking(np.argsort(-fn(q))[:K], rel[i]) for i, q in enumerate(tests.question)])
            out.append(dict(sweep=label, alpha=a, hit5=m.hit5.mean(), mrr=m.mrr.mean(), ndcg=m.ndcg.mean()))
    return pd.DataFrame(out).round(3)


if __name__ == "__main__":
    tests = load_testset()
    ret = Retriever()
    summary, per_q = evaluate(ret, tests, configs("--claude" in sys.argv))
    summary.to_csv(DATA_DIR / "eval_retrieval.csv", index=False)
    per_q.to_csv(DATA_DIR / "eval_retrieval_perq.csv")
    sw = sweep(ret, tests)
    sw.to_csv(DATA_DIR / "eval_sweeps.csv", index=False)
    print("\n", summary.to_string(index=False), "\n\n", sw.to_string(index=False))
    print(f"\nClaude cost this run: ${cost_so_far():.3f}  (n={len(tests)} questions)")
