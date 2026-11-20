"""Phase 2 smoke test: 12 paraphrased questions with known answer locations.

NOT the real evaluation (that's Phase 4, ~75 questions). It's a quick sanity check used to
choose the embedding model. Questions avoid the key term on purpose, so a model has to
match *meaning*, not words. Gold = chunks in the right course that contain the key term.

Run:  .venv/bin/python -m ragtraining.eval.smoke
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from ragtraining.config import DATA_DIR
from ragtraining.index.embed import MODELS, embed

SMOKE = [  # (question, course substring, gold regex)
    ("How do we measure how similar two word vectors are, independent of their length?",
     "ed3book", r"cosine"),
    ("Which block cipher design splits the block into two halves and swaps them over many rounds?",
     "Information Security", r"Feistel"),
    ("Why do banks exist? The argument about monitoring borrowers on behalf of many small depositors",
     "Financial Intermediation", r"delegated monitoring"),
    ("In which market structure are orders collected and executed together only at specific times?",
     "Trading", r"call market"),
    ("Which normal form removes transitive dependencies between non-key attributes?",
     "Data Management", r"third normal form|3NF"),
    ("How can replicated servers agree on one value even when some of them crash?",
     "Distributed Systems", r"Raft|Paxos|consensus"),
    ("Coverage criterion requiring both outcomes of every decision in the code to be exercised",
     "Software_Quality", r"branch coverage"),
    ("Locking protocol with a growing and a shrinking phase that guarantees serializability",
     "Data Management", r"two-phase locking|2PL"),
    ("How to capture the causal order of events across processes without a global clock?",
     "Distributed Systems", r"vector clock|Lamport"),
    ("What is the cost of trading immediately, measured between the best buy and sell quotes?",
     "Trading", r"bid-ask spread|bid–ask spread"),
    ("Proof technique: assume the claim is false and derive something impossible",
     "Discrete", r"contradiction"),
    ("Risk-adjusted performance: excess return per unit of total volatility",
     "Investments", r"Sharpe ratio"),
]


def build_subset(chunks: pd.DataFrame, n_distractors: int = 3000, seed: int = 0):
    gold = [set(chunks.index[chunks.path.str.contains(c) & chunks.text.str.contains(p, case=False, regex=True)])
            for _, c, p in SMOKE]
    all_gold = set().union(*gold)
    rest = chunks.drop(index=list(all_gold)).sample(n_distractors, random_state=seed)
    sub = pd.concat([chunks.loc[sorted(all_gold)], rest])
    return sub, gold


def metrics(rankings: list[list[int]], gold: list[set[int]]) -> dict:
    hit5 = np.mean([bool(set(r[:5]) & g) for r, g in zip(rankings, gold)])
    hit10 = np.mean([bool(set(r[:10]) & g) for r, g in zip(rankings, gold)])
    mrr = np.mean([next((1 / (i + 1) for i, d in enumerate(r[:10]) if d in g), 0) for r, g in zip(rankings, gold)])
    return dict(hit5=round(hit5, 2), hit10=round(hit10, 2), mrr10=round(mrr, 2))


def main(models: list[str] | None = None):
    chunks = pd.read_parquet(DATA_DIR / "chunks.parquet")
    sub, gold = build_subset(chunks)
    ids = sub.index.to_numpy()
    questions = [q for q, _, _ in SMOKE]
    res_file = DATA_DIR / "smoke_results.json"
    results = json.loads(res_file.read_text()) if res_file.exists() else {}

    tok = lambda s: s.lower().split()  # noqa: E731
    bm25 = BM25Okapi([tok(t) for t in sub.text])
    ranks = [list(ids[np.argsort(-bm25.get_scores(tok(q)))[:10]]) for q in questions]
    results["BM25 (reference)"] = dict(**metrics(ranks, gold), dim="-", chunks_per_s="-")

    for m in models or MODELS:
        t0 = time.time()
        D = embed(m, sub.text.tolist(), "doc", progress=True)
        speed = len(sub) / (time.time() - t0)
        Q = embed(m, questions, "query")
        ranks = [list(ids[np.argsort(-(D @ q))[:10]]) for q in Q]     # cosine (vectors are unit length)
        results[m] = dict(**metrics(ranks, gold), dim=D.shape[1], chunks_per_s=round(speed))
        np.save(DATA_DIR / f"smoke_{m}.npy", D)
        print(m, results[m], flush=True)

        res_file.write_text(json.dumps(results, indent=2, default=str))
    out = pd.DataFrame(results).T
    out["gold_chunks_total"] = sum(len(g) for g in gold)
    out["subset_size"] = len(sub)
    print(out.to_string())
    res_file.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)   # e.g.  python -m ragtraining.eval.smoke bge-m3 embeddinggemma
