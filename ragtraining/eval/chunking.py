"""Phase 4 (optional depth): compare the 4 chunking strategies (D12) on the same retrieval eval.  -> LOGBOOK §7.6

Each alternative chunk set gets its own Qwen3 embeddings (local, resumable, ~14 chunks/s) and BM25 index,
then the same 60 questions are scored. Relevance = the chunk overlaps the source page, so it is comparable
across sets. Caveat: chunks spanning MORE pages overlap the gold page more easily (fixed: 67% multi-page),
so also look at tokens per chunk (what the tutor has to read, and pay for).

Cost: $0 (all local). Time: ~35 min of embedding the first time, then cached in data/emb_qwen3_<set>.npy.
Run: .venv/bin/python -m ragtraining.eval.chunking
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ragtraining.config import DATA_DIR
from ragtraining.eval.retrieval import evaluate, load_testset
from ragtraining.index.build import DENSE_MODEL, _embed_resumable
from ragtraining.index.sparse import SparseIndex
from ragtraining.retrieve.search import Retriever

SETS = ["page", "fixed", "recursive", "hybrid"]
CFGS = {"cosine": dict(method="cosine"), "BM25": dict(method="bm25"),
        "hybrid RRF": dict(method="hybrid", how="rrf"),
        "hybrid RRF + local rerank": dict(method="hybrid", how="rrf", rerank="local")}


def retriever_for(s: str) -> Retriever:
    if s == "hybrid":
        return Retriever()                                       # the production indexes
    chunks = pd.read_parquet(DATA_DIR / f"chunks_{s}.parquet")
    f = DATA_DIR / f"emb_qwen3_{s}.npy"
    D = np.load(f) if f.exists() else _embed_resumable(f"qwen3_{s}", DENSE_MODEL, chunks.text.tolist(), raw=False)
    r = Retriever(chunks=chunks)                                 # loads the production files, then we swap them
    r.D, r.R, r.Rn, r.bm25 = D, None, None, SparseIndex(chunks.text.tolist())
    return r


if __name__ == "__main__":
    tests, rows = load_testset(), []
    for s in SETS:
        r = retriever_for(s)
        summary, _ = evaluate(r, tests, CFGS)
        t = r.chunks.n_tokens
        rows += [dict(chunking=s, chunks=len(r.chunks), median_tok=int(t.median()),
                      pct_multi_page=round(100 * (r.chunks.page_start.notna() & (r.chunks.page_end != r.chunks.page_start)).mean()),
                      **row) for row in summary.to_dict("records")]
        print(f"--- {s} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "eval_chunking.csv", index=False)
    print(out.pivot(index="chunking", columns="method", values="hit5").to_string())
    print(out.pivot(index="chunking", columns="method", values="mrr").to_string())
