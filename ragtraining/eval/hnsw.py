"""Phase 7: FAISS HNSW (approximate) vs. exact NumPy search on the Qwen3 vectors (LOGBOOK §7.2).

Run it in its OWN process: `python -m ragtraining.eval.hnsw` prints one JSON object.
Why: PyTorch (local reranker) and FAISS each bundle their own OpenMP runtime. Loading both in one process makes
macOS abort/segfault (OMP Error #15, P-48). This module therefore imports only NumPy + FAISS, never torch,
and the notebook calls it via subprocess. Cost: $0, a few seconds.
"""
from __future__ import annotations

import json
import time

import numpy as np

from ragtraining.config import DATA_DIR

CONFIGS = [(16, 32), (16, 128), (32, 256)]      # (M = graph links per node, efSearch = candidates kept while walking)


def run(n_queries: int = 200, k: int = 10, seed: int = 1) -> dict:
    import faiss
    D = np.load(DATA_DIR / "emb_qwen3.npy").astype("float32")
    Q = D[np.random.default_rng(seed).choice(len(D), n_queries, replace=False)] + 0.01   # near-duplicate queries
    t0 = time.time(); S = Q @ D.T; t_batch = (time.time() - t0) / len(Q) * 1000     # all queries in one matmul
    exact = np.argsort(-S, axis=1)[:, :k]
    t0 = time.time()                                       # one question at a time = how the tutor really searches
    for q in Q:
        np.argpartition(-(D @ q), k)[:k]
    t_single = (time.time() - t0) / len(Q) * 1000
    rows = []
    for M, ef in CONFIGS:
        idx = faiss.IndexHNSWFlat(D.shape[1], M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efSearch = ef
        idx.add(D)
        _, I = idx.search(Q, k)
        t0 = time.time()
        for q in Q:
            idx.search(q[None], k)
        t = (time.time() - t0) / len(Q) * 1000
        recall = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(I, exact)]))
        rows.append(dict(M=M, efSearch=ef, recall_at_10=round(recall, 3), ms_per_query=round(t, 3)))
    # batch timing divides one big BLAS matmul by n_queries: much faster per query, but not how a user asks (P-49)
    return dict(n_vectors=len(D), dim=D.shape[1], exact_ms_per_query=round(t_single, 3),
                exact_ms_per_query_batched=round(t_batch, 3), hnsw=rows)


if __name__ == "__main__":
    print(json.dumps(run()))
