"""Phase 2: build the indexes. Row i of every matrix = row i of data/chunks.parquet.

  dense  : Qwen3-Embedding 0.6B, normalized vectors (D13)   -> data/emb_qwen3.npy
  raw    : nomic-embed-text RAW vectors, for D1 (D14)         -> data/emb_nomic_raw.npy
  sparse : BM25 with stemming (D15)                           -> data/bm25.pkl

Embedding is resumable: progress is saved every 1,000 chunks to *.partial.npy, so an
interrupted run continues where it stopped.

Run:  .venv/bin/python -m ragtraining.index.build dense|raw|sparse
"""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from ragtraining.config import DATA_DIR
from ragtraining.index.embed import embed
from ragtraining.index.sparse import SparseIndex

DENSE_MODEL, RAW_MODEL = "qwen3-embedding:0.6b", "nomic-embed-text"
STEP = 1000


def _embed_resumable(name: str, model: str, texts: list[str], raw: bool) -> np.ndarray:
    out, part = DATA_DIR / f"emb_{name}.npy", DATA_DIR / f"emb_{name}.partial.npy"
    parts = [np.load(part)] if part.exists() else []
    start = len(parts[0]) if parts else 0
    t0 = time.time()
    for i in range(start, len(texts), STEP):
        parts.append(embed(model, texts[i:i + STEP], "doc", raw=raw))
        np.save(part, np.vstack(parts))
        n = i + len(parts[-1])
        rate = (n - start) / (time.time() - t0)
        print(f"{name}: {n}/{len(texts)}  {rate:.0f} chunks/s  ~{(len(texts) - n) / rate / 60:.1f} min left", flush=True)
    M = np.vstack(parts).astype(np.float32)
    np.save(out, M)
    part.unlink(missing_ok=True)
    return M


def write_meta():
    """Rebuild index_meta.json from the files on disk (safe even if builds ran in parallel, P-28)."""
    chunks = pd.read_parquet(DATA_DIR / "chunks.parquet", columns=["chunk_id"])
    meta = dict(n_chunks=len(chunks), first_id=chunks.chunk_id.iloc[0], last_id=chunks.chunk_id.iloc[-1])
    for key, name, model in [("dense", "qwen3", DENSE_MODEL), ("raw", "nomic_raw", RAW_MODEL)]:
        f = DATA_DIR / f"emb_{name}.npy"
        if f.exists():
            M = np.load(f, mmap_mode="r")
            n = np.linalg.norm(M, axis=1)
            meta[key] = dict(model=model, shape=list(M.shape), rows_match_chunks=M.shape[0] == len(chunks),
                             norm_mean=round(float(n.mean()), 4), norm_cv=round(float(n.std() / n.mean()), 4))
    if (DATA_DIR / "bm25.pkl").exists():
        meta["sparse"] = dict(file="bm25.pkl", stem=True, k1=1.5, b=0.75)
    (DATA_DIR / "index_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(what: str):
    texts = pd.read_parquet(DATA_DIR / "chunks.parquet").text.tolist()
    if what == "dense":
        _embed_resumable("qwen3", DENSE_MODEL, texts, raw=False)
    elif what == "raw":
        _embed_resumable("nomic_raw", RAW_MODEL, texts, raw=True)
    elif what == "sparse":
        with open(DATA_DIR / "bm25.pkl", "wb") as f:
            pickle.dump(SparseIndex(texts, stem=True), f)
    print(what, "done:", write_meta().get(what))


if __name__ == "__main__":
    main(sys.argv[1])
