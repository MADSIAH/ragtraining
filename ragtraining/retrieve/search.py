"""Phase 3: every retrieval method, so Phase 4 can measure them side by side.

Dense (meaning)                          Sparse (exact words)
  cosine      Qwen3 normalized (D13)       bm25   BM25 + stemming (D15)
  cos_nomic   nomic, cosine on raw
  euclid      nomic, L2 distance on RAW vectors (D1 / D14)
  cos_euclid  nomic, cosine and Euclid combined (D1)

Fusion: RRF (rank-based) or weighted (min-max normalized scores).
Post-processing: metadata filters, doc_type weights (summaries lower, D10),
per-file cap (source balancing), MMR (diversity).

All scorers return an array of length n_chunks where HIGHER = BETTER.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ragtraining.config import DATA_DIR
from ragtraining.index.build import DENSE_MODEL, RAW_MODEL
from ragtraining.index.embed import embed

DOC_TYPE_WEIGHT = {"summary": 0.85}   # D10: summaries rank below official sources on ties


# ---------------- fusion helpers (pure functions, unit-tested) ----------------
def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)


def ranks_of(scores: np.ndarray) -> np.ndarray:
    """rank[i] = position of chunk i (0 = best)."""
    r = np.empty(len(scores), dtype=np.int64)
    r[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
    return r


def rrf(score_lists: list[np.ndarray], k: int = 60, weights: list[float] | None = None) -> np.ndarray:
    """Reciprocal Rank Fusion: sum_i w_i / (k + rank_i). Uses ranks only -> no score calibration needed."""
    weights = weights or [1.0] * len(score_lists)
    return sum(w / (k + 1 + ranks_of(s)) for w, s in zip(weights, score_lists))


def weighted(score_lists: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted sum of min-max-normalized scores. Sensitive to score distributions (unlike RRF)."""
    return sum(w * minmax(s) for w, s in zip(weights, score_lists))


def mmr(cands: np.ndarray, rel: np.ndarray, vecs: np.ndarray, k: int, lam: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance: trade relevance vs. similarity to what's already picked."""
    rel_c = minmax(rel[cands])
    V = vecs[cands]
    chosen: list[int] = []
    for _ in range(min(k, len(cands))):
        if chosen:
            redundancy = (V @ V[chosen].T).max(axis=1)
        else:
            redundancy = np.zeros(len(cands))
        score = lam * rel_c - (1 - lam) * redundancy
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return [int(cands[i]) for i in chosen]


# ---------------- the retriever ----------------
@dataclass
class Retriever:
    chunks: pd.DataFrame = field(default_factory=lambda: pd.read_parquet(DATA_DIR / "chunks.parquet"))

    def __post_init__(self):
        self.D = np.load(DATA_DIR / "emb_qwen3.npy")                          # (n, 1024), unit length
        raw_f = DATA_DIR / "emb_nomic_raw.npy"
        self.R = np.load(raw_f) if raw_f.exists() else None                   # (n, 768), raw length
        self.Rn = self.R / np.linalg.norm(self.R, axis=1, keepdims=True) if self.R is not None else None
        with open(DATA_DIR / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.type_w = self.chunks.doc_type.map(DOC_TYPE_WEIGHT).fillna(1.0).to_numpy()
        self._qcache: dict[tuple[str, str], np.ndarray] = {}

    # --- query embeddings (cached: the same question is scored by many methods) ---
    def _qvec(self, q: str, which: str) -> np.ndarray:
        key = (which, q)
        if key not in self._qcache:
            if which == "dense":
                self._qcache[key] = embed(DENSE_MODEL, [q], "query")[0]
            else:
                self._qcache[key] = embed(RAW_MODEL, [q], "query", raw=True)[0]
        return self._qcache[key]

    # --- scorers: higher = better ---
    def cosine(self, q: str) -> np.ndarray:
        return self.D @ self._qvec(q, "dense")

    def cos_nomic(self, q: str) -> np.ndarray:
        qv = self._qvec(q, "raw")
        return self.Rn @ (qv / np.linalg.norm(qv))

    def euclid(self, q: str) -> np.ndarray:
        """Negative L2 distance on RAW vectors. ||d-q||^2 = ||d||^2 + ||q||^2 - 2 d.q  -> length matters."""
        return -np.linalg.norm(self.R - self._qvec(q, "raw"), axis=1)

    def cos_euclid(self, q: str, alpha: float = 0.5, how: str = "weighted") -> np.ndarray:
        parts = [self.cos_nomic(q), self.euclid(q)]
        return weighted(parts, [alpha, 1 - alpha]) if how == "weighted" else rrf(parts)

    def bm25_scores(self, q: str) -> np.ndarray:
        return self.bm25.scores(q)

    def hybrid(self, q: str, how: str = "rrf", w_dense: float = 0.5) -> np.ndarray:
        parts = [self.cosine(q), self.bm25_scores(q)]
        return rrf(parts) if how == "rrf" else weighted(parts, [w_dense, 1 - w_dense])

    # --- query expansion (Phase 7): one Haiku call per question, cached ---
    def multi_query(self, q: str, fuse: str = "rrf") -> np.ndarray:
        from ragtraining.retrieve.expand import multi_query_scores
        return multi_query_scores(self, q, fuse)

    def hyde(self, q: str, dense_only: bool = False) -> np.ndarray:
        from ragtraining.retrieve.expand import hyde_scores
        return hyde_scores(self, q, dense_only)

    def scores(self, q: str, method: str, **kw) -> np.ndarray:
        return {"cosine": self.cosine, "cos_nomic": self.cos_nomic, "euclid": self.euclid,
                "cos_euclid": self.cos_euclid, "bm25": self.bm25_scores, "hybrid": self.hybrid,
                "multi_query": self.multi_query, "hyde": self.hyde}[method](q, **kw)

    # --- selection ---
    def search(self, q: str, method: str = "hybrid", k: int = 10, *, courses: list[str] | None = None,
               doc_types: list[str] | None = None, per_file_cap: int | None = None,
               use_type_weights: bool = True, mmr_lambda: float | None = None,
               rerank: str | None = None, rerank_top: int = 30, rerank_model: str | None = None,
               **kw) -> pd.DataFrame:
        if rerank:   # first retrieve a wider shortlist, then let the reranker re-sort it
            from ragtraining.retrieve.rerank import rerank_claude, rerank_local
            short = self.search(q, method, rerank_top, courses=courses, doc_types=doc_types,
                                per_file_cap=per_file_cap, use_type_weights=use_type_weights,
                                mmr_lambda=mmr_lambda, **kw)
            short["first_stage_rank"] = short["rank"]
            if rerank == "local":
                short["score"] = rerank_local(q, short.text.tolist())
            else:
                short["score"] = rerank_claude(q, short.text.tolist(), **({"model": rerank_model} if rerank_model else {}))
            if use_type_weights:   # keep D10 after reranking: summaries rank below official sources (P-29)
                short["score"] = short.score * short.doc_type.map(DOC_TYPE_WEIGHT).fillna(1.0)
            short = short.sort_values("score", ascending=False, kind="stable").head(k)
            short["rank"] = range(1, len(short) + 1)
            return short
        s = self.scores(q, method, **kw).astype(float)
        if use_type_weights:
            s = np.where(s > 0, s * self.type_w, s / self.type_w)    # down-weight works for negative scores too
        mask = np.ones(len(s), bool)
        if courses:
            mask &= self.chunks.course.isin(courses).to_numpy()
        if doc_types:
            mask &= self.chunks.doc_type.isin(doc_types).to_numpy()
        s[~mask] = -np.inf
        order = np.argsort(-s)[: max(k * 10, 100)]
        order = order[np.isfinite(s[order])]
        if mmr_lambda is not None:
            order = np.array(mmr(order, s, self.D, k * 3, mmr_lambda))
        picked, per_file = [], {}
        for i in order:
            f = self.chunks.path.iat[i]
            if per_file_cap and per_file.get(f, 0) >= per_file_cap:
                continue
            per_file[f] = per_file.get(f, 0) + 1
            picked.append(i)
            if len(picked) == k:
                break
        out = self.chunks.iloc[picked].copy()
        out.insert(0, "score", s[picked])
        out.insert(0, "rank", range(1, len(picked) + 1))
        return out
