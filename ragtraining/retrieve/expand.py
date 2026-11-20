"""Phase 7: query expansion BEFORE retrieval (Haiku 5.5, cheap; generations cached in data/expansions.json).

  multi-query  Haiku writes 3 alternative searches (other wording / the terms a slide would use; for an
               ambiguous term, one search per meaning). Each is searched with hybrid RRF and the rankings
               (original question included) are fused with RRF again.
  HyDE         "Hypothetical Document Embeddings": Haiku writes a short passage that WOULD answer the question.
               We embed it as a document and search with it, so we compare answer-to-answer instead of
               question-to-answer. The passage may contain wrong details; it's only used to search, never shown.

A reranker afterwards still scores the pooled shortlist against the ORIGINAL question.
"""
from __future__ import annotations

import json
import threading

import numpy as np

from ragtraining.config import DATA_DIR
from ragtraining.index.build import DENSE_MODEL
from ragtraining.index.embed import embed
from ragtraining.llm import ask_json

MODEL = "claude-haiku-5-5"
CACHE_F = DATA_DIR / "expansions.json"
_lock = threading.Lock()
_cache: dict[str, dict] | None = None

MQ_SYS = """You turn a student's question into search queries over their own course material.
Courses: {courses}
Return exactly 3 short queries (max 12 words each). Use different wording, synonyms and the technical terms a
textbook or lecture slide would use. If a term in the question means different things in different courses
(e.g. "key" in cryptography vs. databases), give each meaning its own query."""
MQ_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["queries"],
             "properties": {"queries": {"type": "array", "items": {"type": "string"}}}}

HYDE_SYS = """Write a short passage (80-120 words) as it could appear in a university textbook or lecture slide,
answering the student's question. Use the technical terms an expert would use. No introduction, no hedging.
It is only used as a search query, never shown to anyone."""
HYDE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["passage"],
               "properties": {"passage": {"type": "string"}}}


def _load() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_F.read_text()) if CACHE_F.exists() else {}
    return _cache


def _cached(kind: str, q: str, make) -> dict:
    key = f"{kind}::{q}"
    with _lock:
        hit = _load().get(key)
    if hit is not None:
        return hit
    out = make()
    with _lock:
        _load()[key] = out
        CACHE_F.write_text(json.dumps(_cache, indent=1, ensure_ascii=False))
    return out


def alt_queries(q: str, courses: list[str]) -> list[str]:
    out = _cached("mq", q, lambda: ask_json(MQ_SYS.format(courses=", ".join(courses)), f"<question>{q}</question>",
                                            MQ_SCHEMA, model=MODEL, effort="low", max_tokens=2000))
    return out["queries"][:3]


def hypothetical_passage(q: str) -> str:
    return _cached("hyde", q, lambda: ask_json(HYDE_SYS, f"<question>{q}</question>", HYDE_SCHEMA,
                                               model=MODEL, effort="low", max_tokens=2000))["passage"]


def multi_query_scores(r, q: str, fuse: str = "rrf", depth: int = 100) -> np.ndarray:
    """fuse='rrf': sum of 1/(60+rank) -> rewards chunks that SEVERAL queries agree on (= the majority meaning).
    fuse='rr' (round-robin): each query's next-best chunk in turn -> every phrasing/meaning gets a share of the top."""
    from ragtraining.retrieve.search import rrf
    queries = [q] + alt_queries(q, sorted(r.chunks.course.unique()))
    parts = [r.hybrid(x) for x in queries]
    if fuse == "rrf":
        return rrf(parts)
    orders = [np.argsort(-p)[:depth] for p in parts]
    out, pos = np.zeros(len(parts[0])), 0
    for rank in range(depth):
        for order in orders:
            i = order[rank]
            if out[i] == 0:
                out[i] = 1.0 / (1 + pos)
                pos += 1
    return out


def hyde_scores(r, q: str, dense_only: bool = False) -> np.ndarray:
    from ragtraining.retrieve.search import rrf
    hv = embed(DENSE_MODEL, [hypothetical_passage(q)], "doc")[0]            # embedded as a DOCUMENT, not a query
    dense = r.D @ hv
    return dense if dense_only else rrf([dense, r.bm25_scores(q)])
