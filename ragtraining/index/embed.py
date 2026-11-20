"""Phase 2: local embeddings through Ollama.

Two endpoints matter (PROBLEMS_LOG P-23):
  /api/embed       (ollama.embed)      -> batched, but returns L2-NORMALIZED vectors (length 1)
  /api/embeddings  (ollama.embeddings) -> one text per call, returns RAW vectors (length varies)
For cosine search normalized vectors are all we need. For the D1 Euclidean experiment we need RAW ones.
"""
from __future__ import annotations

import numpy as np
import ollama

# Each model was trained with its own instruction prefixes. Using the wrong ones silently hurts quality.
MODELS = {
    "nomic-embed-text": dict(dim=768, doc="search_document: ", query="search_query: ", ctx=2048),
    "mxbai-embed-large": dict(dim=1024, doc="",
                              query="Represent this sentence for searching relevant passages: ", ctx=512),
    "bge-m3": dict(dim=1024, doc="", query="", ctx=8192),
    "qwen3-embedding:0.6b": dict(dim=1024, doc="",
                                 query="Instruct: Given a student's question, retrieve course material "
                                       "passages that answer it\nQuery: ", ctx=8192),
    "embeddinggemma": dict(dim=768, doc="title: none | text: ", query="task: search result | query: ", ctx=2048),
}


def _with_prefix(model: str, texts: list[str], kind: str) -> list[str]:
    p = MODELS[model][kind]
    return [p + t for t in texts]


def _embed_one_safe(model: str, text: str, raw: bool) -> list[float]:
    """Embed one text; if it exceeds the model's context, cut 25% of the characters and retry (P-24)."""
    for _ in range(6):
        try:
            if raw:
                return ollama.embeddings(model=model, prompt=text).embedding
            return ollama.embed(model=model, input=text, truncate=True).embeddings[0]
        except ollama.ResponseError as e:
            if "context length" not in str(e):
                raise
            text = text[: int(len(text) * 0.75)]
    raise RuntimeError(f"could not fit text into {model} context")


def embed(model: str, texts: list[str], kind: str = "doc", raw: bool = False,
          batch: int = 64, progress: bool = False) -> np.ndarray:
    """Return an (n, dim) float32 matrix. kind = 'doc' | 'query'. raw=True -> un-normalized vectors."""
    texts = _with_prefix(model, texts, kind)
    out: list[list[float]] = []
    step = 1 if raw else batch
    for i in range(0, len(texts), step):
        chunk = texts[i:i + step]
        if raw:
            out.append(_embed_one_safe(model, chunk[0], raw=True))
        else:
            try:
                out.extend(ollama.embed(model=model, input=chunk, truncate=True).embeddings)
            except ollama.ResponseError:                # one bad text in the batch -> go one by one
                out.extend(_embed_one_safe(model, t, raw=False) for t in chunk)
        if progress and (i // step) % 50 == 0:
            print(f"  {model}: {i + len(chunk)}/{len(texts)}", flush=True)
    return np.asarray(out, dtype=np.float32)
