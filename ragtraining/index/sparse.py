"""Phase 2: sparse (keyword) index with BM25.

BM25 score of a chunk for a query = sum over query terms of
    IDF(term) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len/avg_len))
-> rare terms matter more (IDF), repeated terms saturate (k1), long chunks are penalized (b).
"""
from __future__ import annotations

import re

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

STOP = set("""a an the and or but if then else of to in on at by for with from as is are was were be been being
this that these those it its into than so such not no nor can could would should will shall may might must do does
did done have has had having which who whom whose what when where why how all any each both few more most other
some own same very just also only over under again further once here there about above below between through during
before after up down out off we you he she they them our your their i me my his her us""".split())
_stem = snowballstemmer.stemmer("english")
TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")   # keeps "2pl", "3nf", "bid-ask"


def tokenize(text: str, stem: bool = True) -> list[str]:
    toks = [t for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]
    return _stem.stemWords(toks) if stem else toks


class SparseIndex:
    def __init__(self, texts: list[str], stem: bool = True, k1: float = 1.5, b: float = 0.75):
        self.stem = stem
        self.bm25 = BM25Okapi([tokenize(t, stem) for t in texts], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(tokenize(query, self.stem))
