"""Phase 7: does query expansion (multi-query, HyDE) beat the plain pipeline?  -> LOGBOOK §7.5

  1. retrieval metrics on the 60 synthetic questions (local rerank = free)
  2. sense coverage on the 4 ambiguous questions (P-38): do the top-8 chunks contain EVERY meaning?
     Also with the Claude reranker (production, D19), because a reranker scores against the original
     question and can undo the diversity that multi-query added.

Cost: 2 Haiku calls per question (cached in data/expansions.json) + 8 Claude reranks ~ $0.05.
Run: .venv/bin/python -m ragtraining.eval.advanced
     .venv/bin/python -m ragtraining.eval.advanced retrieval    (part 1 only, current test set -> eval_advanced_70q.csv)
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ragtraining.config import DATA_DIR
from ragtraining.eval.guardrails import CASES
from ragtraining.eval.retrieval import evaluate, load_testset
from ragtraining.llm import cost_so_far
from ragtraining.retrieve.expand import alt_queries, hypothetical_passage
from ragtraining.retrieve.search import Retriever

CFGS = {
    "cosine (Qwen3)": dict(method="cosine"),
    "HyDE dense": dict(method="hyde", dense_only=True),
    "hybrid RRF": dict(method="hybrid", how="rrf"),
    "multi-query hybrid (RRF fuse)": dict(method="multi_query"),
    "multi-query hybrid (round-robin)": dict(method="multi_query", fuse="rr"),
    "HyDE hybrid": dict(method="hyde"),
    "hybrid RRF + local rerank": dict(method="hybrid", how="rrf", rerank="local"),
    "multi-query RRF + local rerank": dict(method="multi_query", rerank="local"),
    "multi-query RR + local rerank": dict(method="multi_query", fuse="rr", rerank="local"),
    "HyDE hybrid + local rerank": dict(method="hyde", rerank="local"),
}
CLAUDE = dict(rerank="claude", rerank_model="claude-haiku-5-5")
SENSE_CFGS = {
    "hybrid RRF": CFGS["hybrid RRF"],
    "multi-query RRF": CFGS["multi-query hybrid (RRF fuse)"],
    "multi-query RR": CFGS["multi-query hybrid (round-robin)"],
    "hybrid + local rerank": CFGS["hybrid RRF + local rerank"],
    "multi-query RR + local rerank": CFGS["multi-query RR + local rerank"],
    "hybrid + Claude rerank (D19)": dict(method="hybrid", how="rrf", **CLAUDE),
    "multi-query RR + Claude rerank": dict(method="multi_query", fuse="rr", **CLAUDE),
}


def pregenerate(ret: Retriever, questions: list[str]) -> float:
    """Fill the expansion cache in parallel; returns mean seconds per Haiku call (the latency expansion adds)."""
    courses = sorted(ret.chunks.course.unique())

    def one(q):
        t0 = time.time(); alt_queries(q, courses); t1 = time.time(); hypothetical_passage(q)
        return (t1 - t0 + time.time() - t1) / 2

    with ThreadPoolExecutor(max_workers=8) as ex:
        secs = list(ex.map(one, questions))
    return sum(secs) / len(secs)


def sense_coverage(ret: Retriever) -> pd.DataFrame:
    rows = []
    for case in [c for c in CASES if c["kind"] == "ambiguous"]:
        for name, cfg in SENSE_CFGS.items():
            hits = ret.search(case["question"], k=8, **cfg)
            text = " ".join(hits.text).lower()
            covered = [s for s, ms in case["senses"].items() if any(m in text for m in ms)]
            rows.append(dict(question=case["question"], config=name, covered=len(covered), of=len(case["senses"]),
                             courses=hits.course.nunique()))
    return pd.DataFrame(rows)


if __name__ == "__main__" and sys.argv[1:] == ["retrieval"]:
    # S4: retrieval part only, on the current test set (58 synthetic + Max's 12, split by source). Only Max's
    # questions need new expansions (~24 Haiku calls). The Claude-rerank sense check is skipped (already in §7.5).
    ret, tests = Retriever(), load_testset()
    sec = pregenerate(ret, list(tests.question))
    print(f"expansions ready: {sec:.1f} s per call (mostly cached), cost so far ${cost_so_far():.3f}")
    summary, per_q = evaluate(ret, tests, CFGS)
    summary.to_csv(DATA_DIR / f"eval_advanced_{len(tests)}q.csv", index=False)
    per_q.to_csv(DATA_DIR / f"eval_advanced_{len(tests)}q_perq.csv")
    print("\n", summary.to_string(index=False), f"\n\ntotal Claude cost this run: ${cost_so_far():.3f}")
elif __name__ == "__main__":
    ret, tests = Retriever(), load_testset()
    amb = [c["question"] for c in CASES if c["kind"] == "ambiguous"]
    sec = pregenerate(ret, list(tests.question) + amb)
    print(f"expansions ready: {sec:.1f} s per Haiku call (cached afterwards), cost so far ${cost_so_far():.3f}")
    summary, per_q = evaluate(ret, tests, CFGS)
    summary.to_csv(DATA_DIR / "eval_advanced.csv", index=False)
    per_q.to_csv(DATA_DIR / "eval_advanced_perq.csv")
    sc = sense_coverage(ret)
    sc.to_csv(DATA_DIR / "eval_ambiguous_senses.csv", index=False)
    print("\n", summary.to_string(index=False))
    print("\nsenses covered in top-8 (ambiguous questions):\n",
          sc.pivot(index="config", columns="question", values="covered").to_string())
    print(f"\ntotal Claude cost this run: ${cost_so_far():.3f}")
