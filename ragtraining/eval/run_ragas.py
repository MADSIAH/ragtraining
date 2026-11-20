"""Phase 6: run RAGAS-style evaluation on N test questions for one pipeline.

Pipelines:
  baseline    dense cosine only (Qwen3), no rerank
  prod_local  hybrid RRF + local cross-encoder rerank (what session 1 called "prod")
  prod        hybrid RRF + Claude rerank (Haiku), the D19 default
  agent       agentic loop (plan -> search -> grade -> re-search) + tutor
  local       the D22 default engine: hybrid RRF + local rerank, answers by qwen3:14b on this Mac ($0)

Answers: Opus 5.5 (every pipeline except `local`). Judge: Haiku 5.5 (D17) for all, so they stay comparable.
Run:      .venv/bin/python -m ragtraining.eval.run_ragas prod 20        (random 20, seed 42 -> data/ragas_prod.json)
          .venv/bin/python -m ragtraining.eval.run_ragas local all      (every question -> data/ragas_local_all.json)
          .venv/bin/python -m ragtraining.eval.run_ragas prod manual    (Max's questions -> data/ragas_prod_manual.json)
Re-judge: .venv/bin/python -m ragtraining.eval.run_ragas rejudge prod
          (scores the STORED answers again with the current judge: Haiku cost only, no new Opus answers)
Patch:    .venv/bin/python -m ragtraining.eval.run_ragas patch local_all
          (re-answers only rows with citation problems/errors on the same evidence -> ragas_local_all_patched.json)
Gate:     .venv/bin/python -m ragtraining.eval.run_ragas gate local_all      (P11 pass/fail bars, exit code 1 on FAIL)

Every row stores the chunk_ids the tutor answered from, so a re-judge rebuilds exactly the same
evidence. Session-1 files have no chunk_ids: their evidence is re-retrieved (deterministic for
baseline/prod_local; the agent's Haiku planner is not deterministic, so agent files can't be re-judged).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ragtraining.config import CLAUDE_MODEL, DATA_DIR, LOCAL_MODEL
from ragtraining.eval.ragas_lite import judge
from ragtraining.eval.retrieval import load_testset
from ragtraining.generate.tutor import Tutor
from ragtraining.llm import USAGE, cost_so_far

PIPELINES = {"baseline": dict(method="cosine"),
             "prod_local": dict(method="hybrid", how="rrf", rerank="local"),
             "prod": dict(method="hybrid", how="rrf", rerank="claude", rerank_model="claude-haiku-5-5"),
             "local": dict(method="hybrid", how="rrf", rerank="local")}
ANSWER_MODEL = {"local": LOCAL_MODEL}                           # every other pipeline answers with Opus
LEGACY = {"baseline": "baseline", "prod": "prod_local"}     # session-1 file name -> retrieval it used
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _summary(df: pd.DataFrame, **extra) -> dict:
    return dict(**extra, n=len(df), **{m: round(float(np.nanmean(df[m])), 3) for m in METRICS},
                cost_usd=round(cost_so_far(), 3))


def _safe(fn, *args, **kw) -> dict:
    """One failing question (e.g. a local answer hitting the output cap, P-43) must not kill a 30-min run (P-47)."""
    try:
        return fn(*args, **kw)
    except Exception as e:                                                        # noqa: BLE001 - recorded per row
        return dict(error=f"{type(e).__name__}: {e}")


def _score_all(questions, answers, evid, references) -> list[dict]:
    def one(i):
        if answers[i] is None:                                                    # the answer itself failed
            return {m: np.nan for m in METRICS} | dict(error="no answer")
        return _safe(judge, questions[i], answers[i], evid[i], references[i])
    with ThreadPoolExecutor(max_workers=4) as ex:
        out = list(ex.map(one, range(len(questions))))
    return [{m: np.nan for m in METRICS} | s for s in out]


def main(pipeline: str, n: int | str, testset: pd.DataFrame | None = None, name: str | None = None):
    """n = an int (random sample, seed 42) · "all" (every test question) · "manual" (only Max's questions)."""
    if testset is None:
        testset = load_testset()
        testset = testset[testset.source == "manual"] if n == "manual" else \
            testset if n == "all" else testset.sample(int(n), random_state=42)
    tests = testset.reset_index(drop=True)
    n_q = len(tests)
    name = name or (pipeline if isinstance(n, int) or str(n).isdigit() else f"{pipeline}_{n}")   # never overwrite S1 files
    tutor = Tutor(model=ANSWER_MODEL.get(pipeline, CLAUDE_MODEL))
    if pipeline == "agent":
        from ragtraining.agent.loop import gather
        evid = [gather(q, tutor)[0] for q in tests.question]                     # sequential (local reranker)
    else:
        evid = [tutor.retriever.search(q, k=tutor.k, **PIPELINES[pipeline]) for q in tests.question]

    workers = 1 if tutor.model.startswith("ollama:") else 4                       # one local model: no point queuing
    with ThreadPoolExecutor(max_workers=workers) as ex:
        answers = list(ex.map(lambda i: _safe(tutor.answer, tests.question[i], hits=evid[i], rerank="none"), range(n_q)))
    (DATA_DIR / f"ragas_{name}_answers.json").write_text(json.dumps(answers, indent=1, default=str))  # saved before judging
    errors = [a for a in answers if "error" in a]
    print(f"answers: {n_q - len(errors)}/{n_q} ok, cost so far ${cost_so_far():.3f}", flush=True)

    scores = _score_all(list(tests.question), [None if "error" in a else a["answer"] for a in answers], evid,
                        list(tests.answer))
    src = tests.get("source", pd.Series(["synthetic"] * n_q)).fillna("synthetic").to_numpy()
    summary = _full_summary(scores, answers, src, pipeline=pipeline, retrieval=PIPELINES.get(pipeline, "agent"),
                            answer_model=tutor.model,
                            local_loop_retries=sum(v for (_, k), v in USAGE.items() if k == "loop_retries"))
    rows = [dict(question=q, source=s, answer=a.get("answer"), answer_error=a.get("error"),
                 chunk_ids=e.chunk_id.tolist(), sources=a.get("sources"), **sc)
            for q, s, a, e, sc in zip(tests.question, src, answers, evid, scores)]
    (DATA_DIR / f"ragas_{name}.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1, default=str))
    print(json.dumps(summary))


def _full_summary(scores: list[dict], answers: list[dict], src, **extra) -> dict:
    df = pd.DataFrame(scores)
    errors = sum("error" in a for a in answers)
    n_cite = [len(a.get("citation_problems") or []) for a in answers]
    src = np.asarray(src)
    by_source = {g: {m: round(float(np.nanmean(df[m][src == g])), 3) for m in METRICS} | dict(n=int((src == g).sum()))
                 for g in sorted(set(src))}
    return _summary(df, **extra, answered=f"{sum(bool(a.get('answerable')) for a in answers)}/{len(answers)}",
                    answer_errors=errors,
                    judge_errors=int(df.get("error", pd.Series(dtype=str)).notna().sum()) - errors,
                    citation_problems=sum(n_cite), citation_problem_answers=sum(n > 0 for n in n_cite),
                    citation_retries=sum(bool(a.get("citation_retry")) for a in answers), by_source=by_source)


def patch(name: str):
    """Re-answer ONLY the rows of data/ragas_<name>.json whose stored answer had a citation problem or an error,
    with the CURRENT tutor on the SAME stored evidence (chunk_ids), re-judge only those rows, merge them back
    and write data/ragas_<name>_patched.json. Cost = the judge on those rows only (local answers are $0)."""
    old = json.loads((DATA_DIR / f"ragas_{name}.json").read_text())
    answers = json.loads((DATA_DIR / f"ragas_{name}_answers.json").read_text())
    rows = old["rows"]
    redo = [i for i, a in enumerate(answers) if a.get("citation_problems") or "error" in a]
    tutor = Tutor(model=old["summary"]["answer_model"])
    chunks = tutor.retriever.chunks.set_index("chunk_id", drop=False)
    tests = load_testset(drop_bad=False).set_index("question")
    evid = [chunks.loc[rows[i]["chunk_ids"]].reset_index(drop=True).assign(score=answers[i].get("top_score", np.nan))
            for i in redo]
    print(f"re-answering {len(redo)} rows: {redo}", flush=True)
    new = [_safe(tutor.answer, rows[i]["question"], hits=e, rerank="none") for i, e in zip(redo, evid)]
    qs = [rows[i]["question"] for i in redo]
    scores = _score_all(qs, [None if "error" in a else a["answer"] for a in new], evid, [tests.answer[q] for q in qs])
    for i, a, s in zip(redo, new, scores):
        answers[i] = a
        rows[i] = {**{k: v for k, v in rows[i].items() if k != "error"}, "answer": a.get("answer"), "answer_error": a.get("error"), "patched": True,
                   "old_answer": rows[i]["answer"], **s}
    all_scores = [{m: r.get(m, np.nan) for m in METRICS} | ({"error": r["error"]} if r.get("error") else {})
                  for r in rows]
    summary = _full_summary(all_scores, answers, [r["source"] for r in rows],
                            **{k: old["summary"][k] for k in ("pipeline", "retrieval", "answer_model")},
                            patched_from=f"ragas_{name}.json", patched_rows=redo,
                            old={m: old["summary"][m] for m in METRICS + ["citation_problems"]})
    summary["cost_usd"] = round(cost_so_far(), 3)                                # this patch only
    (DATA_DIR / f"ragas_{name}_patched_answers.json").write_text(json.dumps(answers, indent=1, default=str))
    (DATA_DIR / f"ragas_{name}_patched.json").write_text(
        json.dumps(dict(summary=summary, rows=rows), indent=1, default=str))
    print(json.dumps(summary))


# P11 (D25, Max S4): regression gate. Measured on ALL test questions (12 are too few). Each bar sits at least one
# judge-noise step (±0.03, P-36) below the local engine's S3 score, so a real drop fails, noise doesn't.
GATE = {"faithfulness": 0.92, "context_recall": 0.90, "context_precision": 0.80, "answer_relevancy": 0.65}
GATE_SOFT = {"answer_relevancy"}          # noisy and low on vague questions (§7.3b): warns, doesn't fail
MAX_CITATION_RATE = 0.03                  # share of answers with any citation problem (G2)


def gate(name: str) -> bool:
    """PASS/FAIL table for data/ragas_<name>.json against the P11 bars. Returns True if every hard bar passes."""
    s = json.loads((DATA_DIR / f"ragas_{name}.json").read_text())["summary"]
    n_bad = s.get("citation_problem_answers", s["citation_problems"])       # older files: count of problems
    checks = [(m, f"{s[m]:.3f}", f">= {t}", s[m] >= t, m in GATE_SOFT) for m, t in GATE.items()]
    checks += [("answer errors", str(s["answer_errors"]), "= 0", s["answer_errors"] == 0, False),
               ("citation problems", f"{n_bad}/{s['n']} = {n_bad / s['n']:.0%}", f"<= {MAX_CITATION_RATE:.0%}",
                n_bad / s["n"] <= MAX_CITATION_RATE, False)]
    for metric, value, bar, ok, soft in checks:
        print(f"{'PASS' if ok else ('warn' if soft else 'FAIL'):5} {metric:18} {value:>14}  (bar {bar})")
    n_all = len(load_testset())
    if s["n"] < n_all:
        print(f"note: {s['n']} questions, the gate is meant for all {n_all}")
    passed = all(ok for *_, ok, soft in checks if not soft)
    print("GATE:", "PASS" if passed else "FAIL")
    return passed


def rejudge(name: str):
    """Score the answers stored in data/ragas_<name>.json again with the current judge."""
    src = DATA_DIR / f"ragas_{name}.json"
    old = json.loads(src.read_text())
    rows = old["rows"]
    tests = load_testset(drop_bad=False).set_index("question")   # S1 files may hold items later marked BAD
    tutor = Tutor(model=CLAUDE_MODEL)       # only its retriever/chunks are used here
    chunks = tutor.retriever.chunks.set_index("chunk_id", drop=False)
    if all("chunk_ids" in r for r in rows):
        evid = [chunks.loc[r["chunk_ids"]].reset_index(drop=True) for r in rows]
    elif name in LEGACY:
        cfg = PIPELINES[LEGACY[name]]
        evid = [tutor.retriever.search(r["question"], k=tutor.k, **cfg) for r in rows]
    else:
        raise SystemExit(f"{src.name} has no stored chunk_ids and its retrieval isn't reproducible.")

    qs = [r["question"] for r in rows]
    scores = _score_all(qs, [r["answer"] for r in rows], evid, [tests.answer[q] for q in qs])
    df = pd.DataFrame(scores)
    summary = _summary(df, pipeline=name, rejudged_from=src.name,
                       old={m: old["summary"][m] for m in METRICS})
    out_rows = [dict(question=r["question"], answer=r["answer"], chunk_ids=e.chunk_id.tolist(),
                     old_unsupported=r.get("unsupported"), **s) for r, e, s in zip(rows, evid, scores)]
    (DATA_DIR / f"ragas_{name}_rejudged.json").write_text(
        json.dumps(dict(summary=summary, rows=out_rows), indent=1, default=str))
    print(json.dumps(summary))


if __name__ == "__main__":
    if sys.argv[1] == "rejudge":
        rejudge(sys.argv[2])
    elif sys.argv[1] == "patch":
        patch(sys.argv[2])
    elif sys.argv[1] == "gate":
        sys.exit(0 if gate(sys.argv[2]) else 1)
    else:
        main(sys.argv[1], sys.argv[2] if sys.argv[2] in {"all", "manual"} else int(sys.argv[2]))
