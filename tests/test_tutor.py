"""P-53 citation retry + P11 regression gate. No model calls: ask_json is replaced by a fake."""
import json

import pandas as pd

from ragtraining.eval import run_ragas
from ragtraining.generate import tutor as tutor_mod
from ragtraining.generate.tutor import CITE_REMINDER, Tutor

HITS = pd.DataFrame(dict(text=["A Feistel round swaps halves.", "DES uses 16 rounds."], page_start=[3, 4],
                         page_end=[3, 4], section=[None, None], course=["InfoSec", "InfoSec"],
                         path=["a/slides.pdf", "a/slides.pdf"], doc_type=["slides", "slides"], score=[0.9, 0.8]))


def _fake(replies, calls):
    def ask_json(system, user, schema, **kw):
        calls.append(user)
        return replies[len(calls) - 1]
    return ask_json


def _answer(monkeypatch, replies):
    calls = []
    monkeypatch.setattr(tutor_mod, "ask_json", _fake(replies, calls))
    return Tutor(retriever=None).answer("How does DES work?", hits=HITS, rerank="none"), calls


def test_cited_answer_no_retry(monkeypatch):
    res, calls = _answer(monkeypatch, [dict(answerable=True, answer="- **16 rounds** [S2]")])
    assert len(calls) == 1 and not res["citation_retry"] and res["citation_problems"] == []


def test_uncited_answer_retried_and_fixed(monkeypatch):                # P-53: Review line is not a citation
    res, calls = _answer(monkeypatch, [dict(answerable=True, answer="- **16 rounds**\nReview: S2, p.4"),
                                       dict(answerable=True, answer="- **16 rounds** [S2]")])
    assert len(calls) == 2 and calls[1].endswith(CITE_REMINDER)
    assert res["citation_retry"] and res["citation_problems"] == [] and "[S2]" in res["answer"]


def test_retry_that_refuses_keeps_first_answer(monkeypatch):           # a refusal on retry = false refusal
    first = dict(answerable=True, answer="- **16 rounds**")
    res, _ = _answer(monkeypatch, [first, dict(answerable=False, answer=tutor_mod.NOT_COVERED)])
    assert res["answer"] == first["answer"] and res["answerable"]
    assert res["citation_retry"] and res["citation_problems"] == ["answer has no citations"]


def test_refusal_is_not_retried(monkeypatch):
    res, calls = _answer(monkeypatch, [dict(answerable=False, answer=tutor_mod.NOT_COVERED)])
    assert len(calls) == 1 and not res["citation_retry"]


def _summary(**kw):
    s = dict(n=70, faithfulness=0.953, context_recall=0.967, context_precision=0.873, answer_relevancy=0.773,
             answer_errors=0, citation_problems=1, citation_problem_answers=1)
    return s | kw


def _gate(tmp_path, monkeypatch, summary):
    (tmp_path / "ragas_x.json").write_text(json.dumps(dict(summary=summary, rows=[])))
    monkeypatch.setattr(run_ragas, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_ragas, "load_testset", lambda: pd.DataFrame(dict(q=range(70))))
    return run_ragas.gate("x")


def test_gate(tmp_path, monkeypatch):
    assert _gate(tmp_path, monkeypatch, _summary())                                       # S3 scores, 1/70 = 1%
    assert not _gate(tmp_path, monkeypatch, _summary(citation_problem_answers=8))        # 11% > 3%
    assert not _gate(tmp_path, monkeypatch, _summary(faithfulness=0.90))                  # real drop
    assert _gate(tmp_path, monkeypatch, _summary(answer_relevancy=0.60))                  # soft bar: warns only
    assert not _gate(tmp_path, monkeypatch, _summary(answer_errors=1))
