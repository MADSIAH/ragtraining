"""Offline tests for evaluation code (no API calls): one per problem fixed in session 2."""
import numpy as np
import pandas as pd

from ragtraining.eval import guardrails, ragas_lite
from ragtraining.generate.tutor import REFUSE_BELOW


def _hits():
    return pd.DataFrame(dict(text=["Some slide text."], course=["Course A"],
                             path=["x/intro.pdf"], page_start=[1.0], page_end=[4.0],
                             section=[None], doc_type=["slides"], chunk_id=["hyb-1"]))


def test_p33_judge_sees_source_metadata(monkeypatch):
    seen = {}

    def fake_ask(system, user, schema, **kw):
        seen["user"] = user
        return dict(claims=[dict(claim="c", supported=True)], context_useful=[True],
                    reference_statements=[dict(statement="s", attributable=True)], regenerated_questions=["q"])

    monkeypatch.setattr(ragas_lite, "ask_json", fake_ask)
    monkeypatch.setattr(ragas_lite, "embed", lambda m, t, kind: np.ones((len(t), 2)) / np.sqrt(2))
    out = ragas_lite.judge("q", "a", _hits(), "ref")
    assert 'file="intro.pdf"' in seen["user"] and 'page="p.1-4"' in seen["user"]
    assert out["faithfulness"] == 1.0


def test_p37_vague_real_question_not_refused():
    # "What is a key?" scored 4/10 on the Claude reranker; it is covered, so the gate must let it through
    assert REFUSE_BELOW["claude"] <= 4


def test_p38_ambiguity_graded_by_senses_not_courses():
    case = next(c for c in guardrails.CASES if c["question"] == "What is regression?")
    sources = ["S1 Investments · a.pdf · p.1", "S2 NLP · b.pdf · p.2", "S3 Trading · c.pdf · p.3"]
    one_sense = dict(answer="Linear regression slope [S1][S2][S3]", answerable=True, sources=sources)
    both = dict(answer="Linear regression slope [S1]; regression testing re-runs tests [S2]", answerable=True,
                sources=sources)
    assert not guardrails._verdict(case, one_sense)[0]        # 3 courses, 1 meaning -> fail
    assert guardrails._verdict(case, both)[0]


def test_incremental_suite_reruns_only_gate_changes():
    case = guardrails.CASES[0]
    gate = REFUSE_BELOW["claude"]
    assert guardrails._needs_rerun(case, None)
    assert not guardrails._needs_rerun(case, dict(blocked="low_confidence", top_score=gate - 1, passed=True))
    assert guardrails._needs_rerun(case, dict(blocked="low_confidence", top_score=gate + 1, passed=True))


def test_manual_gold_refs_and_matching():                 # Max's questions: many gold refs per question
    import pandas as pd
    from ragtraining.eval.manual import gold_mask, parse_ref
    assert parse_ref("week07.pdf p.24-28") == dict(file="week07.pdf", p0=24, p1=28, needle=None)
    assert parse_ref("notes.md §Duration Gap")["needle"] == "Duration Gap"
    assert parse_ref("x.pdf p.3")["p1"] == 3
    chunks = pd.DataFrame(dict(path=["C/a/week07.pdf", "C/a/week07.pdf", "C/b/notes.md", "C/b/notes.md"],
                               course=["Course A", "Course A", "Course A", "Other"], page_start=[20, 25, None, None],
                               page_end=[21, 26, None, None], section=[None, None, "5. Duration Gap", "x"],
                               text=["...", "...", "...", "Duration Gap here"]))
    gold = [parse_ref("week07.pdf p.24-28"), parse_ref("notes.md §Duration Gap")]
    assert gold_mask(chunks, gold, ["Course A"]).tolist() == [False, True, False, False]    # section must START with it
    assert gold_mask(chunks, [parse_ref("notes.md §5. Duration")], ["Course A"]).tolist() == [False, False, True, False]
    assert gold_mask(chunks, gold, []).tolist() == [False, True, False, True]          # no course filter
