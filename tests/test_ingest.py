"""Regression tests for Phase 1. Each test pins down a problem from PROBLEMS_LOG.md."""
from pathlib import Path

import pandas as pd

from ragtraining.ingest.chunk import chunk_corpus, split_recursive, ntok
from ragtraining.ingest.discover import classify
from ragtraining.ingest.extract import clean, is_toc_or_index, ocr_quality_gate, strip_boilerplate


def c(path, course="X", pages=5):
    return classify(Path(path), course, pages)[0]


# ---- discovery (P-05..P-09) ----
def test_example_is_not_exam():                      # P-08
    assert c("Semester 1/Course A/codes/01code/example data/values.txt") == "code"
    assert c("Semester 2/Course B/old exams/midterm-A.pdf") == "exam"
    assert c("Semester 2/Course C/mock exam/practice_1.md") == "exam"


def test_own_and_classmate_work_excluded(monkeypatch):   # P-05, P-07 (own name comes from corpus_rules.local.json)
    import ragtraining.ingest.discover as discover
    monkeypatch.setattr(discover, "OWNER", discover.R("jane|doe"))
    assert c("Semester 2/Course B/HW/HW4/HW4 Jane Doe.pdf") == "personal"
    assert c("Semester 2/Course B/HW/HW4/HW4_John_Roe.pdf") == "personal"


def test_solution_and_assignment_by_name():          # P-06
    assert c("Semester 2/Course D/HW/l3-01.pdf") == "solution"
    assert c("Semester 2/Course D/HW/u3-01.pdf") == "problem_set"


def test_lecture_inside_hw_folder_is_slides():       # P-09
    assert c("Semester 2/Course E/HW/HW4/lecture05.pdf", pages=53) == "slides"


def test_textbook_by_pages():
    assert c("Semester 2/Course D/4110_notes_OA.pdf", pages=163) == "textbook"


# ---- extraction (P-11..P-16) ----
def test_clean_ligatures_and_hyphenation():          # P-16
    assert clean("ﬁnancial infor-\nmation") == "financial information"


def test_boilerplate_removed_but_numbered_titles_kept():   # P-12, P-21
    problems = [1, 1, 2, 2, 3, 3]
    pages = [f"Faculty of Science, Some University\nProblem {k}\nSome content here\nPage {i + 1} of 6"
             for i, k in enumerate(problems)]
    out = strip_boilerplate(pages)
    assert all("Faculty" not in p for p in out)                       # identical header removed
    assert all("Page" not in p for p in out)                          # page counter removed
    assert all(f"Problem {k}" in p for p, k in zip(out, problems))    # numbered titles kept


def test_page_counter_removed_even_with_stray_numbers():   # P-22
    eq = ["(2)", "(5)", "", "(1)", "(7)", "", "(4)", "(2)"]            # stray equation labels
    body = "A long paragraph of real content explaining the topic in detail, number {}."
    pages = [f"{body.format(i * 7 + 3)}\nMore text.\nEven more text.\n{eq[i]}\nClosing line.\n{100 + i}"
             for i in range(8)]
    out = strip_boilerplate(pages)
    assert all(str(100 + i) not in p.splitlines() for i, p in enumerate(out))   # page counter removed
    assert all(body.format(i * 7 + 3) in p for i, p in enumerate(out))          # content kept


def test_index_page_detected():                      # P-14
    index = " ".join(f"term{i}, {100 + i}–{105 + i}," for i in range(30))
    assert is_toc_or_index(index)
    assert not is_toc_or_index("The capital asset pricing model relates expected return to beta. " * 10)


def test_ocr_gate_drops_noise():                     # P-11
    assert ocr_quality_gate("J=6 | 6 »R | Ps > aR") == ""
    good = "\n".join(["Every asparagus that is not raw is not thin."] * 3)
    assert ocr_quality_gate(good)


# ---- chunking (P-18, P-20) ----
def _pages(n=20, words=120):
    return pd.DataFrame([dict(path="S/C/book.pdf", course="C", semester="S2", doc_type="textbook",
                              page=i + 1, section=None, ocr=False,
                              text=" ".join(f"w{i}_{j}" for j in range(words)) + ".")
                         for i in range(n)])


def test_recursive_respects_target():
    parts = split_recursive("Sentence one is here. " * 400, target=400)
    assert all(ntok(p) <= 400 for p in parts) and len(parts) > 1


def test_fixed_chunks_sized_and_paged():             # P-18 + P-20
    df = chunk_corpus(_pages(), "fixed")
    assert df.n_tokens.median() < 700                # not 2,000-token chunks
    assert df.page_start.nunique() > 3               # not everything on page 1
    first_page_of_text = df.text.map(lambda t: int(t.split("_")[0][1:]) + 1)
    assert (first_page_of_text == df.page_start).all()


def test_hybrid_merges_tiny_slides():
    slides = pd.DataFrame([dict(path="S/C/deck.pdf", course="C", semester="S2", doc_type="slides",
                                page=i + 1, section=None, ocr=False, text=f"Tiny slide {i} title")
                           for i in range(6)])
    df = chunk_corpus(slides, "hybrid")
    assert len(df) < 6 and (df.page_end >= df.page_start).all()


def test_merged_chunk_keeps_its_own_section_label():     # P-50
    # 3 tiny .md sections, then a long one: the merged chunk starts in "A" and must be labelled "A",
    # not with the section of the next (unmerged) chunk.
    secs = [("A", "alpha " * 20), ("B", "beta " * 20), ("C", "gamma " * 20), ("D", "delta " * 400)]
    units = pd.DataFrame([dict(path="S/C/notes.md", course="C", semester="S2", doc_type="exam", page=None,
                               section=s, ocr=False, text=f"## {s}\n{t}") for s, t in secs])
    df = chunk_corpus(units, "hybrid")
    word = dict(secs)
    assert df.section.iloc[0] == "A"
    for r in df.itertuples():           # the chunk's first content word belongs to the section it's labelled with
        first = r.text.split("\n", 1)[1] if r.text.startswith("## ") else r.text
        assert first.split()[0] == word[r.section].split()[0], (r.section, r.text[:20])
