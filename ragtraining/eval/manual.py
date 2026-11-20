"""Phase 4: Max's own test questions (testset/manual_questions.md) + the synthetic spot-check (synthetic_review.md).

manual_questions.md is the single source of truth (human-editable). Each question has:
  kind             answerable | vague (still answerable) | not_covered (tutor must refuse)
  course           one or more courses, ';'-separated (gold sources are only matched inside them)
  expected source  ';'-separated gold refs:  "<file-name part> p.<a>-<b>"  or  "<file-name part> §<text>"
  key points       reference answer (RAGAS context recall)

Relevance (works for every chunking strategy, LOGBOOK §7.6): a chunk is relevant if it matches ANY gold ref:
  page ref   its file name contains the part AND its pages overlap a..b
  § ref      its file name contains the part AND (its section label starts with the text OR its text contains it)
Unlike the synthetic items (one gold passage each), a question here can have many valid sources.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ragtraining.config import ROOT

MANUAL = ROOT / "testset" / "manual_questions.md"
REVIEW = ROOT / "testset" / "synthetic_review.md"
KINDS = {"answerable", "vague", "not_covered"}
_REF = re.compile(r"^(?P<file>.+?)\s+(?:p\.(?P<p0>\d+)(?:-(?P<p1>\d+))?|§(?P<needle>.+))$")


def parse_ref(ref: str) -> dict:
    m = _REF.match(ref.strip())
    if not m:
        raise ValueError(f"bad gold ref {ref!r}: use '<file> p.<a>-<b>' or '<file> §<text>'")
    p0 = int(m["p0"]) if m["p0"] else None
    return dict(file=m["file"].strip(), p0=p0, p1=int(m["p1"]) if m["p1"] else p0,
                needle=m["needle"].strip() if m["needle"] else None)


def load_manual(path=MANUAL) -> pd.DataFrame:
    rows = []
    for block in re.split(r"^## (M\d+)\s*$", path.read_text(), flags=re.M)[1:]:
        if re.fullmatch(r"M\d+", block):
            rows.append(dict(qid=block)); continue
        f = dict(re.findall(r"^(question|course|kind|expected source|key points):[ \t]*(.*)$", block, re.M))
        if not f.get("question"):
            rows.pop(); continue                                    # empty template block
        kind = f.get("kind", "answerable").strip()
        if kind not in KINDS:
            raise ValueError(f"{rows[-1]['qid']}: kind {kind!r} not in {KINDS}")
        refs = [parse_ref(r) for r in f.get("expected source", "").split(";") if r.strip()]
        if kind != "not_covered" and not refs:
            raise ValueError(f"{rows[-1]['qid']}: an answerable question needs at least one expected source")
        courses = [c.strip() for c in f.get("course", "").split(";") if c.strip() and c.strip() != "-"]
        rows[-1].update(question=f["question"].strip(), courses=courses, kind=kind, gold=refs,
                        answer=f.get("key points", "").strip(), source="manual")
    return pd.DataFrame(rows)


def gold_mask(chunks: pd.DataFrame, gold: list[dict], courses: list[str]) -> np.ndarray:
    """True for every chunk that matches at least one gold ref (see module docstring)."""
    name = chunks.path.str.rsplit("/", n=1).str[-1]
    in_course = chunks.course.isin(courses).to_numpy() if courses else np.ones(len(chunks), bool)
    hit = np.zeros(len(chunks), bool)
    for g in gold:
        same = name.str.contains(g["file"], regex=False).to_numpy() & in_course
        if g["needle"] is not None:
            m = chunks.section.fillna("").str.startswith(g["needle"]) | chunks.text.str.contains(g["needle"], regex=False)
        else:
            m = (chunks.page_start <= g["p1"]) & (chunks.page_end >= g["p0"])
        hit |= same & m.fillna(False).to_numpy()
    return hit


def check_refs(chunks: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """Every gold ref must match >= 1 chunk (catches typos in file names / page ranges)."""
    rows = []
    for t in tests.itertuples():
        for g in t.gold:
            n = int(gold_mask(chunks, [g], t.courses).sum())
            rows.append(dict(qid=t.qid, ref=f"{g['file']} " + (f"§{g['needle']}" if g["needle"] else f"p.{g['p0']}-{g['p1']}"),
                             chunks=n))
    return pd.DataFrame(rows)


def review_verdicts(path=REVIEW) -> dict[str, str]:
    """question -> 'OK' / 'BAD' / '' from the spot-check file (an 'x' in the OK or BAD box)."""
    out = {}
    for block in re.split(r"^## S\d+.*$", path.read_text(), flags=re.M)[1:]:
        q = re.search(r"^\*\*Q:\*\*\s*(.+)$", block, re.M)
        if not q:
            continue
        ok = re.search(r"- \[[xX]\] OK", block)
        bad = re.search(r"- \[[xX]\] BAD", block)
        out[q.group(1).strip()] = "BAD" if bad else "OK" if ok else ""
    return out
