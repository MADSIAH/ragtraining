"""Phase 1, step 1: find every course document, classify it, and write data/manifest.csv.

Each file gets a doc_type and a keep decision:
  keep = "yes"    -> goes into the RAG
  keep = "no"     -> excluded (personal work, other students, admin, code/data, duplicates)
  keep = "review" -> bucket that Max decides on (see LOGBOOK P2)

Precedence of the rules = order of the checks in classify(). Bucket decisions made by Max
are applied in BUCKET_DECISIONS at the bottom.

Corpus-specific rules (own name, misplaced files, excluded courses, ...) live in the git-ignored
corpus_rules.local.json, so no names or file names from the private corpus are in the code.
Template: corpus_rules.example.json.

Run:  .venv/bin/python -m ragtraining.ingest.discover
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import pymupdf

from ragtraining.config import CORPUS_DIR, DATA_DIR, ROOT

NOISE_DIRS = {"node_modules", ".venv", "venv", "env", "site-packages", "__pycache__",
              ".git", "lib", ".ipynb_checkpoints", "dist", "build"}
CONTENT_EXT = {".pdf", ".md", ".docx", ".doc", ".pptx", ".tex", ".txt", ".ipynb"}
TEXTBOOK_MIN_PAGES = 150
RULES_FILE = next((p for p in (ROOT / "corpus_rules.local.json", ROOT / "corpus_rules.example.json") if p.exists()), None)
LOCAL = json.loads(RULES_FILE.read_text()) if RULES_FILE else {}


def _alt(base: str, key: str) -> str:
    """base regex, plus the corpus-specific alternatives from corpus_rules.local.json (if any)."""
    return f"{base}|{LOCAL[key]}" if LOCAL.get(key) else base


COURSE_OVERRIDES = LOCAL.get("course_overrides", {})  # file name -> real course, for misplaced files (P-30)
EXCLUDED_COURSES = LOCAL.get("excluded_courses", {})  # course -> reason (D10: language course)
DROP_FILES = LOCAL.get("drop_files", {})              # name part -> reason (content duplicates with other names)

R = lambda p: re.compile(p, re.I)  # noqa: E731
OWNER = R(LOCAL.get("owner_pattern") or r"(?!)")                   # own work (never matches if unset)
OTHER_STUDENT = R(r"HW\d+_[A-Z][a-z]+_[A-Z][a-z]+|student-\d+")    # classmates' submissions
ADMIN = R(_alt(r"schedule|regulation|template|guidelines", "extra_admin_pattern"))
TEXTBOOK_NAME = R(_alt(r"book", "textbook_name_pattern"))
PERSONAL_NAME = R(r"proposal|report|group\d+|pitch.?deck|portfolio|script|for_dummies|summary_for|copy\b")
WORKING_FILE = R(r"drawio|correction|explanation|table_creation|processed|spool|diagram")

FOLDER_RULES = [  # folder name -> doc_type, deepest folder first
    ("personal", R(_alt(r"portfolio|github|^codes?$|\d+code|student-\d+|create_copies|diagrams|"
                        r"presentation|round\d|^report$", "extra_personal_folder_pattern"))),
    ("project", R(r"project|assingment|assignment|case\d")),
    ("exam", R(r"(?<![a-z])exams?(?![a-z])|midterm")),   # not "example"
    ("summary", R(r"summar|cheat|formula")),
    ("problem_set", R(r"problem.?set|exercis|esercizi|lab\d*_solutions")),
    ("homework", R(r"^hw\d*$")),
    ("slides", R(r"lecture|slide|theory|syllabus")),
]
NAME_RULES = [  # file name -> doc_type
    ("solution", R(r"solution|_sol\b|_sol\.|soluzion")),
    ("exam", R(r"(?<![a-z])exams?(?![a-z])|midterm|mock")),
    ("problem_set", R(r"problem.?set|\bps\d|exercis|_ex\d")),
    ("summary", R(r"summar|cheat|formula")),
    ("slides", R(r"slide|lecture|class \d|session|topic\d|chapter|week")),
]
KEEP = {"slides": "yes", "textbook": "yes", "problem_set": "yes", "solution": "yes", "exam": "yes",
        "project": "yes",
        "summary": "yes",      # D10: kept, but tagged so retrieval can rank them below official sources
        "personal": "no", "admin": "no", "duplicate": "no", "code": "no",
        "language": "no",      # D10: language course excluded
        "other": "review"}
DEDUP_ORDER = [".md", ".pdf", ".docx", ".doc", ".pptx", ".tex", ".txt", ".ipynb"]  # cleanest text first


def pdf_info(path: Path) -> tuple[int | None, bool | None]:
    """Page count + 'scanned' flag (almost no extractable text in the first 8 pages)."""
    try:
        with pymupdf.open(path) as doc:
            n = min(len(doc), 8)
            chars = sum(len(doc[i].get_text().strip()) for i in range(n))
            return len(doc), chars / max(n, 1) < 80
    except Exception:
        return None, None


def folder_type(rel: Path) -> tuple[str, str] | None:
    folders = rel.parts[2:-1]  # skip "Semester X/<course>"
    personal = FOLDER_RULES[0][1]
    for folder in folders:     # a personal/code folder anywhere in the path wins
        if personal.search(folder):
            return "personal", f"folder '{folder}'"
    for folder in reversed(folders):
        for doc_type, pattern in FOLDER_RULES[1:]:
            if pattern.search(folder):
                return doc_type, f"folder '{folder}'"
    return None


def name_type(name: str) -> tuple[str, str] | None:
    return next(((t, f"name ~ {p.pattern.split('|')[0]}") for t, p in NAME_RULES if p.search(name)), None)


def classify(rel: Path, course: str, pages: int | None) -> tuple[str, str]:
    """Return (doc_type, reason). The order of the checks is the precedence."""
    name, ext = rel.name, rel.suffix.lower()
    if OWNER.search(str(rel)):
        return "personal", "own name in path"
    if OTHER_STUDENT.search(str(rel)):
        return "personal", "another student's submission"
    if ADMIN.search(name):
        return "admin", "admin/organisational doc"
    if ext in {".txt", ".ipynb"} or name == "header.tex":
        return "code", "code/data file (D3: code excluded)"
    if course in EXCLUDED_COURSES:
        return "language", EXCLUDED_COURSES[course]
    if (pages or 0) >= TEXTBOOK_MIN_PAGES or TEXTBOOK_NAME.search(name):
        return "textbook", f"{pages} pages" if pages else "name"

    ft, nt = folder_type(rel), name_type(name)
    if ft and ft[0] == "personal":
        return ft
    if ft and ft[0] == "project":
        if ext != ".pdf" or PERSONAL_NAME.search(name) or R(r"summary").search(name):
            return "personal", "own file in project folder"
        return "project", "official material in project/case folder"
    if nt and nt[0] == "solution":
        return nt
    if ft and ft[0] == "homework":
        if R(r"lecture").search(name) or (pages or 0) >= 30:
            return "slides", "lecture slides inside HW folder"
        if ext in {".txt", ".ipynb", ".md"} or WORKING_FILE.search(name) or R(r"report|submission").search(name):
            return "personal", "homework working file / own report"
        if R(r"^l\d").search(name):
            return "solution", "official solution (files named l<N>)"
        return "problem_set", "official assignment sheet"
    if ft:
        return ft
    if PERSONAL_NAME.search(name):
        return "personal", "name looks like own work"
    if nt:
        return nt
    if ext in {".txt", ".ipynb"}:
        return "personal", "loose .txt/.ipynb (code/data)"
    if ext == ".pdf" or len(rel.parts) <= 3:
        return "slides", "unclassified course material (default)"
    return "other", "no rule matched"


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Same folder + same file stem in several formats -> keep the cleanest, non-scanned one."""
    df["stem_key"] = df.path.map(lambda p: str(Path(p).with_suffix("")).lower())
    df["rank"] = df.ext.map(lambda e: DEDUP_ORDER.index(e) if e in DEDUP_ORDER else 99)
    df.loc[df.scanned == True, "rank"] += 50  # noqa: E712  (never prefer a scanned copy)
    for _, grp in df[df.keep != "no"].groupby("stem_key"):
        if len(grp) > 1:
            best = grp.sort_values("rank").index[0]
            for i in grp.index.drop(best):
                df.loc[i, ["doc_type", "keep", "reason"]] = ["duplicate", "no", f"dup of {df.path[best]}"]
    # Explicit content duplicates with different names
    for part, why in DROP_FILES.items():
        df.loc[df.path.str.contains(part, regex=False), ["doc_type", "keep", "reason"]] = ["duplicate", "no", why]
    return df.drop(columns=["stem_key", "rank"])


def discover(corpus: Path = CORPUS_DIR) -> pd.DataFrame:
    rows = []
    for root, dirs, files in os.walk(corpus):
        dirs[:] = [d for d in dirs if d not in NOISE_DIRS and not d.startswith(".")]
        for f in files:
            path = Path(root) / f
            if path.suffix.lower() not in CONTENT_EXT or f.startswith("~$"):
                continue
            rel = path.relative_to(corpus)
            parts = rel.parts
            in_sem = parts[0].startswith("Semester")
            semester = parts[0].replace("Semester ", "S") if in_sem else "-"
            course = parts[1] if in_sem and len(parts) > 2 else parts[0]
            course = COURSE_OVERRIDES.get(f, course)       # misfiled files (P-30)
            pages, scanned = pdf_info(path) if path.suffix.lower() == ".pdf" else (None, None)
            doc_type, reason = classify(rel, course, pages)
            rows.append(dict(path=str(rel), semester=semester, course=course, ext=path.suffix.lower(),
                             pages=pages, scanned=scanned, doc_type=doc_type, keep=KEEP[doc_type],
                             reason=reason))
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    return mark_duplicates(df)


if __name__ == "__main__":
    df = discover()
    out = DATA_DIR / "manifest.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} files)\n")
    print(df.pivot_table(index="doc_type", columns="keep", values="path", aggfunc="count", fill_value=0))
    print("\nPages kept:", int(df.loc[df.keep == "yes", "pages"].sum()))
