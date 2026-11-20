"""Phase 1, step 2: extract clean text from every kept file -> data/pages.parquet.

One row ("unit") per PDF page, or per heading section for .md/.docx. Cleaning steps:
  1. OCR for scanned PDFs (D11) + line-level quality gate (drops handwriting/image noise)
  2. Remove repeated headers/footers (lines on >=40% of a document's pages)
  3. De-hyphenate words split across lines, fix ligatures, normalize whitespace
  4. Slides: drop "incremental build" pages (page fully contained in the next page)
  5. Textbooks: drop table-of-contents / index pages (mostly page numbers -> would pollute BM25)
  6. Drop near-empty pages (< MIN_CHARS characters)

Run:  .venv/bin/python -m ragtraining.ingest.extract
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

import docx
import pandas as pd
import pymupdf

from ragtraining.config import CORPUS_DIR, DATA_DIR

TESSDATA = "/opt/homebrew/share/tessdata"
MIN_CHARS = 40
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "­": ""}
REAL_WORD = re.compile(r"^[A-Za-zÀ-ÿ'’-]*[aeiouyAEIOUY][A-Za-zÀ-ÿ'’-]*$")


# ---------- text cleaning helpers ----------
def clean(text: str) -> str:
    for a, b in LIGATURES.items():
        text = text.replace(a, b)
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)          # de-hyphenate
    text = re.sub(r"[ \t ]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def ocr_quality_gate(text: str) -> str:
    """Keep OCR lines that look like language: >=2 tokens and >=60% real words."""
    kept = []
    for line in text.splitlines():
        toks = [t.strip(".,;:()[]?!\"'") for t in line.split()]
        if len(toks) >= 2 and sum(bool(REAL_WORD.match(t)) for t in toks) / len(toks) >= 0.6:
            kept.append(line)
    out = "\n".join(kept)
    return out if sum(bool(REAL_WORD.match(t)) for t in out.split()) >= 15 else ""


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def strip_boilerplate(pages: list[str]) -> list[str]:
    """Remove headers/footers: short lines repeated on >=40% of pages.

    - identical lines (e.g. "Faculty of X, University Y") -> removed
    - lines that differ only by a number are removed ONLY if that number moves with the page
      index (a page counter like "Page 7 of 36"). "Problem 1", "Problem 2"... are kept (P-21).
    """
    if len(pages) < 4:
        return pages
    threshold = max(3, 0.4 * len(pages))
    exact = collections.Counter()
    numbered = collections.defaultdict(list)          # pattern -> [(page_idx, first number)]
    for idx, p in enumerate(pages):
        lines = {l.strip() for l in p.splitlines() if l.strip()}
        exact.update(l.lower() for l in lines)
        for l in lines:
            m = re.search(r"\d+", l)
            if m and len(l) <= 80:
                numbered[_norm(l)].append((idx, int(m.group())))
    boiler_exact = {l for l, c in exact.items() if c >= threshold and len(l) <= 80}
    counters = {}                                        # pattern -> offset (number = page_idx + offset)
    for pat, occ in numbered.items():
        offset, _ = collections.Counter(n - i for i, n in occ).most_common(1)[0]
        pages_matching = {i for i, n in occ if n - i == offset}
        if len(pages_matching) >= threshold:            # other stray numbers ("3", eq. labels) ignored
            counters[pat] = offset

    def is_boiler(line: str, idx: int, edge: bool) -> bool:
        s = line.strip()
        if s.lower() in boiler_exact:
            return True
        m = re.search(r"\d+", s)
        short = len(re.sub(r"\d+", "", s).strip()) <= 20       # "Page # of #", "# / #", "Slide #"
        return bool(m) and edge and short and counters.get(_norm(s)) == int(m.group()) - idx

    out = []
    for idx, p in enumerate(pages):
        lines = p.splitlines()
        nonempty = [i for i, l in enumerate(lines) if l.strip()]
        edges = set(nonempty[:3] + nonempty[-3:])              # counters sit at the top/bottom
        out.append("\n".join(l for i, l in enumerate(lines) if not is_boiler(l, idx, i in edges)))
    return out


def is_toc_or_index(text: str) -> bool:
    toks = text.split()
    numeric = sum(bool(re.fullmatch(r"[\divxlc]*\d[\d–\-,.;:]*|[ivxlc]+[,.]?", t)) for t in toks)
    return len(toks) > 30 and numeric / len(toks) > 0.2   # page refs like "291–294," or "xiv"


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------- per-format readers ----------
def read_pdf(path: Path, scanned: bool) -> list[dict]:
    with pymupdf.open(path) as doc:
        raw = []
        for page in doc:
            if scanned:
                tp = page.get_textpage_ocr(language="eng", dpi=300, full=True, tessdata=TESSDATA)
                raw.append(ocr_quality_gate(page.get_text(textpage=tp)))
            else:
                raw.append(page.get_text())
    texts = [clean(t) for t in strip_boilerplate(raw)]
    return [dict(page=i + 1, section=None, text=t, ocr=scanned) for i, t in enumerate(texts)]


def _split_sections(lines: list[tuple[str | None, str]]) -> list[dict]:
    """lines = [(heading_or_None, text)] -> one unit per heading section."""
    units, head, buf = [], None, []
    for h, t in lines:
        if h is not None:
            if buf:
                units.append(dict(page=None, section=head, text=clean("\n".join(buf)), ocr=False))
            head, buf = h, [t]
        else:
            buf.append(t)
    if buf:
        units.append(dict(page=None, section=head, text=clean("\n".join(buf)), ocr=False))
    return units


def read_md(path: Path) -> list[dict]:
    lines = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        lines.append((m.group(2).strip(), line) if m else (None, line))
    return _split_sections(lines)


def read_docx(path: Path) -> list[dict]:
    d = docx.Document(path)
    lines = [((p.text.strip() if p.style.name.startswith("Heading") else None), p.text) for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            lines.append((None, " | ".join(c.text.strip() for c in row.cells)))
    return _split_sections(lines)


# ---------- main ----------
def extract(manifest: pd.DataFrame) -> tuple[pd.DataFrame, collections.Counter]:
    rows, dropped = [], collections.Counter()
    for _, f in manifest[manifest.keep == "yes"].iterrows():
        path = CORPUS_DIR / f.path
        if f.ext == ".pdf":
            units = read_pdf(path, bool(f.scanned))
        elif f.ext == ".md":
            units = read_md(path)
        elif f.ext == ".docx":
            units = read_docx(path)
        else:
            continue

        # slides: drop incremental builds (this page's text fully contained in the next page)
        if f.doc_type in {"slides", "project"}:
            flat = [_flat(u["text"]) for u in units]
            for i in range(len(units) - 1):
                if flat[i] and len(flat[i]) < len(flat[i + 1]) and flat[i] in flat[i + 1]:
                    units[i]["text"] = ""
                    dropped["incremental slide build"] += 1

        n_pages = len(units)
        for u in units:
            if not u["text"]:
                if u["ocr"]:
                    dropped["OCR page failed quality gate"] += 1
                continue
            # TOC lives in the first 6% of a book, index + bibliography in the last 15%.
            # Mid-book tables/code/truth tables look "numeric" too and must be kept (P-22).
            front = u["page"] is not None and u["page"] <= 0.06 * n_pages
            back = u["page"] is not None and u["page"] >= 0.85 * n_pages
            if f.doc_type == "textbook" and (front or back) and is_toc_or_index(u["text"]):
                dropped["textbook TOC/index page"] += 1
                continue
            if f.doc_type == "textbook" and back and len(re.findall(r"\b(?:19|20)\d{2}\b", u["text"])) >= 15:
                dropped["textbook bibliography page"] += 1
                continue
            if len(u["text"]) < MIN_CHARS:
                dropped["near-empty page"] += 1
                continue
            rows.append(dict(path=f.path, course=f.course, semester=f.semester, doc_type=f.doc_type,
                             **u, n_chars=len(u["text"])))
    df = pd.DataFrame(rows)
    df["n_tokens_est"] = (df.n_chars / 4).round().astype(int)  # ~4 chars/token; replaced by real tokenizer later
    return df, dropped


if __name__ == "__main__":
    manifest = pd.read_csv(DATA_DIR / "manifest.csv")
    df, dropped = extract(manifest)
    out = DATA_DIR / "pages.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out}: {len(df)} units from {df.path.nunique()} files\n")
    print("Dropped:", dict(dropped))
    print("\nUnits / est. tokens by doc_type:")
    print(df.groupby("doc_type").agg(units=("text", "count"), median_tok=("n_tokens_est", "median"),
                                     total_tok=("n_tokens_est", "sum")).to_string())
    print("\nOCR units kept:", int(df.ocr.sum()))
