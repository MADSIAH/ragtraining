"""Phase 1, step 3: turn extracted units (pages/sections) into retrieval chunks.

Four strategies, so we can compare them (LOGBOOK P4):
  page      1 page/section = 1 chunk (long pages split recursively)
  fixed     sliding window of TARGET tokens with OVERLAP, ignoring structure
  recursive split on paragraph -> line -> sentence -> word, pack up to TARGET tokens
  hybrid    page-level for slides/exams/problem sets/solutions (tiny slides merged),
            recursive with overlap for textbooks/summaries

Every chunk keeps: chunk_id, path, course, semester, doc_type, page_start, page_end, section, text, n_tokens.
"""
from __future__ import annotations

import re

import pandas as pd

TARGET, OVERLAP, MAX_TOK, MIN_TOK = 400, 60, 512, 100
PAGE_TYPES = {"slides", "exam", "problem_set", "solution", "project"}
SEPARATORS = ["\n\n", "\n", ". ", " "]


def ntok(text: str) -> int:
    return max(1, round(len(text) / 4))   # estimate; swapped for the embedding tokenizer in Phase 2


def split_recursive(text: str, target: int = TARGET, seps: list[str] = SEPARATORS) -> list[str]:
    """Split on the coarsest separator that works, then greedily pack pieces up to `target` tokens."""
    if ntok(text) <= target:
        return [text]
    sep, rest = seps[0], seps[1:]
    pieces = [p for p in text.split(sep) if p.strip()]
    if len(pieces) == 1 and rest:
        return split_recursive(text, target, rest)
    out, buf = [], ""
    for p in pieces:
        if ntok(p) > target and rest:              # piece itself too big -> go finer
            if buf:
                out.append(buf); buf = ""
            out.extend(split_recursive(p, target, rest))
        elif ntok(buf + sep + p) <= target:
            buf = f"{buf}{sep}{p}" if buf else p
        else:
            out.append(buf); buf = p
    if buf:
        out.append(buf)
    return out


def with_overlap(chunks: list[str], overlap: int = OVERLAP) -> list[str]:
    """Prefix each chunk with the last ~`overlap` tokens of the previous one."""
    out = chunks[:1]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap * 4:]
        tail = tail[tail.find(" ") + 1:] if " " in tail else tail
        out.append(f"{tail} {cur}")
    return out


def _doc_stream(g: pd.DataFrame) -> tuple[str, list[tuple[int, int | None]]]:
    """Concatenate a document's units; remember the char offset where each page starts."""
    text, marks = "", []
    for u in g.itertuples():
        marks.append((len(text), u.page))
        text += u.text + "\n\n"
    return text, marks


def _page_at(marks: list[tuple[int, int | None]], pos: int) -> int | None:
    page = marks[0][1]
    for off, p in marks:
        if off > pos:
            break
        page = p
    return page


def _row(meta, text, p0, p1, section=None) -> dict:
    return dict(path=meta.path, course=meta.course, semester=meta.semester, doc_type=meta.doc_type,
                page_start=p0, page_end=p1, section=section, text=text.strip(), n_tokens=ntok(text))


def chunk_page(g: pd.DataFrame, merge_small: bool = False) -> list[dict]:
    # p0 / s0 = page and section where the current merged chunk STARTS. The flushed chunk must be labelled
    # with s0, not with u.section (= the unit that didn't fit, i.e. the next chunk's start; P-50).
    rows, buf, p0, s0 = [], None, None, None
    for u in g.itertuples():
        parts = split_recursive(u.text, MAX_TOK) if ntok(u.text) > MAX_TOK else [u.text]
        for part in parts:
            if not merge_small:
                rows.append(_row(u, part, u.page, u.page, u.section)); continue
            # hybrid: merge consecutive tiny slides until >= MIN_TOK (never above MAX_TOK)
            if buf is None:
                buf, p0, s0 = part, u.page, u.section
            elif ntok(buf) < MIN_TOK and ntok(buf + part) <= MAX_TOK:
                buf += "\n\n" + part
            else:
                rows.append(_row(u, buf, p0, prev_page, s0)); buf, p0, s0 = part, u.page, u.section
            prev_page = u.page
    if merge_small and buf is not None:
        rows.append(_row(g.iloc[-1], buf, p0, g.iloc[-1].page, s0))
    return rows


def chunk_stream(g: pd.DataFrame, mode: str) -> list[dict]:
    text, marks = _doc_stream(g)
    meta = g.iloc[0]
    if mode == "fixed":   # track pages by word position (whitespace is rewritten, so text search fails)
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        words = [text[a:b] for a, b in spans]
        step, size = (TARGET - OVERLAP) * 3 // 4, TARGET * 3 // 4                  # ~0.75 words/token
        rows = []
        for i in range(0, max(len(words) - OVERLAP * 3 // 4, 1), step):
            j = min(i + size, len(words))
            rows.append(_row(meta, " ".join(words[i:j]),
                             _page_at(marks, spans[i][0]), _page_at(marks, spans[j - 1][0])))
        return rows
    else:  # recursive (+ overlap in hybrid)
        pieces = split_recursive(text, TARGET)
        if mode == "recursive_overlap":
            pieces = with_overlap(pieces)
    rows, cursor = [], 0
    for p in pieces:
        anchor = p[-200:].strip()[:80] or p[:80]
        end = text.find(anchor, cursor)
        end = end if end >= 0 else cursor
        start = max(0, end - len(p))
        rows.append(_row(meta, p, _page_at(marks, start), _page_at(marks, end)))
        cursor = max(cursor, start + 1)
    return rows


def chunk_corpus(pages: pd.DataFrame, strategy: str) -> pd.DataFrame:
    rows = []
    for _, g in pages.groupby("path", sort=False):
        g = g.sort_values("page", na_position="last")
        dt = g.iloc[0].doc_type
        if strategy == "page":
            rows += chunk_page(g)
        elif strategy in {"fixed", "recursive"}:
            rows += chunk_stream(g, strategy)
        elif strategy == "hybrid":
            rows += chunk_page(g, merge_small=True) if dt in PAGE_TYPES else chunk_stream(g, "recursive_overlap")
        else:
            raise ValueError(strategy)
    df = pd.DataFrame(rows)
    df = df[df.text.str.len() > 0].reset_index(drop=True)
    df.insert(0, "chunk_id", [f"{strategy[:3]}-{i:06d}" for i in range(len(df))])
    return df


def stats(df: pd.DataFrame) -> dict:
    t = df.n_tokens
    return dict(chunks=len(df), median_tok=int(t.median()), p90_tok=int(t.quantile(.9)),
                pct_tiny_lt50=round(100 * (t < 50).mean(), 1),
                pct_multi_page=round(100 * (df.page_start.notna() & (df.page_end != df.page_start)).mean(), 1),
                total_tok=int(t.sum()))


if __name__ == "__main__":
    # Rebuild all 4 chunk sets from data/pages.parquet; chunks.parquet = hybrid (D12).
    # WARNING: changes chunk rows -> the embeddings (data/emb_*.npy) must be rebuilt afterwards (index/build.py).
    from ragtraining.config import DATA_DIR
    pages = pd.read_parquet(DATA_DIR / "pages.parquet")
    for s in ["page", "fixed", "recursive", "hybrid"]:
        df = chunk_corpus(pages, s)
        df.to_parquet(DATA_DIR / f"chunks_{s}.parquet", index=False)
        print(s, stats(df))
        if s == "hybrid":
            df.to_parquet(DATA_DIR / "chunks.parquet", index=False)
