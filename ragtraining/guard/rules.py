"""Phase 5: custom guardrails (GUIDELINES.md). Cheap, deterministic checks in code.

  check_input        G4/G5  user input: injection patterns, empty/too long
  sanitize_chunk     G5     remove instruction-like lines from retrieved text before Claude sees it
  confident_enough   G3/G4  refuse early (no LLM call) if even the best chunk is a poor match
  validate_citations G2     every [S#] must exist; an answerable reply must cite something
  mask_contacts      G12    emails / phone numbers -> [contact removed], in sources AND answers (D28)
  drop_banned_words  G12    the tutor never writes a word with "@", ".com" or ".ch" (answers + chunk view)
"""
from __future__ import annotations

import re

INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)|disregard (the )?(system|previous)|"
    r"you are now|new instructions:|system prompt|reveal (your|the) (prompt|instructions)|"
    r"act as (an? )?(unrestricted|jailbroken)|\bDAN\b|developer mode", re.I)
MAX_QUESTION_CHARS = 2000
MIN_RERANK_SCORE = 0.05     # local cross-encoder score (0..1). Below -> "not in your materials" (tuned in eval)


def check_input(question: str) -> tuple[bool, str]:
    q = question.strip()
    if not q:
        return False, "Empty question."
    if len(q) > MAX_QUESTION_CHARS:
        return False, f"Question too long (> {MAX_QUESTION_CHARS} characters)."
    if INJECTION.search(q):
        return False, "This looks like an attempt to change my instructions. I can only help with your course material."
    return True, ""


def sanitize_chunk(text: str) -> tuple[str, int]:
    """Drop lines that look like instructions aimed at an AI. Returns (clean_text, n_removed)."""
    kept, removed = [], 0
    for line in text.splitlines():
        if INJECTION.search(line):
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept), removed


# G12 (D28, P-54): personal contact details never reach the model or the screen. Masked at query time, so the
# index needs no re-embedding. Only emails and phone numbers WITH a cue are masked, never bare numbers (finance
# tables, formulas). RFC 2606 documentation domains stay: they're textbook examples, not people.
CONTACT_REMOVED = "[contact removed]"
EMAIL = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
DOC_DOMAINS = re.compile(r"^(?:[\w-]+\.)*example\.(?:com|org|net)$|\.(?:example|invalid|test)$", re.I)
PHONE = re.compile(r"(?i)(?:\b(?:tel|phone|mobile|cell|fax)\b\.?\s*(?:no\.?|number)?\s*[:.]?\s*)\+?\d[\d ()./-]{5,}\d"
                   r"|\+\d{1,3}[ .]?\(?\d{1,4}\)?(?:[ .-]?\d{2,4}){2,4}")


def mask_contacts(text: str) -> tuple[str, int]:
    """Replace email addresses and phone numbers with [contact removed]. Returns (text, n_masked)."""
    n = 0

    def email(m):
        nonlocal n
        if DOC_DOMAINS.search(m.group(1)):
            return m.group(0)
        n += 1
        return CONTACT_REMOVED

    def phone(m):
        nonlocal n
        n += 1
        return CONTACT_REMOVED

    return PHONE.sub(phone, EMAIL.sub(email, text)), n


# G12, stricter (Max, S4): the tutor never WRITES an "@" or a ".com"/".ch" domain, not even a textbook example.
# Any whitespace-delimited word containing one is skipped entirely, so no address or domain can be spelled out.
BANNED_WORD = re.compile(r"\S*(?:@|\.(?:com|ch)\b)\S*", re.I)


def drop_banned_words(text: str) -> tuple[str, int]:
    """Remove every word containing '@', '.com' or '.ch'. Returns (text, n_dropped)."""
    out, n = BANNED_WORD.subn("", text)
    if n:
        out = re.sub(r"[ \t]{2,}", " ", out)                              # close the gaps, keep line breaks
        out = re.sub(r"[ \t]+([,.;:)])", r"\1", out)
        out = re.sub(r"[ \t]+$", "", out, flags=re.M)
    return out, n


def confident_enough(top_score: float, threshold: float = MIN_RERANK_SCORE) -> bool:
    return top_score >= threshold


def validate_citations(answer: str, n_sources: int, answerable: bool) -> list[str]:
    problems = []
    cited = {int(x) for x in re.findall(r"\[S(\d+)\]", answer)}
    bad = sorted(c for c in cited if not 1 <= c <= n_sources)
    if bad:
        problems.append(f"cites non-existent sources {bad}")
    if answerable and not cited:
        problems.append("answer has no citations")
    return problems
