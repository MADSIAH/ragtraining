# Summary: what we've done so far

Short and plain. Details: `LOGBOOK.md` (plan, decisions, checklist) · `PROBLEMS_LOG.md` (54 problems & fixes) · `DECISIONS.md` (one-page decision record) · `GUIDELINES.md` (tutor rules, final).
Notebooks `01`–`07` show each phase with charts (inspection only, $0).

## Phase 0 · Setup ✅
API key in `.env` (hidden, git-ignored), verified. Python env `.venv`. Local git. 8 framing decisions (D1–D8).

## Phase 1 · Ingestion ✅
497 files classified → **328 kept** (no personal homework, classmates' work, code, duplicates, German).
Clean text + OCR → **8,544 pages**. Hybrid chunking (page-level for slides/exams, ~400-token overlapping chunks for books) → **10,943 chunks**.
Textbooks = 75% of all tokens, so sources need balancing.

## Phase 2 · Indexing ✅
- Compared 5 local embedding models → **Qwen3-Embedding 0.6B** ranks best.
- **BM25** keyword index with stemming.
- Discovery: Ollama normalizes vectors by default, and some models normalize internally. Raw vectors are only possible with nomic/bge-m3.

## Phase 3 · Retrieval ✅
Cosine, Euclid, cos+Euclid, BM25, hybrid fusion (RRF/weighted), filters, source caps, MMR, 2 rerankers (local cross-encoder + Claude).

## Phase 4 · Evaluation (60 synthetic questions written by Claude Opus for $0.64)
- **Hybrid (dense + BM25) is best: Hit@5 0.92**, vs dense alone 0.90 and BM25 alone 0.88.
- **Your Euclid + cosine question:** Euclid changes ~17% of the top-10 but quality is *identical* to cosine. Combining adds nothing. The embedding model matters far more (Qwen3 vs nomic: +15 points).
- **Rerankers are the biggest win:** local cross-encoder Hit@5 0.95; **Claude (Haiku) reranker 1.00, MRR 0.91** → default pipeline (D19), ~$0.0016/question.
- HNSW (approximate search) finds 99% of the true top-10 **1–8× faster** (the earlier "25–70×" was an unfair timing, P-49), but exact search already takes 0.3 ms here, so we keep it.

## Phase 5 · Tutor + guardrails ✅
Answers only from your material, with [S#] citations to course · file · page. Summaries are marked "(summary)". **Answer style: concise bullets, key words in bold** (G8, G11). **Guidelines confirmed final** (D21).
- **Test suite, 26 cases → 23 pass.** Off-topic 6/6, not-in-materials 6/6, injection in the question 5/5, injection hidden in a document 3/3, controls 2/2.
- **Weak spot: ambiguous one-liners (1/4).** "What is a key?" only gets the crypto meaning, because all 8 retrieved chunks are crypto. That's a retrieval problem, and the case for multi-query.
- **Refusal threshold measured, not guessed:** Claude reranker < 3/10 → "not covered" at $0. Raising it to 6 looked perfect on synthetic questions but wrongly refused "What is a key?" → back to 3.

## Phase 6 · RAGAS ✅
- **Judge fixed (P-33):** it now sees the same sources (with page numbers) as the tutor → **faithfulness 0.89 → 0.95–0.96**, re-scoring the old answers for $0.06.
- **Context recall: 0.93 → 0.99–1.0 with hybrid + rerank**, 0 citation errors.
- **Passing grade set (D25, your call):** faithfulness ≥ 0.92, context recall ≥ 0.90, context precision ≥ 0.80, answer relevancy ≥ 0.65 (warning only), 0 errors, ≤ 3% answers without proper citations. Every bar is a bit below today's score, so a real drop fails but the judge's own wobble (±0.03) doesn't. One command checks it: `run_ragas gate local_all`.

## Phase 7 · Agentic RAG ✅ (first version)
Plan → search → grade evidence → rewrite searches → answer. It turned a vague "growing/shrinking phase" question into "two-phase locking" and widened a CAPM-vs-Sharpe search across courses.
- **Multi-query and HyDE re-measured on all 70 questions (S4):** without a reranker, multi-query lifts your short questions a lot (MRR 0.79 → 0.96). With the reranker, every variant scores the same (0.857 vs 0.858). **Your call (D26): the default stays as is; multi-query, HyDE and the agent stay available as opt-in tools.**

## Phase 8 · App ✅ (first version)
`.venv/bin/streamlit run app.py --server.address localhost`: chat, tutor/agent mode, sources, retrieved chunks, agent trace.

## Free local mode (D22) ✅ verified
- **The tutor now runs on your Mac by default:** `qwen3:14b` via Ollama + the free local reranker. **$0, no API key.** Claude (Opus) is still one click away in the app sidebar.
- **Tested end to end:** Feistel and CAPM-vs-Sharpe answered correctly with citations (~20–26 s each), and the pizza question was refused in 1.5 s.
- **Bug found and fixed (P-43):** with temperature 0 the local model repeated itself for 3 minutes and crashed. Qwen's recommended settings plus a fixed seed fixed it.
- **Trade-off:** answers are a bit less tidy than Opus (repeated bullets, sometimes no "Review:" line, P-45). Real quality gets measured with `run_ragas local`.

## Your test questions ✅ (the reality check)
- **Your 15 questions** are in the test set: 12 answerable, 3 "not covered". Wording kept exactly as you typed it. Sources and key points come from your material.
- **The synthetic questions flattered keyword search:** BM25 drops from 0.88 to **0.75** on your questions, because you don't reuse the slide wording. The rerankers do best on your questions too (MRR **0.94** local, **0.96** Claude).
- **The refusal thresholds held up:** none of your real questions got refused, and the local engine blocks all 3 "not covered" ones for free.
- **Two things found:** section labels of `.md` notes were off by one (fixed, P-50). Your Python course's lectures are `.py` files that we excluded, so "how does a python loop work?" is currently "not covered" (P-51, your call).

## Final RAGAS ✅ (session 3)
- **The free local engine is about as good as Opus was:** faithfulness **0.953** on all 70 questions (Opus/Claude engine **0.987** on your 12). Context recall 0.97.
- **The real gap is retrieval:** with the Claude reranker, 95% of the 8 sources are relevant (context precision); with the local reranker, 75% (on your questions).
- **Local weak spot fixed (P-53):** 8 of 70 answers forgot their [S#] citations. In 6 of them qwen only listed sources in the final "Review:" line. Now the tutor **asks once more with a reminder** when an answer has no citations → **8/8 cited**, quality unchanged, $0. The local engine **passes the gate** (it failed before, on citations only).
- Cost: local run $0.065 (judge only), Claude run $0.45.

## Session 4 ✅ (wrap-up)
- **Chunking re-checked on all 70 questions:** the hybrid chunking we chose is best or tied once the reranker runs (Hit@5 0.97). D12 stands.
- **Results deck for recruiters (D27):** 8 slides, private link in LOGBOOK §0.1 (downloads as PowerPoint/PDF). Put your name on the cover and share it before sending.
- **`DECISIONS.md`:** one page, 6 decisions with our numbers + 3 problem stories, honest that Claude Code was the pair programmer.

## Ready for GitHub ✅ (session 4)
- **Privacy check (your idea):** asked for contact details, the tutor gave out an instructor's email (59 chunks contain addresses). Now every email and phone number is masked **before the model sees it** and the tutor never writes a word with `@`, `.com` or `.ch` (it skips it) → **0 of 14** tricky questions leaked, and "How does PGP email encryption work?" still gets answered. Names still show (masking them would break "Sharpe ratio" & co.).
- **Public copy built** in `../ragtraining` by `scripts/export_public.py`: no test set, no notebook outputs, no course file names, no emails, no classmates' names. An automatic leak check compares every file against the whole corpus and **passes**. Fresh git history, so the old commits (with the test set and your email) stay private.

## Still to do (all optional)
1. Review the deck and `DECISIONS.md`.
2. A fresh full gate run (~25 min, ≈ $0.065) to confirm the citation fix on all 70 questions.
3. NeMo comparison, agent tuning, a mode for ambiguous one-liners.

**Total spent on Claude API calls for the whole project: ≈ $4.37.**
