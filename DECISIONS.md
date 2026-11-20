# RAG Training: decision record

An exam-prep tutor over my own university material (328 files, 17 courses, 10,943 chunks). It answers only from the material, cites course · file · page, and refuses when the material doesn't cover a question.
**How it was built:** with Claude Code as a pair programmer. I made the design decisions, designed the evaluation and judged the trade-offs; much of the code was written with Claude. Every number below comes from our own measurements. The full history (29 decisions, 54 problems) is in [LOGBOOK.md](LOGBOOK.md) and [PROBLEMS_LOG.md](PROBLEMS_LOG.md).

## Decisions

| Decision | Options considered | Evidence | Outcome |
|---|---|---|---|
| **Cosine vs. Euclidean** (D1) | cosine only · Euclidean only · both combined | For unit vectors ‖a−b‖² = 2 − 2·cos θ, so the rankings are identical. Tested on **raw, un-normalized** vectors (nomic, the model whose length varies most): Euclidean reordered ~17% of the top 10, MRR 0.59 = 0.59 = 0.59 (70 questions) | Combining adds nothing. Dense + BM25 hybrid is the retriever |
| **Embedding model** (D13) | 5 local models (nomic, mxbai, bge-m3, embeddinggemma, Qwen3) | Smoke test on paraphrased questions, not reputation: Qwen3 MRR **0.83** vs 0.74 for the runner-up. On 70 questions Qwen3 beats nomic by **+0.22 MRR** | Qwen3-Embedding 0.6B. The model mattered far more than the distance metric |
| **Test-set bias** (D6, P-34) | synthetic questions only · synthetic + my own | Synthetic questions reuse the pages' wording, which favours keyword search. On my 12 real questions BM25 Hit@5 fell **0.88 → 0.75**, and rerankers ranked best (MRR 0.94–0.96) | Mixed test set (58 synthetic + 12 mine). Every shipped pipeline ends with a reranker |
| **Quality vs. cost** (D19, D22) | Claude (Opus answers + Claude reranker) · local (qwen3:14b + local reranker) | On my questions: retrieval MRR 0.96 vs 0.94; faithfulness 0.99 vs 0.96. Claude ≈ $0.04 per question, local **$0** | Local engine is the default; Claude stays one click away in the app |
| **Thresholds from data** (D21, P-37) | guess a refusal threshold · calibrate on answerable vs. not-covered questions | A stricter threshold (6/10) looked perfect on synthetic questions, but the end-to-end suite showed it **wrongly refused** "What is a key?" (score 4) | Reverted to 3/10 (local 0.05): 0/70 false refusals. Test end to end, not just on the calibration set |
| **Scope** (D23) | widen the index to Python lecture code · keep it excluded | The Python lectures are `.py` files, excluded with all code (D3). "How does a python loop work?" scores 0.041, below the local engine's 0.05 gate | Kept excluded; the gap is **documented** as "not covered" instead of silently widening the scope |

**Quality gate (D25):** faithfulness ≥ 0.92 · context recall ≥ 0.90 · context precision ≥ 0.80 · answer relevancy ≥ 0.65 (soft) · ≤ 3% uncited answers. Local engine on all 70 questions: 0.953 / 0.970 / 0.873 / 0.775 / 0% → **PASS** (uncited answers were 8/70 until the tutor re-asks once, P-53).

**Privacy check before release (D28, P-54):** asked for contact details, the tutor handed out an instructor's email, because 59 chunks contain addresses. Emails and phone numbers are now masked in the sources *before the model sees them* and in the answer: on top, the tutor never writes a word containing `@`, `.com` or `.ch`. **0/14** adversarial probes leaked, and e-mail as a topic (PGP) is still answered.

## Three problem stories (verify results instead of trusting them)

1. **A 3-minute loop (P-43).** The first local answer repeated the same bullets for 181 s and crashed with a JSON error. Cause: greedy decoding (temperature 0), which suits large API models but makes small local models loop. Fix: the vendor's recommended sampling, a fixed seed, an output cap, and a retry when the cap is hit. → *Use the vendor's settings and fail loudly on a cut-off reply.*
2. **A speed-up that wasn't (P-49).** Approximate search (FAISS HNSW) looked 25–70× faster than exact search. I had timed a batch of queries against single queries. Timed the same way, it is 1–8× faster, and exact search already takes 0.3 ms, so exact stayed. → *Time both sides the same way.*
3. **A silent label bug (P-50).** Retrieval scores were fine, but 139 Markdown chunks carried the *next* section's heading, so citations pointed to the wrong section. Found while labelling my own test questions; fixed without re-embedding, plus a regression test. → *Metadata bugs are silent: spot-check them against the text.*

<sub>Not published: the test sets and `data/` contain excerpts of course material, and `.env` holds the API key.</sub>
