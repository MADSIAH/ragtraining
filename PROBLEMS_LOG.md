# Problems Log

Every problem we hit, why it happened, how we fixed it, and the general lesson. These are the problems you'll meet in **any** real RAG project.

Format: **Symptom → Cause → Fix → Lesson**. Newest at the bottom. Phase in brackets.

---

## Setup

**P-01 · `.env` file "missing" [Phase 0]**
- Symptom: Max couldn't see `.env` in Finder.
- Cause: macOS hides files whose name starts with a dot.
- Fix: press `Cmd + Shift + .` in Finder, or `open -e .env`.
- Lesson: secrets live in dotfiles *on purpose*: hidden, git-ignored, readable only by you (`chmod 600`).

**P-02 · API key valid but calls fail [Phase 0]**
- Symptom: `400 – Your credit balance is too low`.
- Cause: a key alone isn't enough. The API is prepaid and separate from a Claude.ai subscription.
- Fix: buy credits. The check script tests the key (free call) and billing (tiny paid call) **separately**, so you know which one failed.
- Lesson: always verify credentials with a cheap call *before* a long pipeline run.

**P-03 · `fitz` deprecation warning [Phase 1]**
- Cause: PyMuPDF renamed its import from `fitz` to `pymupdf`.
- Fix: use `import pymupdf`.
- Lesson: pin versions in `requirements.txt` once the project is stable.

## Corpus discovery (deciding *what* goes into the RAG)

**P-04 · 25,000 junk files in the corpus [Phase 1]**
- Cause: Python virtual envs and `node_modules` inside course project folders.
- Fix: skip noise directories during the walk (`NOISE_DIRS`).
- Lesson: never index "everything in a folder". Library source code would flood retrieval with irrelevant matches.

**P-05 · Personal work mixed with official material [Phase 1]**
- Symptom: `HW/` folders contain the official assignment sheet, the official solution, *and* your submission.
- Fix: layered rules with explicit precedence (own name → classmates → admin → textbook → folder → filename). Every file gets a `reason` column, so each decision is auditable.
- Lesson: data curation is a design decision. The RAG will confidently cite whatever you feed it, mistakes included.

**P-06 · Opaque file names (e.g. `l3-01.pdf`, `u3-01.pdf`) [Phase 1]**
- Fix: opened the files. `l` = *Lösung* (solution), `u` = *Übung* (assignment).
- Lesson: when a rule depends on a guess, **verify on the content**.

**P-07 · A classmate's homework in your folders [Phase 1]**
- Example: `HW4_Firstname_Lastname.pdf` (a classmate's name).
- Fix: pattern `HW<n>_Firstname_Lastname` → excluded.
- Lesson: privacy and correctness both need checking, not just relevance.

**P-08 · Regex false positive: "exam" matched "**exam**ple" [Phase 1]**
- Symptom: 7 data `.txt` files (in an `example …/` data folder) were tagged as exams.
- Fix: letter-boundary lookarounds `(?<![a-z])exams?(?![a-z])`.
- Lesson: after writing rules, **look at the output table**. Substring rules always over-match somewhere.

**P-09 · Lecture slides stored inside HW folders [Phase 1]**
- Fix: in an HW folder, a file with ≥30 pages or "lecture" in its name → slides.
- Lesson: folder structure is a hint, not the truth.

**P-10 · Same document in several formats [Phase 1]**
- Example: `cheat_sheet.md` + `.pdf` + `.docx`, and two Sipser copies (scanned + readable).
- Fix: dedup by folder + file stem, preference `md > pdf > docx`, never a scanned copy.
- Lesson: duplicates waste top-k slots. The same fact retrieved 3× pushes out other useful chunks.

## Text extraction

**P-11 · Scanned PDFs have no text [Phase 1]**
- Fix: OCR (Tesseract via PyMuPDF) + a line-level quality gate (≥60% real words).
- Result: 75 pages kept, 18 dropped (handwriting, photos).
- Lesson: OCR output **must** be filtered. Garbage text still gets embedded and can be retrieved.

**P-12 · Repeated headers/footers on every page [Phase 1]**
- Example: "Faculty of X, University Y", course name, page numbers.
- Fix: drop short lines that appear on ≥40% of a document's pages.
- Lesson: boilerplate makes every chunk look similar, which hurts both dense and BM25 ranking.

**P-13 · Incremental slide builds [Phase 1]**
- Cause: the same slide exported 3× with one more bullet each time.
- Fix: drop a page if its text is fully contained in the next page (33 dropped).

**P-14 · Textbook TOC / index pages [Phase 1]**
- Symptom: index pages are walls of keywords + page numbers. BM25 loves them, but they contain no explanations.
- First fix missed ranges like `291–294,`. Improved numeric-token regex → 135 pages dropped.
- Lesson: check a filter's *output*, not just that it runs.

**P-15 · Textbook bibliography pages [Phase 1]**
- Fix: a textbook page with ≥15 year-like tokens → dropped (75 pages).

**P-16 · Ligatures & hyphenation [Phase 1]**
- Cause: PDFs store "ﬁ" as one character and split "infor-mation" across lines, so search for "information" fails.
- Fix: ligature map + de-hyphenation of lowercase line breaks.

**P-17 · Textbooks dominate the corpus [Phase 1]**
- Numbers: 6 textbooks = 43% of pages but **75% of tokens**.
- Risk: retrieval is biased toward books over your course slides.
- Fix (planned, Phase 3): source balancing / MMR.

## Chunking

**P-18 · Fixed-window chunker produced 2,000-token chunks [Phase 1]**
- Cause: split on `" "` only, so whole lines joined by `\n` counted as one "word".
- Fix: split on any whitespace.
- Lesson: always print chunk-size stats (median, p90) after chunking. Bugs show up as absurd numbers.

**P-20 · Wrong page numbers on fixed-window chunks [Phase 1]**
- Symptom: every fixed chunk was labelled page 1, which would have produced wrong citations.
- Cause: the chunker rebuilt the text with single spaces, so searching for the chunk in the original text failed silently.
- Fix: track pages by **word position** instead of text search.
- Lesson: test metadata (page numbers, sources), not just text. Citations are only as good as this bookkeeping.

**P-21 · Header filter would delete numbered titles [Phase 1, caught by a test]**
- Symptom: lines like "Problem 1", "Problem 2"… normalize to the same pattern "problem #". On exams they repeat on ≥40% of pages, so they were treated as a footer and deleted.
- Fix: a numbered line is boilerplate only if its number **moves with the page index** (a page counter, e.g. "Page 7 of 36"). Identical lines are still removed.
- Lesson: write tests for cleaning rules with tricky cases. Over-aggressive cleaning silently deletes real content.

**P-22 · TOC/index filter deleted real content pages [Phase 1]**
- Symptom: 97 mid-book pages were dropped as "TOC/index", e.g. modular-arithmetic tables (crypto book), truth tables (discrete math), code listings with line numbers (testing book), fund data figures (Investments).
- Cause: these pages are number-heavy, just like an index. Fixing P-21 kept more page numbers, which made it worse.
- Fix: apply the TOC filter only to the first 6% of a book and the index/bibliography filters only to the last 15%. Page-counter detection now ignores stray numbers.
- Lesson: **inspect what a filter removes, not just how many.** A heuristic that looks right on average can delete exactly the technical content students need. Use position and structure, not just token statistics.

## Embeddings

**P-23 · Ollama silently normalizes embeddings [Phase 2]**
- Symptom: every vector from `ollama.embed` (`/api/embed`) has length exactly 1.000.
- Consequence: cosine and Euclidean would rank *identically* (‖a−b‖² = 2 − 2cos θ), so the D1 experiment would be meaningless.
- Fix: the legacy endpoint `ollama.embeddings` (`/api/embeddings`) returns **raw** vectors (nomic: length 19.0 ± 0.85). Raw vectors are used only for the Euclid experiment.
- Lesson: know what your embedding API does *after* the model. Many providers normalize by default.

**P-24 · mxbai-embed-large crashes on long text [Phase 2]**
- Symptom: `400 – the input length exceeds the context length`, even with `truncate=True`.
- Cause: the context is only 512 tokens. Ollama 0.17.7 truncates moderately long inputs but errors on very long or symbol-heavy ones.
- Fix: `_embed_one_safe` cuts 25% of the characters and retries. A batch with one bad text falls back to one-by-one.
- Lesson: an embedding model's context limit is a hard constraint on chunk size. A silently truncated chunk is embedded **without its ending**.

**P-25 · Same text, very different token counts per model [Phase 2]**
- Example: one 1,830-character chunk = 586 tokens (nomic/BERT tokenizer) but 1,020 tokens (bge-m3/XLM-R tokenizer). Symbols and math inflate counts.
- Lesson: "400 tokens" depends on the tokenizer. Our chars/4 estimate matches BERT-style tokenizers well (real/est ≈ 0.94).

**P-26 · Some models normalize *inside* the model, so raw vectors are impossible [Phase 2]**
- Symptom: even the raw endpoint returns length 1.000 for mxbai-embed-large, qwen3-embedding and embeddinggemma.
- Cause: their architecture ends with a normalization layer.
- Measured raw-norm variation (CV): nomic 4.1% (shorter texts get longer vectors, corr −0.52), bge-m3 1.5%, the others 0%.
- Consequence: the D1 Euclid-vs-cosine experiment is only meaningful with nomic (most variation) or bge-m3.
- Lesson: "Euclidean vs cosine" isn't only a math question. It depends on whether your model even outputs magnitude information.

**P-27 · Legacy raw endpoint has no truncation [Phase 2]**
- Symptom: `ollama.embeddings` → 500 "input length exceeds the context length" (mxbai, 1,500 chars).
- Fix: `_embed_one_safe(raw=True)` shortens and retries.

**P-28 · Parallel jobs overwriting a shared metadata file [Phase 2]**
- Symptom: 3 index builds (dense, raw, sparse) ran at the same time. Each read `index_meta.json` at start and wrote it at the end, so the last writer erased the others' entries.
- Cause: classic read-modify-write race condition.
- Fix: `write_meta()` rebuilds the metadata from the files actually on disk. It's idempotent and safe to run anytime.
- Lesson: when jobs run in parallel, never let them each own a copy of shared state. Derive it from the artifacts instead.

## Generation & agent

**P-29 · Reranking erased the "summaries rank lower" rule [Phase 5]**
- Symptom: the agent's evidence for a 2PL question was 3 personal summaries + a testing textbook, not the Data Management slides.
- Cause: doc_type weights (D10) were applied to first-stage scores. The reranker then *replaced* those scores, so the weight vanished.
- Fix: re-apply the weights after reranking, in `search()` and in the agent.
- Lesson: every stage that rewrites scores must re-apply business rules. Check the **final** ranking, not each stage alone.

**P-30 · A file saved in the wrong course folder [Phase 5]**
- Example: a Data Management summary saved in another course's folder, so it carried the wrong course label and would break course filters.
- Fix: `COURSE_OVERRIDES` in discovery, plus a metadata-only patch of the parquet files. Text is unchanged, so embeddings stay aligned row by row.
- Lesson: metadata comes from humans' folder habits. Audit it, especially when it drives filters.

**P-31 · Over-strict grader → agent always ran the maximum rounds [Phase 7]**
- Symptom: Haiku judged the evidence "insufficient" every round, so 3 rounds ran and answers took 35–38 s.
- Fix: grader prompt says "sufficient if a correct, useful answer to the MAIN question is possible; minor details missing is OK".
- Lesson: an agent's stopping criterion is a tunable parameter. Too strict wastes time and money, too loose answers on thin evidence.

**P-32 · "nan" shown as a page for Markdown sources [Phase 5]**
- Cause: missing `section` values are float NaN, which is *truthy* in Python, so `x or "-"` didn't trigger.
- Fix: explicit `isinstance(x, str)` check.

## Evaluation

**P-33 · RAGAS faithfulness penalizes the tutor's page pointers [Phase 6]**
- Symptom: 17–19 of 20 answers had "unsupported claims" like "Lecture 3 slides p.1-4 contain the criteria".
- Cause: the tutor ends with "review slides p.X" (rule G8). The page numbers come from source *metadata*, but the judge only receives chunk *text*, so it can't verify them.
- Fix (2026-11-19): the judge now gets the **same `<source>` blocks** as the tutor (`format_sources`: metadata + full sanitized text; the old 1,500-char truncation is gone too). Re-judged the stored answers: faithfulness **0.891 → 0.963** (prod) and **0.891 → 0.952** (baseline). Unsupported claims went 42 → 13 and 39 → 14. The ones left are real tutor inferences (e.g. "the empty-string check must come first because email[0] would fail").
- Cost trick: re-judging stored answers uses Haiku only (**$0.03 per 20 questions**), against $0.84 to regenerate them with Opus. Rows now store `chunk_ids`, so a re-judge rebuilds exactly the same evidence.
- Lesson: **an eval must see exactly what the model saw.** Otherwise you measure your harness, not your system. And **store what the eval saw**, so fixing the judge doesn't force paying for new answers.

**P-36 · LLM scores are noisy [Phase 5/6]**
- Symptom: the same contexts, judged again, moved context precision 0.876 → 0.851 and recall 0.992 → 1.0. The Claude reranker scored "Tell me a joke about cats." **7**, then **4**, then **6** across three runs.
- Cause: LLM judges and rerankers are not deterministic, and a 0–10 integer score jumps by whole points.
- Lesson: treat differences under ~0.03 (RAGAS means over 20 questions) or ~2 points (a single rerank score) as **noise**. A threshold needs a **margin**, not a value sitting right at the edge of the data.

**P-37 · A threshold calibrated on synthetic questions was too strict [Phase 5]**
- Symptom: synthetic answerable questions all scored ≥ 8/10 on the Claude reranker, and off-topic ones ≤ 7. A threshold of 6 looked perfect (0/60 false refusals). Then the end-to-end suite showed "What is a key?" scoring **4**, so the tutor refused it, although your material covers keys.
- Cause: synthetic questions are **specific** (one per passage), so they score high. Real, vague questions score lower and fall outside the calibration data.
- Fix: back to **3**. The gate only saves cost: anything that slips through still reaches Opus, which refuses off-topic questions itself (G3/G4, verified: "Should I quit my job?" scored 6, passed the gate, and Opus said "not covered"). A false refusal is the worse error for a tutor.
- Lesson: **calibrate on the questions real users ask.** Test a threshold end to end, not only on the score histogram.

**P-38 · Ambiguous one-line questions collapse onto one meaning [Phase 5]**
- Symptom: "What is a key?" → 8/8 chunks from the InfoSec crypto textbook, so **primary key** (Data Management) was never retrieved. "What is regression?" missed **regression testing** (SQT), and "What is a transaction?" missed the blockchain sense. Only "normalization" covered both of its senses.
- Cause: dense + rerank pull the strongest and most frequent sense, and the biggest source wins. The tutor can't mention a meaning it never sees.
- Also: my first check ("cites ≥ 2 courses") was too lenient. *Regression* cited 3 courses but only one meaning. It's now graded by **sense coverage** (marker words per meaning).
- Fix (next): multi-query (one search per phrasing or meaning, then fuse), or per-course diversity for very short questions. Measure it by sense coverage of the retrieved chunks.
- Lesson: an answer can only be as broad as its **retrieval**. Count *meanings*, not *courses*.

**P-41 · Multi-query with RRF fusion still returned only one meaning [Phase 7]**
- Symptom: for "What is a key?" Haiku correctly wrote a separate database query ("primary key candidate key relational database"), but the fused top-8 was still 8/8 crypto.
- Cause: **RRF rewards agreement.** The original question and the crypto query both rank crypto chunks high, so those appear in 2 of 4 lists and outscore database chunks that appear in only 1. That's right for dense + BM25 fusion (two views of the *same* need) and wrong for diversity (*different* needs).
- Fix: a **round-robin** fusion option (`fuse="rr"`): each query's next-best chunk in turn, so every phrasing or meaning gets slots.
- Lesson: choose the fusion rule by what you want. Consensus → RRF. Coverage → interleave.

**P-39 · Test labels need checking too [Phase 5]**
- "What is beta?" looked ambiguous, but all 458 corpus mentions are the finance sense, so answering from Investments alone is right. Reclassified as a control.
- Lesson: verify test-case labels against the corpus, just like gold labels (see 7.0, Q3/Q6).

**P-40 · Small notebook bugs [Phase 2/3 notebooks]**
- `plt.hist` failed with "Too many bins": all Qwen3 norms are 1.0, so the value range is ~0. That's actually proof the vectors are normalized. Fix: explicit `range=`.
- P-32 again: a NaN `section` is truthy, so `(x.section or '')[:15]` crashed. Same fix (`isinstance(x, str)`). Lesson: when a bug repeats, reuse the fixed helper instead of rewriting it.

**P-34 · Synthetic questions favour keyword search [Phase 4]**
- BM25 scored 0.88 Hit@5, close to dense, even though questions were told to paraphrase. Some wording leaks from the source passage.
- Mitigation: Max's 15 hand-written questions (D6) will be the reality check.

## Deployment

**P-35 · Streamlit exposed the app to the whole network [Phase 8]**
- Symptom: on launch Streamlit printed a *Network URL* and an *External URL* (public IP), so anyone on the network could use the tutor, read course material and spend the API credits.
- Fix: `--server.address localhost` (also set in `.streamlit/config.toml`).
- Lesson: an app that holds an API key and private data must bind to localhost unless you add authentication.

**P-19 · Token counts are estimates [Phase 1]**
- `n_tokens ≈ chars / 4` until the embedding model (and its tokenizer) is chosen in Phase 2. Real counts may differ by ±20%.

## Local model (D22)

**P-42 · Ollama's default context would silently cut off the sources [D22]**
- Symptom (prevented, found while building the backend): the tutor prompt is ~3,400–4,000 tokens (8 sources + rules), but Ollama's default context window is only 2–4K tokens.
- Cause: Ollama **truncates the prompt without any error** when it's longer than `num_ctx`. The model would answer from half the sources and nobody would notice.
- Fix: `num_ctx` 12,288 in `ragtraining/llm.py`, plus a check that raises an error if the prompt fills the whole window.
- Lesson: local runtimes have small, **silent** defaults. Always set the context size explicitly and check the token count you actually sent.

**P-43 · Local model looped until the token cap: 181 s, then a JSON error [D22]**
- Symptom: the first real local answer (Feistel) crashed with `JSONDecodeError: Unterminated string`. Raw output: the same bullets repeated over and over, `done_reason=length`, 4,096 tokens, 181 s.
- Cause: **greedy decoding** (`temperature 0`). Qwen3's own docs warn it causes endless repetition. With JSON-constrained output the string never closes, so the reply can't be parsed.
- Fix: Qwen's recommended non-thinking sampling (temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5) + **fixed seed 42** (reproducible: the same answer twice in a row). Output cap 4,096 → **1,500** (a concise answer is ~300 tokens, so the cap only catches loops), and `done_reason == "length"` now raises a clear error instead of a cryptic JSON one. Result: 20 s, 150 words, valid JSON.
- Lesson: "temperature 0 = most reliable" holds for big API models, **not for small local models**. Use the vendor's recommended sampling and pin a seed for reproducibility. Check `done_reason`, so a truncated reply fails loudly.

**P-45 · Local model follows the style rules less strictly [D22]**
- Symptom (smoke test, 3 questions): answers are correct and cited (0 citation errors, pizza refused at $0), but the Feistel answer has **10 bullets that repeat each other** and no "Review:" line, and the CAPM answer opens with an intro sentence. Both break G8.
- Cause: a 14B model follows long instruction lists less reliably than Opus. This was expected in D22 ("weaker discipline").
- Fix (next): measure it with `run_ragas local` vs `prod` on the final test set. If needed, add a post-check (e.g. drop duplicate bullets) or a shorter local-specific prompt.
- Lesson: changing the model means **re-testing the guidelines**, not only the accuracy. RAGAS doesn't score conciseness, so check the style separately.

## Tooling & workflow

**P-44 · `.venv/bin/pytest` couldn't find `ragtraining` [tests]**
- Symptom: `ModuleNotFoundError: No module named 'ragtraining'` in all 5 test files, "5 errors during collection".
- Cause: the package isn't installed. `python -m pytest` adds the current folder to the import path, but the `pytest` launcher doesn't. The LOGBOOK told us to use the launcher.
- Fix: `pytest.ini` with `pythonpath = .` → **31 passed**.
- Lesson: make the documented test command work from a clean shell. Don't rely on how it happened to be run before.

**P-46 · A file the LOGBOOK pointed to didn't exist [workflow]**
- Symptom: LOGBOOK §0.1 and §9 referenced `HANDOFF.md` ("ownership in HANDOFF.md §5"), but the file isn't in the repo.
- Cause: two Claude sessions ran in parallel. One wrote HANDOFF.md, the other "merged" it into the LOGBOOK, and the file itself was lost before any commit.
- Fix: LOGBOOK §0.1 is now the only handoff. Ownership was rebuilt from `git diff`.
- Lesson: with parallel sessions, **commit early** and keep one source of truth. Uncommitted files are the first thing to disappear.

**P-47 · A long test run's output was lost [workflow]**
- Symptom: the re-run smoke test (~1 min) came back as "Tool result missing due to internal error". The answers were gone and had to be re-generated.
- Fix: long runs now write to a file first (`> out.txt`), and the file gets read afterwards.
- Lesson: for anything slow or costly, **save results to disk before looking at them**. (The eval scripts already do this: `data/*.json`.)

**P-48 · Notebook kernel crashed at `import faiss` (segfault) [Phase 7 notebook]**
- Symptom: `DeadKernelError: Kernel died` while executing notebook 07. As a plain script: exit code 139 (segfault) exactly at the HNSW cell.
- Cause: earlier cells load **PyTorch** (local reranker). PyTorch and **FAISS each bundle their own OpenMP runtime** (the library for multi-core loops). Two copies in one process → `OMP Error #15` → crash. In session 1 the HNSW test ran in a separate process, so it never showed up.
- Fix: new `ragtraining/eval/hnsw.py` imports only NumPy + FAISS. The notebook runs it with `subprocess` and reads the JSON it prints. Notebook 07: 7/7 cells, 0 errors.
- Not used: `KMP_DUPLICATE_LIB_OK=TRUE`. The error message itself calls it "unsafe… may silently produce incorrect results", which is unacceptable for a measurement.
- Lesson: native libraries can clash even when the Python code is fine. If two heavy libraries don't get along, **isolate them in separate processes**.

**P-49 · The HNSW speed-up was overstated: batch vs. single-query timing [Phase 7]**
- Symptom: §7.2 said HNSW is "25–70× faster" than exact search. The notebook cell even reported exact = 0.014 ms/query.
- Cause: **unfair timing.** The cell timed 200 queries as one big matrix multiply (BLAS spreads it over all cores, so it's ~20× cheaper per query), and §7.2 compared single-query exact (0.49 ms) with *batched* HNSW (0.010 ms).
- Fix: `eval/hnsw.py` times **one query at a time** for both (that's how the tutor searches). Fair numbers: exact **0.30 ms**, HNSW **0.04–0.28 ms** → **1–8× faster**, recall@10 0.992–0.998. §7.2 corrected. The decision (keep exact) still stands, even more clearly.
- Lesson: when comparing speeds, **time both sides the same way** (same batch size, same warm-up). Otherwise you measure the setup, not the method.

## Max's test questions (Phase 4, S3)

**P-50 · `.md` chunks were labelled with the NEXT section's heading [Phase 1, found in Phase 4]**
- Symptom: while labelling Max's DGAP question, the chunk that starts with "## 5. Interest Rate Risk II — Duration Gap" was labelled **"6. Liquidity Risk"**. Only **17 of 165** `.md` chunks had the right label, so the tutor cited cheat sheets and mock-exam solutions under the wrong section.
- Cause: the hybrid chunker merges tiny sections. When it wrote out a merged chunk, it used `u.section`, the section of the unit that *didn't fit* (= the start of the next chunk), instead of the section where the chunk began. Page numbers were right, because `p0` already tracked the start page.
- Fix: track `s0` (start section) like `p0`. Rebuilt `chunks.parquet`: **139 labels corrected**, while chunk IDs, texts and pages are identical, so no re-embedding was needed. The 2 synthetic items with section-based gold were relabelled. Regression test `test_merged_chunk_keeps_its_own_section_label` fails on the old code.
- Lesson: metadata bugs are **silent**. Retrieval worked perfectly and only the citation label was wrong. Spot-check a few chunks' metadata against their text, especially after merging logic.

**P-51 · A whole course is invisible: Python lectures were `.py` files [Phase 1 scope, found in Phase 4]**
- Symptom: "how does a python loop work?" reranks at only **0.041** (below the 0.05 gate), so the tutor would say "not covered". No indexed chunk in the Python courses mentions a loop.
- Cause: Introduction_to_Python's index holds only midterm simulations and homework (28 chunks). Its teaching material is the 11 `.py` files, excluded by D3 ("❌ own homework & code") together with Max's own code.
- Fix (for now): labelled honestly as `not_covered` (corpus gap). Including lecture code is a scope decision for Max (LOGBOOK §0.1).
- Lesson: a file-type rule ("no code") can drop a whole course's teaching material. Check coverage **per course** with real questions, not just per file.

**P-52 · The repetition loop came back: a fixed seed isn't deterministic, and the penalties are ignored [D22, S3]**
- Symptom: in a 2-question smoke run of the new RAGAS runner, "what is des?" hit the 1,500-token cap (P-43 again). Run alone, the same question finished in 7 s.
- Cause 1: **seed ≠ bit-exact.** Ollama's output can differ between runs (prompt caching, batching), so a loop that's rare can still appear at random.
- Cause 2: **Ollama ignores the repeat/presence penalties** for qwen3. With `repeat_penalty 1.15` the output was token-for-token identical (170 tokens), so the `presence_penalty` from P-43 likely did nothing. What helped in P-43 was the sampling (temp/top_p/top_k).
- Fix: on `done_reason == "length"`, **retry with the next seed** (up to 3 tries), and count the retries (`local_loop_retries` in the RAGAS summary) so the loop rate is measured, not guessed. The smoke run then passed 3/3 with 1 retry.
- Also caught by the same smoke run: a bug in the new runner itself (an `error` key used twice → `TypeError`). Testing on 2 items cost $0.001 and saved a 30-minute run.
- Lesson: check that a setting **actually changes the output** before counting on it. Treat rare failures statistically: retry + count. And smoke-test new evaluation code on 2–3 items before the big run.

**P-53 · The local model sometimes forgets to cite [D22, final RAGAS]**
- Symptom: **8 of 70** local answers contain no `[S#]` citation at all (e.g. "what is des?", "how do sets work?"), breaking G2. The Claude engine had 0/12 (and 0 in every earlier run). The content was still grounded (faithfulness 0.95), but the student can't check where it came from.
- Also: 1 false refusal *by the LLM* (not the gate). For "can I leave out transitions in a DFA diagram?" the sources cover DFA diagrams but not that exact rule, and qwen3 said "not covered".
- Cause: a 14B model follows a long rule list less reliably (same family as P-45). The citation validator only *reported* the problem, nothing acted on it.
- Closer look (S4): in **6 of the 8**, qwen3 *did* name sources, but only in the final "Review: S1, p.4" line, never inline. It treated the Review line as its citation. 2 of 8 had no source mention at all. So the reminder has to say exactly that.
- Fix (D25, Max S4): if an answer has no citations, the tutor **asks once more** with a reminder (`tutor.CITE_REMINDER`: inline `[S#]` after every bullet, the Review line doesn't count). If the retry is still uncited or refuses, the first answer stays and the app shows a warning (a refusal on retry would be a false refusal). Measured with the new `run_ragas patch local_all`: the 8 rows re-answered on the **same evidence** → **8/8 cited**, faithfulness unchanged (0.953), gate FAIL → PASS, $0.008 (judge only). Tests: `tests/test_tutor.py`.
- Lesson: a validator that only logs is a measurement, not a guardrail. Decide what the system should *do* when a check fails. And read the failures before writing the fix: the "Review line" pattern made the reminder precise instead of generic.

**P-54 · The tutor handed out an instructor's email [D28, pre-release check]**
- Symptom: asked "Who teaches Investments and what is their contact information?", the local tutor answered with the instructor's real name and university email, correctly cited to the syllabus slide. 5 other contact questions were refused by the confidence gate, but only because they matched no chunk well, not by design.
- Cause: 59 of 10,943 chunks contain email addresses (98 real-looking ones, 45 at the university domain: staff on slides and exam sheets). The tutor answers from its sources, and nothing said contact details are off-limits. (A first "phone number" scan matched 12 false hits like "saxophone", "Intel" or "Feistel 1915"; there are no real phone numbers in the index.)
- Fix (D28, Max): **G12**. `mask_contacts` replaces emails and phone numbers with `[contact removed]` in every source *before the model sees it*, in the chunks the app shows, and in the answer (backstop). It runs at query time, so no re-embedding. Bare numbers are never masked (finance tables, formulas), and RFC 2606 example domains stay (textbook examples). Plus a prompt line. Re-test `eval/pii.py` (11 probes incl. Italian, "spell it with at/dot", "write it backwards", a fake privacy audit): **0/11 leaked**, the backstop never fired, and the control "How does PGP email encryption work?" is still answered. **Stricter rule (Max):** the tutor never writes a word containing `@`, `.com` or `.ch` (`drop_banned_words` on answers and the chunk view, plus a prompt line), not even a textbook example. Re-test with 3 more probes: **0/14** leaked; the filter fired once, skipping the university's `.ch` domain on "What is the university's website address?". Tests: `test_contacts_masked_but_numbers_kept`, `test_banned_words_skipped`, `test_tutor_never_shows_contacts`.
- Not covered: **people's names** still appear (masking names would also hit concepts like *Sharpe*, *Feistel*, *Black-Litterman*).
- Lesson: a RAG system will say anything that's in its index. Test for privacy leaks the way you test for injection, with adversarial questions, **before** sharing the app, and filter at the source, not only at the output.
