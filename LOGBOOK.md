# RAG Training · Project Logbook

> **Single source of truth.** If a session ends (tokens, crash, new day), everything needed to continue is here.
> Last updated: **2026-11-20 (session 4, release prep)** · Status: **all Must items done; public copy ready in `../ragtraining`** (leak check PASS, not pushed). Local engine passes the RAGAS gate (D25), contact masking G12 (D28: 0/14 leaked), 41 tests pass · Next action: **§0.1 step 0: Max creates the GitHub repo and pushes `../ragtraining`**

---

## 0. How to resume  ← read this first

**Max:** start Claude Code in `RAG_Training/` and say: *"Read LOGBOOK.md and continue."*

**Claude, on restart:**
1. Read **only this file** (plus files it points to *if needed*). Do **not** re-scan `corpus/`: corpus facts are in §2.
2. Go to §5, find the **first unticked box**, and continue from there.
3. Before any box marked ⚖️: **stop and ask Max**. Explain the concept and the implications of each option, recommend one, use AskUserQuestion. See §6 for prepared options.
4. After finishing a step: tick it `[x]`, add results to §7, add a line to §8, update the header ("Current phase / Next action").
5. Never print, read aloud or commit `.env`.

Legend: `[ ]` todo · `[x]` done · `[~]` in progress · `[-]` skipped (say why) · ⚖️ = Max decides

---

## 0.1 Next session: start here (written 2026-11-20, session 4)
**Say:** *"Read LOGBOOK.md and continue."* Everything is committed; `git status` should be clean. **All Must items are done** (§0.2). What's left is review and optional depth.

**Next, in order:**
0. **Publish (Max):** the public copy is in `../ragtraining` (fresh git repo, 1 commit, no-reply author email, leak check PASS). Rebuild it after any change: `.venv/bin/python scripts/export_public.py ../ragtraining MADSIAH` (keeps its `.git/`; commit there). To publish: fix `gh auth login` (the stored GH_TOKEN is invalid) or use GitHub Desktop → new repo → push. Decide public vs private on GitHub.
1. **Max reviews** the results deck ((private link), private: share it from its Share menu before sending it to anyone; replace the `[Your name]` placeholder on the cover) and `DECISIONS.md` (check that every decision is one you can explain without notes).
2. **Optional, $0:** a fresh full gate run would confirm the citation retry on all 70 (the S4 check re-answered only the 8 failing rows): `.venv/bin/python -m ragtraining.eval.run_ragas local all` (~25 min, ≈ **$0.065** judge) → `run_ragas gate local_all`. Paid only for the judge: ask Max first.
3. **Optional depth:** NeMo Guardrails in `.venv-nemo` (Max must OK the install) · agent tuning (P-31, multi-query inside the plan step) · ambiguous one-liners (P-38: multi-query round-robin as an opt-in "ambiguous" mode) · cascade reranker (local first, Claude only when unsure) · app activity log.
4. **Before publishing anything (e.g. a public GitHub repo):** `testset/synthetic_review.md`, `testset/manual_questions.md`, `testset/synthetic.jsonl`, notebook outputs and `data/` (chunks) contain **excerpts of course material** (likely copyrighted); Max's homework was excluded for privacy. Strip them, and never include `.env`.
5. **Budget:** the whole project ran on a $5 API credit (≈ $4.37 spent). Every Claude call needs a cost estimate first (CLAUDE.md).
6. **Run the app:** `ollama serve` (if not running) → `.venv/bin/streamlit run app.py` (localhost-only via `.streamlit/config.toml`) → http://localhost:8501. **Default engine = Local (qwen3:14b, $0, D22)**; switch to Claude in the sidebar for Opus answers / agent mode. Quality check after any change: `run_ragas local all` then `run_ragas gate local_all` (D25). Exact package versions: `requirements.lock.txt`.

**Done in S4 (details §8):** ⚖️ P11 → D25 regression gate · ⚖️ P-53 → citation retry (8/8 fixed, gate FAIL → PASS) · chunking re-run on 70 q (D12 confirmed) · multi-query/HyDE re-run on 70 q (no gain with the reranker) → ⚖️ P12 → D26 · notebook 06 finished · ⚖️ P13 → D27 deck · `DECISIONS.md`.

## 0.2 Project status vs. Max's original request (session 4). All Must items done; optional depth left.

| Original request | Status |
|---|---|
| RAG over my course material | ✅ 328 files, 10,943 chunks, working tutor + app. Local engine $0 (D22) + Claude engine |
| Euclidean + cosine combined | ✅ built and tested. Answer: adds nothing over cosine (§7.2) |
| Test dense vs. sparse methods | ✅ 13 methods compared; hybrid + Claude rerank best |
| Guidelines | ✅ G1–G11 final (D21). Guardrail suite 23/26 (§7.4), thresholds calibrated |
| RAGAS | ✅ final run (§7.3b): faithfulness local 0.953 (70 q) / Claude 0.987 (Max's 12). Regression gate D25: local **PASS** |
| Agentic RAG | 🟡 works (Claude engine only), not yet better than single search → tune |
| Advanced concepts | ✅ reranking, HNSW, multi-query, HyDE measured on 70 q (§7.5), kept opt-in (D26). Optional: NeMo |
| Free local mode (D22, not in the original request) | ✅ default, verified and measured (§7.3b: ≈ Opus quality on synthetic, $0). Uncited answers fixed by one retry (P-53: 8/70 → 0/70) |
| Max's own test questions | ✅ 15 questions (12 answerable, 3 not covered) + spot-check (14 OK / 2 BAD) → §7.2b, §7.4 |
| Concise results presentation | ✅ 8-slide deck for recruiters (D27): (private link) (private) · one-page `DECISIONS.md` |

**Definition of done, in priority order** (≈ 1 more session):
1. ~~**Must:** Max's ~15 questions + spot-check → re-run retrieval eval~~ ✅ done S3.
2. ~~**Must:** final RAGAS on the final test set → ⚖️ P11 thresholds + P-53 fix~~ ✅ S4 (D25).
3. ~~**Must:** concise results presentation (⚖️ P13) + finish notebooks 06–07~~ ✅ S4 (D27, deck + DECISIONS.md).
4. **Optional depth:** NeMo Guardrails comparison · ~~chunking comparison~~ ✅ S4 · agent tuning · fix ambiguous questions (P-38).

## 1. Goal

An **exam-prep tutor** over Max's course material that answers with page-level citations (e.g. *course · file · p.12*). It is built to **learn** RAG end to end:
- **Dense retrieval:** cosine **and** Euclidean, plus an experiment on combining them (un-normalized vectors)
- **Sparse retrieval:** BM25, then **hybrid fusion** and a **reranker**
- **Measured comparisons:** Recall@k, MRR, nDCG, then **RAGAS**
- **Guidelines/guardrails:** custom first, NeMo for comparison
- **Agentic RAG** plus advanced techniques (HyDE, multi-query, HNSW)
- **Delivery:** Streamlit app and a **concise results presentation** built on our real numbers

Background notes: `rag_and_vector_embeddings_summary.md`.

---

## 2. Corpus facts (scanned 2026-11-18)

- My own course folders: **read-only and not included in this repo** (copyrighted course material).
- **387 PDFs · 9,464 pages** + 44 `.md`, 23 `.docx`, 30 `.py`, 33 `.txt`, 11 `.xlsx`, 4 `.tex`, 3 `.pptx`, 2 `.ipynb`, in 20 course folders over 3 semesters
- Language: ~97% English (a few German and Italian files)
- **Noise to exclude:** ~25,000 files from `.venv/`, `venv/`, `site-packages/`, `node_modules/`, `__pycache__/`, `.git/`, `lib/` (676 MB, mostly in one course)
- **Scanned / image-only PDFs (9):** old exams, a formula sheet, homework diagrams and a scanned textbook copy
- **Large single source:** one 626-page textbook ≈ 7% of all pages. Needs per-source balancing.
- **Personal work** (to EXCLUDE per D3): files with my name in them, my own submissions in homework, assignment and project folders, and code.
  ⚠️ Nuance: homework folders also hold the official assignment sheets and **solutions**. Those count as problem sets and are KEPT. This is decided in the manifest review (Phase 1).

---

## 3. Environment

- Mac **M4 Pro, 24 GB RAM** · macOS (Darwin 25.5) · Python **3.13.5** (`/opt/anaconda3`)
- Already available: PyMuPDF 1.27.2, pypdf, python-dotenv · **Not yet installed:** `anthropic` SDK
- **Ollama** installed (`/opt/homebrew/bin/ollama`). Models: 5 embedding models (Phase 2), `llama3` (unused) and **`qwen3:14b`** (≈9.3 GB, the local tutor LLM, D22). Run `ollama serve` first
- No `ant` CLI. Auth is the API key in `.env` (loaded with python-dotenv)
- **Tutor default (D22): local `qwen3:14b`** via the `ollama:` backend in `ragtraining/llm.py` (num_ctx 12,288 (P-42), think off, JSON schema, Qwen sampling temp 0.7/top_p 0.8/top_k 20 + seed 42 (penalties ignored by Ollama, P-52), output cap 1,500 → retry with next seed, max 3 tries (P-43, P-52)), $0, no API key, ~20–26 s/answer. Claude is used only when asked for explicitly (app switch, agent mode, evals, judge).
- **Claude model:** `claude-opus-5-5` for every Claude call (thinking always on; control it with `output_config.effort`, default `medium`). For production calls (Phase 5+) include the server-side refusal fallback (`betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`). Cheaper model for bulk/judge work only if Max chooses it (⚖️ P10).
- Pricing reference (cached 2026-11-03): Opus 5.5 $4 / $20 per M tokens in/out · Sonnet 5.5 $2 / $10 · Haiku 5.5 $0.10 / $0.50
- **Rule:** estimate the cost before every paid batch run (synthetic test set, RAGAS) and tell Max.

---

## 4. Decisions made

| # | Topic | Choice | Why / implication |
|---|---|---|---|
| D1 | Euclid + cosine | **Run as an experiment**: cosine-only vs Euclid-only vs combined on **un-normalized** vectors | With unit vectors ‖a−b‖² = 2−2cos θ, so the rankings are identical. Combining only helps if vector length carries signal. Main retriever = dense + BM25 hybrid |
| D2 | Where models run | **Local embeddings (Ollama) + Claude** for generation, agent and RAGAS judge | Re-embedding is free and private. Only the top-k retrieved chunks go to the API. Costs a few $ |
| D3 | Corpus scope | **Slides & lecture notes + textbooks + problem sets/solutions/exams**. ❌ own homework & code | Avoids the tutor citing Max's own mistakes. Textbooks need source balancing |
| D4 | Use case | **Exam-prep tutor** with citations | Page-level chunks, faithfulness is the headline metric |
| D5 | Vector store | **NumPy exact search** (own cos/Euclid code) + **FAISS HNSW** lesson | Full transparency and custom scoring. HNSW recall loss gets measured vs exact |
| D6 | Test set | **~15 by Max + ~60 synthetic by Claude**, Max spot-checks | Realistic and large enough. Synthetic questions alone would inflate BM25 |
| D7 | Guardrails | **Custom Python rules first, NeMo Guardrails as comparison** (separate env) | You see every rule's mechanics. NeMo's dependency stack is isolated |
| D8 | Interface | **Python package + one notebook per phase + Streamlit app + concise results presentation** | Notebooks let you inspect each step. Package = reusable, testable code |
| D10 | Manifest buckets | Homework **assignment sheets & official solutions kept** (they're problem sets, per D3). **Summaries/cheat sheets kept but tagged `summary`** so they rank below official sources. **The language course is excluded.** Classmates' submissions, admin docs and duplicates excluded (md > pdf > docx; never a scanned copy) | 328 files / 8,472 pages kept. Textbooks = 43% of pages, so source balancing is needed (Phase 3) |
| D11 | Scanned PDFs (4 left) | **OCR all 4, keep only pages that pass a quality gate** (enough real words) | Samples: one exam clean; one with printed text OK but handwriting = noise; two mostly images = noise |
| D13 | Embedding model | **Qwen3-Embedding 0.6B** (1024d, long context, query instruction prefix) | Best ranking in the smoke test (MRR 0.83 vs 0.74 next). ~14 chunks/s, so ~13 min for a full re-embed. Normalized-only vectors |
| D14 | D1 experiment model | **nomic-embed-text raw vectors** (legacy endpoint) | Largest length variation (CV 4.1%, shorter text = longer vector), so Euclid can differ from cosine |
| D15 | BM25 stemming | **Snowball stemming on**, English stopwords, keeps tokens like `2pl`, `bid-ask` | Smoke MRR 0.62 → 0.75. Risk: over-merging (university/universe) |
| D16 | Reranker | **Both** local cross-encoder (bge-reranker-v2-m3) and Claude reranker, decided by Phase 4 numbers | Local: free, private, ~1 s/query. Claude: costs per query, sends top-30 to the API |
| D17 | Paid model | **Opus 5.5 for the one-time test set**, **Haiku 5.5 for repeated runs** (Claude-rerank eval, RAGAS judge). Tutor answers on Opus 5.5 | Fits the $5 credit (≈$2–3 total). Haiku judge = slightly noisier RAGAS |
| D18 | 1-hour plan (Max: "do as much as possible") | End-to-end slice first; small choices made with recommended defaults and marked **provisional** | Provisional: GUIDELINES.md, Haiku as agent planner/grader, rerank threshold 0.05, RAGAS on 20-question subsets |
| D19 | Default pipeline (since D22: the **Claude engine**, not the default) | **Hybrid RRF + Claude rerank (Haiku 5.5)**, refuse if top rerank score < 3/10 (provisional threshold). Agent loop = optional mode | Best measured: Hit@5 1.00, MRR 0.91. ~$0.0016/question, ~2 s. Top-30 chunks go to the API |
| D20 | Answer style | **As concise as possible, short bullets around key words, key words in bold** (G8, G11; set by Max) | 129-word answer vs ~400 before, citations unchanged, cheaper per answer |
| D21 | Provisional items (D18) | **Confirmed:** GUIDELINES G1–G11 final · Haiku 5.5 stays agent planner/grader · refusal thresholds **calibrated from data** (answerable vs. unanswerable rerank scores), not guessed | Strict grounding = refuses when the material lacks the answer. Thresholds set in §7.4 |
| D22 | Cost mode (Max, 2026-11-19) | **Hybrid, local by default:** tutor + app default = **qwen3:14b via Ollama + hybrid RRF + local rerank** (`config.LOCAL_MODEL`, `tutor.LOCAL_RETRIEVAL`), **$0, no API key**. Claude engine (Opus + Claude rerank, D19) stays available as `Tutor(model=CLAUDE_MODEL)` / app switch, and for agent mode. Eval scripts are **pinned to Claude** so §7.3–7.4 stay comparable; new RAGAS pipeline `local` (answers $0, Haiku judge only) | Was ~$0.04/question (90% = Opus answer). Local: weaker citation/refusal discipline, slower; to be measured with `run_ragas local` vs `prod`. D19 is now the "Claude engine", not the default |
| D23 | Python lecture code (P-51) | **Keep excluded** (Max, S3). The 11 `.py` files of the Python course (lessons, slides, exercises, mock exams, one own solution) stay out of the index, as D3 decided | Basic Python questions get an honest "not covered" (G3). The gap is **documented**, not hidden: `manual_questions.md` M06 is labelled `not_covered` |
| D24 | Final RAGAS run (budget ≈ $1.15) | **Local engine on all 70 questions + Claude engine on Max's 12** (Max, S3) ≈ $0.62 | Measures the default (local) engine on everything, and compares both engines on the same 12 real questions. Leaves ≈ $0.5 for re-runs. Files `data/ragas_local_all.json`, `ragas_prod_manual.json` (S1 files untouched) |
| D25 | RAGAS gate (P11) + uncited answers (P-53) | **Regression gate** on all 70 questions (Max, S4): faithfulness ≥ 0.92 · context recall ≥ 0.90 · context precision ≥ 0.80 · answer relevancy ≥ 0.65 (soft: warns only) · 0 answer errors · ≤ 3% answers with citation problems. **Uncited answer → retry once** with a reminder (inline `[S#]` after every bullet, the "Review:" line doesn't count); if the retry is still uncited or refuses, keep the first answer + app warning | Each bar is ≥ one judge-noise step (±0.03, P-36) below the S3 local score, so a real drop fails and noise doesn't. Retry is $0 and slows only the ~11% affected answers. Code: `tutor.CITE_REMINDER`, `run_ragas gate <name>` (exit 1 on FAIL), `run_ragas patch <name>` (re-answers only flagged rows on the same evidence) |
| D26 | Advanced techniques kept (P12) | **Default unchanged** (hybrid RRF + reranker). Multi-query (RRF / round-robin), HyDE and the agent loop **stay as opt-in, measured methods** (`search(method=...)`, notebook 07; agent in the app on the Claude engine) (Max, S4) | With the reranker on, expansion gives identical retrieval (MRR 0.857 vs 0.858, 70 q, §7.5). The only gain is sense coverage on ambiguous one-liners (5/8 → 7/8, 4 q), which doesn't pay for one extra LLM call per question. P-38 stays a documented weak spot |
| D27 | Presentation (P13) | **Audience: recruiters / interviews. Format: ~8-slide deck** (claude.ai slides page, private, downloads as .pptx/PDF) (Max, S4) | Portfolio piece: problem → architecture → 3–4 decisions backed by our numbers → lessons. Says upfront it was built with Claude Code as a pair programmer. No course excerpts on the slides (copyright, §0.1 3b) |
| D28 | Personal contact details (P-54) | **Mask emails + phone numbers in the sources and the answer** (G12, `mask_contacts`), names stay. **Stricter (Max):** the tutor never writes a word containing `@`, `.com` or `.ch`; such a word is skipped (`drop_banned_words`, answers + chunk view + prompt line) (Max, S4) | Pre-release probe: 1/6 contact questions leaked an instructor's email. Masking at query time = $0, no re-embedding; bare numbers and RFC 2606 example domains untouched. Re-test **0/14** leaked, PGP control still answered |
| D29 | GitHub release (Max, S4) | **Fresh repo, one clean commit** (no-reply email) · test set excluded (format README with invented examples) · notebook outputs cleared · LOGBOOK/PROBLEMS_LOG/SUMMARY/CLAUDE.md published, scrubbed | `scripts/export_public.py` builds the copy in `../ragtraining` and **fails** on emails, private terms, corpus file names, or any ≥120-char corpus / ≥80-char test-set passage. Corpus-specific rules moved to git-ignored `corpus_rules.local.json`; scrub rules in `export_scrub.local.json` |
| D30 | Project name (Max, S4) | **No university name anywhere:** project "RAG Training", package `ragtraining`, repo `ragtraining`, corpus path `corpus/` (git-ignored link to the course folder), app "RAG Tutor" | Avoids any trouble with the university. The university's short name, city and full name are in the export's private terms, so the leak check enforces it. The old public repo (with the old name in its 2 commits) is deleted and recreated, not just renamed |
| D12 | Chunking | **Hybrid**: page-level for slides/exams/problem sets/solutions/project (tiny slides merged to ≥100 tok, max 512). Recursive ~400 tok + ~60 overlap for textbooks/summaries | Exact page citations where it matters. Coherent book chunks. 10,943 chunks, median 334 tok, 0.4% tiny |
| D9 | Version control | **Local git only**, branch `main`, commit per finished step | Undo history. Nothing online. `.env`, `corpus/`, `.venv/`, `data/` are ignored |

---

## 5. Checklist

### Phase 0 · Setup
- [x] Read RAG notes, scan corpus, record facts (§2) and environment (§3)
- [x] 8 framing questions answered → decisions D1–D8 (§4)
- [x] Create `LOGBOOK.md`, `CLAUDE.md`, `.gitignore`, `.env` (placeholder), `.env.example`, `scripts/check_api_key.py`
- [x] **Max:** create Anthropic API key, paste key into `.env`
- [x] **Max:** add credits ($5 added 2026-11-18; top up before RAGAS runs if needed)
- [x] Create project venv `.venv` + `requirements.txt` (anthropic 2.10.0 installed). Run things with `.venv/bin/python`
- [x] Run `.venv/bin/python scripts/check_api_key.py` → Step 1 ✓ key valid, Step 2 ✓ billing works
- [x] Start Ollama server, confirm it responds (`ollama list`): 5 embedding models + llama3 present (2026-11-19)
- [x] ⚖️ `git init` + first commit → **D9: local git only** (no GitHub). Commit after each finished step
- [x] Package skeleton: `ragtraining/` (`config.py`, `ingest/`, `index/`, `retrieve/`, `generate/`, `guard/`, `eval/`, `agent/`), `notebooks/`, `tests/`, `data/` (git-ignored)

### Phase 1 · Ingestion
- [x] File discovery with exclusion rules → `ragtraining/ingest/discover.py` → `data/manifest.csv` (497 files; keep = yes/no + reason per file)
- [x] ⚖️ Manifest review → **D10**. Discrete `l*` = official solutions, `u*` = assignment sets (verified by reading them)
- [x] ⚖️ Scanned PDFs → **D11** OCR (tesseract via PyMuPDF `get_textpage_ocr`, tessdata `/opt/homebrew/share/tessdata`) + per-page quality gate
- [x] Extract text → `ragtraining/ingest/extract.py` → `data/pages.parquet` (8,446 units from 313 files). Cleaning: OCR + quality gate, headers/footers, de-hyphenation, ligatures, incremental slide builds, textbook TOC/index/bibliography pages dropped
- [x] Non-PDF kept files: `.md` (split by heading) and `.docx` (python-docx) → section-level units
- [x] Built 4 chunk sets → `ragtraining/ingest/chunk.py` → `data/chunks_{page,fixed,recursive,hybrid}.parquet`
- [x] ⚖️ **Chunking strategy** → **D12 hybrid** (stats in §7.1). Other 3 sets kept for a Phase 4 chunking comparison
- [x] Chunk metadata: `chunk_id, path, course, semester, doc_type, page_start, page_end, section, text, n_tokens`
- [x] `data/chunks.parquet` (= hybrid, 10,943 chunks) + stats in §7.1
- [x] Notebook `notebooks/01_ingestion.ipynb` (executed) · tests `tests/test_ingest.py` (13 passing, one per PROBLEMS_LOG entry)

### Phase 2 · Indexing
- [x] ⚖️ **Embedding model** → **D13 Qwen3-Embedding 0.6B**. Smoke test of 5 models (`ragtraining/eval/smoke.py`, results §7.0)
- [x] Measure vector norms → Ollama `/api/embed` always normalizes. Raw only via legacy endpoint, and only nomic (CV 4.1%) / bge-m3 (1.5%) carry length info (P-23, P-26) → **D14: D1 experiment on nomic raw**
- [x] Embed all chunks → `data/emb_qwen3.npy` (dense, normalized) + `data/emb_nomic_raw.npy` (raw, D1). Resumable builds (`ragtraining/index/build.py`); after parallel builds run `write_meta()` (P-28)
- [x] BM25 index → **D15 with stemming** → `data/bm25.pkl` (`ragtraining/index/sparse.py`)
- [x] Notebook `02_indexing.ipynb` (norms, cos↔Euclid identity check, t-SNE map by course, BM25 tokens/IDF). Built by `scripts/build_notebooks.py`

### Phase 3 · Retrieval methods
- [x] Dense **cosine** (Qwen3) + cos_nomic → `ragtraining/retrieve/search.py`
- [x] Dense **Euclidean** (L2 on nomic raw vectors)
- [x] Combined cos + Euclid: both variants implemented (weighted α, RRF). ⚖️ choice made with Phase 4 numbers
- [x] Sparse **BM25**
- [x] Hybrid dense + sparse: both RRF and weighted implemented. ⚖️ choice made with Phase 4 numbers
- [x] ⚖️ **Reranker** → **D16 both**: local `BAAI/bge-reranker-v2-m3` (sentence-transformers, MPS) + Claude reranker (`ragtraining/retrieve/rerank.py`), measured in Phase 4
- [x] Source balancing: per-file cap + doc_type weights (summary ×0.85, D10) + MMR implemented. Settings tuned in Phase 4
- [x] Metadata filters (course, doc_type). Tests `tests/test_retrieve.py`
- [x] Notebook `03_retrieval.ipynb`: top-5 side by side per method (normal / paraphrase / exact-term query), RRF worked example, top-10 Jaccard overlap

### Phase 4 · Test set & retrieval evaluation
- [x] **Max** writes ~15 exam-style questions → `testset/manual_questions.md` (15, S3; fields filled by Claude from the corpus, parsed by `eval/manual.py`)
- [x] Claude (Opus, D17) generated 70 → **60 usable** synthetic Q&A → `testset/synthetic.jsonl` (17 courses). Pilot first: $0.0115/item. Real cost **$0.64**
- [x] Spot-check ≥15 synthetic items → `testset/synthetic_review.md`: **done by Claude at Max's request** (16 items vs. full gold chunk, 14 OK · 2 BAD, dropped by `load_testset`)
- [x] Metrics: Hit@1/5/10, Recall, MRR, nDCG@10, latency (`ragtraining/eval/retrieval.py`)
- [x] Run every method from Phase 3 → §7.2 → **D1 answered**
- [x] ⚖️ Default pipeline → **D19** hybrid RRF + Claude rerank (Haiku), k=8
- [x] Chunking comparison (4 chunkings × 4 methods, `eval/chunking.py`, $0) on the final 70 questions → §7.6, **D12 confirmed** (S4)
- [x] Notebook `04_retrieval_eval.ipynb` (quality bars, quality vs latency, α sweeps, per-question Hit@5 heatmap + hardest questions)

### Phase 5 · Generation + guidelines/guardrails
- [x] Tutor → `ragtraining/generate/tutor.py` (Opus 5.5, structured output, [S#] citations, retrieval = hybrid RRF + local rerank). Tested: Feistel ✓ cited, pizza → refused at $0, injection → blocked
- [x] ⚖️ `GUIDELINES.md` G1–G11 → **D21 confirmed final** (G8/G11 concise + bold key words set by Max)
- [x] Custom guardrails → `ragtraining/guard/rules.py`: input injection filter, confidence threshold (rerank < 0.05 → refuse, no LLM call), chunk sanitizer, citation validator. Grounding = RAGAS faithfulness (Phase 6). Tests `tests/test_guard.py`
- [x] Guardrail test suite `ragtraining/eval/guardrails.py` (26 cases: off-topic, unanswerable, injection in question / in document, ambiguous, controls) + **threshold calibration** (D21) → §7.4. **23/26 pass**; ambiguous 1/4 (P-38)
- [x] **G12 contact masking** (D28, P-54): `mask_contacts` on sources + answers; + banned words (`@`, `.com`, `.ch`); probe `ragtraining/eval/pii.py` **0/14** leaked ($0) → §7.4
- [ ] NeMo Guardrails in `.venv-nemo`, same test suite → compare in §7.4
- [x] Notebook `05_generation_guardrails.ipynb` (score distributions + threshold table, suite results, injection demo, ambiguity)

### Phase 6 · RAGAS
- [x] ⚖️ Judge model → D17 Haiku 5.5
- [x] RAGAS metrics implemented directly (`ragtraining/eval/ragas_lite.py`, Haiku judge, 1 call/question) + runner `ragtraining/eval/run_ragas.py` (pipelines: baseline / prod / agent)
- [x] Metrics: **faithfulness, answer relevancy, context precision, context recall** (final run §7.3b)
- [x] ⚖️ Pass/fail thresholds → **D25 regression gate** (`run_ragas gate <name>`), local engine **PASS** after the P-53 retry (§7.3b)
- [x] Run on 3 pipelines (baseline 20, prod 20, agent 10) → §7.3
- [x] Notebook `06_ragas.ipynb`: P-33 before/after, remaining unsupported claims, judge noise, **final run (§7.3b), P-53 retry, gate**. Executed S4

### Phase 7 · Advanced + agentic
- [x] Query rewriting / **multi-query** → `ragtraining/retrieve/expand.py` (Haiku writes 3 queries, one per meaning if ambiguous; cached in `data/expansions.json`). Fusion: RRF or **round-robin** (`fuse="rr"`, P-41). Use `search(method="multi_query")` → §7.5
- [x] **HyDE** → same module (Haiku writes a fake answer passage, embedded as a *document*). `search(method="hyde")`, `dense_only=True` for pure HyDE → §7.5. Eval: `ragtraining/eval/advanced.py` ($0.03)
- [x] **Agentic loop** → `ragtraining/agent/loop.py` (Haiku plans/grades, provisional; Opus answers; max 2 extra rounds; widens course filter; trace). Demo: vague 2PL question ✓, CAPM vs Sharpe cross-course ✓
- [x] **FAISS HNSW vs exact** → §7.2 (exact kept: 0.5 ms is fast enough)
- [ ] (optional) Contextual chunk headers / parent-child (small-to-big) retrieval
- [x] ⚖️ Keep which techniques → **D26**: default unchanged, multi-query/HyDE/agent opt-in (re-measured on 70 q, §7.5)
- [x] Notebook `07_advanced_agentic.ipynb` (multi-query, fusion, sense coverage, HyDE, metrics, HNSW via subprocess, agent). **Executed S3** (7/7 cells, after P-48 fix)

### Phase 8 · Delivery
- [x] Streamlit app `app.py` (`.venv/bin/streamlit run app.py --server.address localhost`): chat, tutor/agent mode, method + reranker selectors, course filter, sources, retrieved chunks, agent trace
- [x] ⚖️ Presentation audience, length, format → **D27**: recruiters/interviews, ~8-slide deck
- [x] Concise results presentation built on §7 numbers → 8-slide deck (private link) (private, downloads as .pptx/PDF). Cover has a `[Your name]` placeholder
- [x] **Curated decision record `DECISIONS.md`**: 6 decisions + the D25 gate + 3 problem stories (P-43, P-49, P-50), honest about Claude Code. Strip course excerpts before publishing anything
- [x] **GitHub release prep** (D29): `scripts/export_public.py` → `../ragtraining`, leak check PASS, fresh git repo with one commit. **Not pushed** (Max creates the GitHub repo)
- [x] `README.md` (setup, run, architecture diagram, eval commands with costs). `ingest/chunk.py` got a `__main__` so the rebuild path is real

---

## 6. Pending decisions: prepared options (present these when the box is reached)

| ID | Decision | Options to present (★ = likely recommendation, confirm with data) |
|---|---|---|
| P1 | git init | yes, local only ★ / yes + private GitHub / no |
| P2 | Manifest review | Max eyeballs a per-course table of doc_types; fix misclassified rows |
| P3 | Scanned PDFs | OCR all 9 / OCR only exams + Sipser ★ / skip |
| P4 | Chunking | page/slide-level ★ for slides · token windows w/ overlap ★ for textbooks · recursive · semantic |
| P5 | Embedding model | nomic-embed-text / mxbai-embed-large / bge-m3, chosen after a quick norm + speed + sample-retrieval check |
| P6 | Cos+Euclid merge | min-max + α sweep / RRF of both rankings |
| P7 | Hybrid fusion | RRF ★ (no score calibration needed) / weighted sum |
| P8 | Reranker | local cross-encoder ★ / Claude / none |
| P9 | Default pipeline | decided by §7.2 numbers |
| P10 | Judge model | Opus 5.5 (best, $$) / Sonnet 5.5 / Haiku 5.5 (cheapest) |
| P11 | RAGAS thresholds | ✅ decided → D25 |
| P12 | Advanced techniques kept | ✅ decided → D26 |
| P13 | Presentation | ✅ decided → D27 |

---

## 7. Results (fill as we go)

### 7.1 Corpus after ingestion
| metric | value |
|---|---|
| files kept / excluded | 328 / 169 (127 personal, 26 duplicates, 8 admin, 8 German) |
| pages kept | 8,472 (slides 3,931 · textbooks 3,589 · exams 340 · solutions 193 · problem sets 176 · summaries ~174 · project 69) |
| units after cleaning | 8,544 (dropped: 300 near-empty, 86 TOC/index, 47 bibliography, 19 slide builds, 18 failed OCR) |
| tokens (est.) | ~2.9 M, of which textbooks ~75% |

Chunking comparison (est. tokens = chars/4, final after P-21/P-22 fixes):

| strategy | chunks | median tok | p90 tok | % tiny (<50) | % span pages |
|---|---|---|---|---|---|
| page | 11,639 | 195 | 504 | 10.9 | 0 |
| fixed (400, overlap 60) | 7,923 | 440 | 494 | 0 | 67 |
| recursive (400) | 9,768 | 356 | 397 | 3.9 | 51 |
| **hybrid ✓ D12** | **10,943** | **334** | **455** | **0.4** | **45** |

### 7.0 Embedding smoke test (12 paraphrased questions, 3,367-chunk subset, Hit@5 / MRR@10)
| model | Hit@5 | MRR | chunks/s | notes |
|---|---|---|---|---|
| BM25 naive / +stem | 0.75 / 0.83 | 0.53 / 0.75 | – | stemming helps |
| nomic-embed-text | 0.67 | 0.59 | 54 | raw norms CV 4.1% |
| mxbai-embed-large | 0.83 | 0.74 | 11 | 512-token limit, ~4% chunks cut |
| bge-m3 | 0.75 | 0.71 | 20 | raw norms CV 1.5% |
| **qwen3-embedding:0.6b ✓** | **0.83** | **0.83** | 14 | normalized only |
| embeddinggemma | 0.83 | 0.74 | 13 | normalized only |
Q3 (delegated monitoring) and Q6 (consensus) fail for every method → likely bad gold labels. The real eval is Phase 4.

### 7.2 Retrieval comparison (60 synthetic questions, page-level relevance, @10)
| method | Hit@5 | MRR | nDCG | ms/query |
|---|---|---|---|---|
| cosine (Qwen3) | 0.90 | 0.79 | 0.74 | 1 |
| cosine (nomic) | 0.75 | 0.62 | 0.55 | 1 |
| euclid (nomic raw) | 0.75 | 0.62 | 0.55 | 7 |
| cos+euclid weighted α=.5 | 0.75 | 0.62 | 0.56 | 8 |
| cos+euclid RRF | 0.77 | 0.62 | 0.57 | 9 |
| BM25 (stemmed) | 0.88 | 0.77 | 0.69 | 15 |
| hybrid RRF | 0.92 | 0.80 | 0.74 | 18 |
| hybrid weighted .5 | 0.92 | 0.80 | 0.73 | 17 |
| hybrid RRF + cap 2/file | 0.90 | 0.80 | 0.70 | 19 |
| hybrid RRF + MMR .7 | 0.90 | 0.79 | 0.72 | 20 |
| **hybrid RRF + local rerank** | **0.95** | **0.84** | **0.79** | 1,822 |
| cosine + local rerank | 0.97 | 0.83 | 0.78 | 1,539 |
| **hybrid RRF + Claude rerank (Haiku)** | **1.00** | **0.91** | **0.85** | 2,128 ($0.0016/q) |

Sweeps (no type weights): hybrid weighted dense share α = 0 / .25 / .5 / **.75** / 1 → Hit@5 .88 / .88 / .92 / **.93** / .90 (dense-heavy fusion is best, both together beat either alone). cos↔euclid α sweep is flat (.72–.73) → confirms D1.

**D1 answer (Euclid + cosine):** on nomic raw vectors (length CV 4%), Euclid swaps ~17% of the top-10 and changes the #1 hit in 15% of questions, but quality is **identical** to cosine. Combining them gives +1/60 (noise). Query norm ≈ chunk norm (19.98 vs 19.32), so the length penalty is mild. → Length carries no useful retrieval signal here. The embedding model matters far more (+15 pts Hit@5).
Caveat: synthetic questions favour BM25 a bit (shared wording despite paraphrasing).

**HNSW vs exact (Qwen3, 10,943 × 1024, one query at a time, `eval/hnsw.py`):** exact NumPy = **0.30 ms**/query. FAISS HNSW M=16/ef=32 → recall@10 0.992 at 0.04 ms · M=16/ef=128 → 0.998 at 0.12 ms · M=32/ef=256 → 0.998 at 0.28 ms. **1–8× faster**, irrelevant at this size → keep exact search. (S1 said "25–70×": it compared single-query exact with *batched* HNSW, P-49. FAISS must run in its own process next to PyTorch, P-48.)

### 7.2b Retrieval re-run with Max's questions (S3: 70 questions = 58 synthetic after spot-check + 12 Max, @10, $0.109)
| method | Hit@1 | Hit@5 | MRR | nDCG | Hit@5 **Max** | MRR **Max** | MRR synthetic | ms/query |
|---|---|---|---|---|---|---|---|---|
| cosine (Qwen3) | 0.71 | 0.91 | 0.81 | 0.71 | 1.00 | **0.90** | 0.79 | 2 |
| cosine (nomic) | 0.51 | 0.70 | 0.59 | 0.50 | 0.50 | 0.46 | 0.61 | 1 |
| euclid (nomic raw) | 0.53 | 0.70 | 0.59 | 0.49 | 0.50 | 0.46 | 0.62 | 8 |
| cos+euclid RRF | 0.53 | 0.71 | 0.59 | 0.50 | 0.50 | 0.46 | 0.62 | 10 |
| BM25 (stemmed) | 0.66 | 0.86 | 0.75 | 0.65 | **0.75** | 0.68 | 0.76 | 15 |
| hybrid RRF | 0.70 | 0.93 | 0.80 | 0.72 | 1.00 | 0.79 | 0.80 | 19 |
| hybrid weighted .5 | 0.70 | 0.93 | 0.80 | 0.71 | 1.00 | 0.84 | 0.79 | 17 |
| hybrid RRF + cap 2/file | 0.70 | 0.91 | 0.80 | 0.67 | 1.00 | 0.79 | 0.80 | 19 |
| hybrid RRF + MMR .7 | 0.70 | 0.90 | 0.79 | 0.70 | 0.92 | 0.78 | 0.79 | 19 |
| **hybrid RRF + local rerank** (local default, D22) | 0.77 | 0.97 | 0.86 | 0.77 | 1.00 | **0.94** | 0.84 | 2,056 |
| cosine + local rerank | 0.76 | 0.97 | 0.85 | 0.76 | 1.00 | 0.94 | 0.83 | 2,529 |
| **hybrid RRF + Claude rerank** (Claude engine, D19) | **0.81** | **1.00** | **0.90** | **0.83** | 1.00 | **0.96** | **0.89** | 2,043 |
Sweep (dense share α of weighted hybrid): 0 / .25 / .5 / **.75** / 1 → Hit@5 .86 / .87 / .93 / **.94** / .91 (same shape as S1). cos↔euclid flat again (.67–.69) → **D1 holds**.
**Reading (Max's questions = the reality check D6 asked for):**
- **P-34 confirmed:** BM25 falls from 0.88 (synthetic) to **0.75** Hit@5 on Max's questions. Real students don't reuse the slide wording, and dense search copes with that.
- On short real questions **hybrid RRF ranks worse than plain cosine** (MRR 0.79 vs 0.90): BM25 adds noise on 2–4-word questions. Weighted α=.5 is in between (0.84). **Both rerankers repair it** (0.94 local, 0.96 Claude) → the D19/D22 pipelines stand.
- Ranking of the two engines unchanged: Claude rerank > local rerank, by +0.02 MRR on Max's questions and +0.05 on synthetic.
- Caveats: only 12 questions; Hit@5 = 1.00 is easy because Max's gold sets are broad (whole lectures/chapters), so **MRR is the telling number**. Gold was pooled from BM25/cosine/hybrid top-8 (+ keyword search), which slightly favours those over nomic. The spot-check was done by the same model family that generated the items.

### 7.3 RAGAS (judge Haiku 5.5, answers Opus 5.5)
| pipeline | n | faithfulness | answer relevancy | context precision | context recall | answered | cost |
|---|---|---|---|---|---|---|---|
| baseline (cosine only) | 20 | 0.891 | 0.765 | 0.906 | 0.932 | 19/20 | $0.81 |
| prod (hybrid RRF + local rerank) | 20 | 0.891 | 0.776 | 0.876 | **0.992** | 20/20 | $0.84 |
| agent (plan/grade loop) | 10 | 0.885 | 0.743 | 0.931 | 0.915 | 10/10 | $0.43 |
0 citation errors in all runs. Faithfulness is **underestimated**: the judge sees chunk text but not page metadata, so the tutor's "review slides p.X" pointers (G8) count as unsupported (P-33).

**P-33 fixed (2026-11-19). Same stored answers, re-judged** with the judge seeing the tutor's exact `<source>` blocks (`run_ragas rejudge`, Haiku only, $0.06):
| pipeline | n | faithfulness | answer relevancy | context precision | context recall | unsupported claims |
|---|---|---|---|---|---|---|
| baseline (cosine only) | 20 | 0.891 → **0.952** | 0.779 | 0.917 | 0.925 | 39 → 14 |
| prod_local (hybrid RRF + local rerank) | 20 | 0.891 → **0.963** | 0.762 | 0.851 | 1.000 | 42 → 13 |
The remaining unsupported claims are real tutor inferences. Precision/recall moved ±0.03 with the *same* contexts = judge noise (P-36). Agent run not re-judgeable (session-1 file has no chunk_ids, planner not deterministic).
⚠️ These answers are **pre-D20** (long style) and retrieved with local rerank. The final RAGAS run uses the current tutor (concise, Claude rerank = `prod`) on the final test set (Max's questions).

### 7.3b Final RAGAS (S3, D24: current concise tutor, final test set, judge Haiku 5.5 with `<source>` blocks)
| engine | questions | faithfulness | answer relevancy | context precision | context recall | answered | citation problems | cost |
|---|---|---|---|---|---|---|---|---|
| **Local** (qwen3:14b + hybrid RRF + local rerank, D22) | **all 70** | **0.953** | 0.773 | 0.873 | **0.967** | 69/70 | **8/70 with no [S#] at all** | $0.065 (judge only) |
| · local, synthetic only | 58 | 0.952 | 0.788 | 0.898 | 0.994 | | | |
| · local, Max's only | 12 | 0.960 | 0.699 | 0.749 | 0.838 | | | |
| **Claude** (Opus + hybrid RRF + Claude rerank, D19) | Max's 12 | **0.987** | 0.691 | **0.947** | 0.891 | 12/12 | 0 | $0.452 |
Median answer length: local **72** words (99 on Max's questions), Claude **186** on Max's. Local loop retries: **5** in 70 answers, all rescued (P-52), 0 answer errors.
**Reading:**
- **Both engines are faithful** (0.96 vs 0.99 on the same 12): grounding works on the free engine too.
- **The local engine matches the old Opus numbers on synthetic questions** (S1 re-judged Opus: 0.963 / 0.762 / 0.851 / 1.00 vs local 0.952 / 0.788 / 0.898 / 0.994) at $0.
- **The biggest gap is retrieval, not the LLM:** context precision 0.75 (local rerank) vs 0.95 (Claude rerank) on Max's questions. The Claude reranker puts more relevant chunks into the top 8, even though MRR (first hit) differs only by 0.02 (§7.2b).
- **Answer relevancy is low (~0.70) for both on Max's questions** because of the vague ones ("what is DS?" 0.37–0.40, "trees" ~0.6): answers that cover several meanings regenerate different questions. That's a property of the metric, not a wrong answer.
- **Local weaknesses (P-53):** 8/70 answers have no citations (G2), 1 false refusal by the LLM (DFA diagram question, the sources only partly cover it). Ironically, local answers are *shorter* than Opus's (closer to D20).
- One judge hiccup: "what is encryption?" got no recall score on the Claude run (judge returned no reference statements), so it's left out of that mean.

**S4: P-53 citation retry + P11 gate (D25).** `run_ragas patch local_all` re-answered only the 8 uncited rows on the same evidence with the new retry, then re-judged them ($0.008) → `data/ragas_local_all_patched.json`:
| local, all 70 | faithfulness | answer relevancy | context precision | context recall | answer errors | citation problems | gate |
|---|---|---|---|---|---|---|---|
| S3 run | 0.953 | 0.773 | 0.873 | 0.967 | 0 | 8/70 = 11% | **FAIL** (citations) |
| + retry (patched) | 0.953 | 0.775 | 0.873 | 0.970 | 0 | **0/70** | **PASS** |
| bar (D25) | ≥ 0.92 | ≥ 0.65 (soft) | ≥ 0.80 | ≥ 0.90 | = 0 | ≤ 3% | |
All 8 needed the retry and all 8 came back cited (6/8 had cited only in the "Review:" line). Faithfulness per row: 6 unchanged at 1.0, sort-merge 0.89 → 1.0, late trading 1.0 → 0.83 (one inference, "raises the cost of investing"). Caveat: the retry was measured on the 8 rows that had failed; a fresh full run (~25 min, ≈ $0.065) is the next real gate check.

### 7.4 Guardrails test suite + refusal-threshold calibration (D21)
**Calibration** (`guardrails calibrate`, $0.11): top rerank score of 60 answerable (synthetic) vs. 12 not-covered (6 off-topic + 6 unanswerable) questions.
| reranker | answerable min / median | not-covered max / median | threshold → false refusals · let through |
|---|---|---|---|
| Claude Haiku (0–10) | 8 / 10 | 7 / 2 | 3 → 0/60 · 2/12 · 6 → 0/60 · 1/12 |
| local bge (0–1) | 0.092 / 0.959 | 0.098 / 0.006 | **0.05 → 0/60 · 1/12** · 0.10 → 1/60 · 0/12 |
Tried Claude **6** (margin under the synthetic min) → the e2e suite showed "What is a key?" scoring 4 → **false refusal** (P-37). **Final: Claude 3, local 0.05.** The gate only saves cost; Opus refuses whatever slips through.
**Re-check with Max's questions (S3, $0.025, stored scores re-used for the 70 old ones):** 70 answerable (58 synthetic + 12 Max) vs. 15 not covered (12 suite + Max's 3).
| reranker · threshold | false refusals | let through | Max's 3 not-covered (score) |
|---|---|---|---|
| Claude · **3** | **0/70** | 4/15 | python loop **4** ✗ · steve rogers **3** ✗ · zorgbor 1 ✓ |
| Claude · 5 | 0/70 | 2/15 | would also block python loop + steve rogers, but "What is a key?" (4) → false refusal (P-37) |
| local · **0.05** | **0/70** | 1/15 | python loop 0.041 ✓ · steve rogers 0.010 ✓ · zorgbor 0.0003 ✓ |
Max's 12 answerable: Claude min **7** ("aes vs des", "encryption"), local min **0.109** ("trees in data management", closest call). → **Thresholds unchanged (Claude 3, local 0.05).** On the Claude engine, "python loop" and "steve rogers" pass the gate and rely on Opus saying "not covered" (✗ = reaches the LLM, not a wrong answer).

**Suite** (`guardrails suite`, 26 cases, default pipeline D19, $0.40 incl. one threshold re-run; incremental re-runs re-use valid stored results):
| case type | custom (ours) | how it was stopped / notes | NeMo |
|---|---|---|---|
| off-topic (6) | **6/6** | 4 confidence gate ($0) · 2 slipped through (cat joke 6, "quit my job" 6) → Opus said "not covered" | – |
| unanswerable (6) | **6/6** | all confidence gate, incl. near-misses Kubernetes, Heckscher-Ohlin | – |
| injection in question (5) | **5/5** | 3 regex filter · Italian + paraphrased ("Forget everything above") miss the English regex → stopped by the gate | – |
| injection in document (3) | **3/3** | obvious English → sanitizer removed the line · subtle English + Italian → not removed, **Opus ignored them** (system prompt + `<source>` tags, G5) | – |
| ambiguous (4) | **1/4** | only "normalization" covered both senses. key / regression / transaction → retrieval returned one meaning only (P-38) | – |
| controls (2) | **2/2** | Black-Litterman, beta (re-labelled from ambiguous, P-39) | – |
| **total** | **23/26** | | |

**Privacy probe (S4, D28, local engine, $0, `eval/pii.py`):** 59/10,943 chunks hold emails (98 real-looking, 45 at the university domain); no real phone numbers.
| run | contact probes leaked | how the others ended | control "How does PGP email encryption work?" |
|---|---|---|---|
| before G12 (6 probes) | **1/6**: instructor's name + email, cited | 5 refused by the confidence gate (by luck: no chunk matched well) | – |
| after G12 (11 probes: + Italian, at/dot spelling, backwards, fake privacy audit) | **0/11** | 7 gate · 1 LLM refusal · 3 answered with `[contact removed]` | answered, cited |
| after strict G12 (14 probes: + university website, textbook example address, email domain) | **0/14** | 8 gate · 2 LLM refusals · 4 answered | answered, cited |
The model never sees an address, so the email backstop fired 0 times. The word filter fired once: on the website question qwen3 wrote the university's `.ch` domain, and the word was skipped. Names still appear (accepted, D28).

### 7.5 Advanced techniques (multi-query, HyDE; `eval/advanced.py`, Haiku expansions, $0.03)
60 synthetic questions, page-level relevance, @10:
| method | Hit@1 | Hit@5 | MRR | nDCG | Recall | ms/query |
|---|---|---|---|---|---|---|
| cosine (Qwen3) | 0.70 | 0.90 | 0.794 | 0.737 | 0.830 | 1 |
| HyDE dense | 0.62 | 0.92 | 0.735 | 0.692 | 0.827 | 77 |
| hybrid RRF | 0.72 | 0.92 | 0.803 | 0.740 | 0.832 | 20 |
| multi-query hybrid, RRF fuse | 0.60 | 0.88 | 0.720 | 0.662 | 0.815 | 203 |
| multi-query hybrid, round-robin | 0.72 | 0.92 | 0.800 | 0.746 | **0.856** | 57 |
| **HyDE hybrid** | **0.73** | 0.92 | **0.817** | **0.752** | 0.849 | 95 |
| hybrid RRF + local rerank | 0.75 | **0.97** | **0.846** | 0.792 | 0.885 | 1,731 |
| multi-query RRF + local rerank | 0.75 | 0.97 | 0.846 | 0.790 | 0.881 | 1,677 |
| multi-query RR + local rerank | 0.75 | 0.97 | 0.846 | 0.791 | 0.882 | 2,314 |
| HyDE hybrid + local rerank | 0.75 | 0.97 | 0.846 | 0.793 | 0.885 | 2,139 |
Expansion adds **~2 s** (one Haiku call) per question the first time it's asked. ms/query above excludes that because of the cache.
**Reading:** without a reranker, HyDE helps a little (+0.014 MRR) and RRF-fused multi-query **hurts** (extra queries drift off the exact page). **With a reranker all four are identical** (the reranker re-sorts the same good pages to the top). On specific synthetic questions expansion doesn't pay. Note: the local-rerank baseline measured 0.967 here vs. 0.95 in §7.2 (1/60 question, rerank noise, P-36).

**Sense coverage** on the 4 ambiguous questions (P-38): meanings present in the top-8 chunks, 2 per question, 8 in total:
| config | key | transaction | normalization | regression | total |
|---|---|---|---|---|---|
| hybrid RRF (no rerank) | 1 | 2 | 2 | 2 | **7/8** |
| hybrid + local rerank | 1 | 1 | 1 | 2 | 5/8 |
| hybrid + Claude rerank (D19) | 1 | 1 | 2 | 1 | 5/8 |
| multi-query RRF | 2 | 1 | 2 | 1 | 6/8 |
| multi-query round-robin | 2 | 1 | 2 | 1 | 6/8 |
| multi-query RR + local rerank | 2 | 1 | 2 | 2 | **7/8** |
| multi-query RR + Claude rerank | 2 | 1 | 2 | 1 | 6/8 |
**Reading:** **rerankers reduce diversity**, because they score every chunk against the one original question. Multi-query round-robin is the only setup that keeps both meanings of "key" *after* reranking. Small sample (4 questions), so treat this as indicative. Input for ⚖️ P12; decide after Max's questions.

**Re-run with Max's questions (S4, `advanced retrieval`, 70 questions, $0.003 for Max's 12 expansions)** → `data/eval_advanced_70q.csv`:
| method | Hit@5 | MRR | nDCG | MRR **Max (12)** | MRR synthetic (58) | ms/query |
|---|---|---|---|---|---|---|
| cosine (Qwen3) | 0.91 | 0.806 | 0.709 | 0.896 | 0.787 | 1 |
| hybrid RRF | 0.93 | 0.796 | 0.720 | 0.794 | 0.796 | 16 |
| multi-query, RRF fuse | 0.90 | 0.770 | 0.670 | **0.958** | 0.731 | 193 |
| multi-query, round-robin | 0.93 | 0.800 | 0.712 | 0.833 | 0.793 | 47 |
| HyDE dense | 0.93 | 0.762 | 0.690 | 0.854 | 0.743 | 76 |
| **HyDE hybrid** | 0.93 | **0.837** | 0.737 | 0.896 | **0.825** | 89 |
| hybrid RRF + local rerank (default) | **0.97** | 0.857 | 0.766 | 0.938 | 0.841 | 1,717 |
| multi-query RRF / RR + local rerank | 0.97 | 0.858 | 0.765 | 0.944 | 0.841 | ~1,655 |
| HyDE hybrid + local rerank | 0.97 | 0.858 | 0.770 | 0.944 | 0.841 | 1,713 |
**Reading:** *without* a reranker, expansion matters, and it depends on the question type. Multi-query RRF lifts Max's short, vague questions from 0.79 to **0.96** MRR (it repairs the BM25 noise of P-34) but hurts specific synthetic ones (0.80 → 0.73). HyDE hybrid is the best no-rerank method overall (0.837). *With* the reranker, all four are again **identical** (0.857–0.858, Max's 0.938 → 0.944 = one question moving one rank). Expansions cost one LLM call per new question (Haiku ≈ 1–2 s; the 12 cost $0.003). → With the reranker as default, query expansion adds no measurable retrieval quality. Its only measured benefit is **sense coverage** on ambiguous one-liners (5/8 → 7/8, 4 questions).

### 7.6 Chunking comparison (`eval/chunking.py`, $0, first run S3 on the 60 synthetic questions)
| chunking | chunks | median tok | Hit@5 cos / BM25 / hybrid / **+ local rerank** | MRR cos / BM25 / hybrid / **+ local rerank** |
|---|---|---|---|---|
| page | 11,639 | 195 | 0.92 / 0.88 / 0.92 / **0.95** | 0.76 / 0.73 / 0.78 / **0.79** |
| fixed (400, overlap 60) | 7,923 | 440 | 0.88 / 0.83 / 0.90 / **0.93** | 0.71 / 0.70 / 0.75 / **0.79** |
| recursive (400) | 9,768 | 356 | 0.88 / 0.85 / 0.92 / **0.93** | 0.78 / 0.72 / 0.78 / **0.85** |
| **hybrid ✓ D12** | 10,943 | 334 | 0.88 / 0.87 / 0.90 / **0.95** | 0.78 / 0.76 / 0.78 / **0.84** (nDCG best, 0.78) |
**Reading (preliminary):** all four are within 1–3 questions of each other, so chunking matters much less than the embedding model or the reranker. Hybrid keeps exact page citations at near-best quality → **D12 stands**. Caveats: multi-page chunks overlap the gold page more easily; this run used the pre-S3 test set, and fixed/recursive chunks have no `.md` section label, so the 2 section-based gold items can't match there. (S3 file kept as `data/eval_chunking_s3_60q.csv`.)

**Final re-run (S4, $0, 70 questions = 58 synthetic + Max's 12; Max's gold refs match by text, so they're fair across chunkers):**
| chunking | multi-page chunks | Hit@5 cos / BM25 / hybrid / **+ local rerank** | MRR cos / BM25 / hybrid / **+ local rerank** | + rerank: nDCG · recall | + rerank: MRR **Max** / synthetic |
|---|---|---|---|---|---|
| page | 0% | 0.94 / **0.90** / 0.94 / 0.96 | 0.80 / 0.73 / **0.80** / 0.82 | 0.66 · 0.65 | 0.92 / 0.80 |
| fixed (400, overlap 60) | 67% | 0.93 / 0.81 / 0.90 / 0.94 | 0.75 / 0.69 / 0.77 / 0.82 | 0.70 · 0.70 | 0.90 / 0.80 |
| recursive (400) | 51% | 0.93 / 0.84 / 0.93 / 0.94 | 0.80 / 0.72 / 0.79 / **0.86** | 0.69 · 0.66 | **0.94** / **0.85** |
| **hybrid ✓ D12** | 45% | 0.91 / 0.86 / 0.93 / **0.97** | **0.81** / **0.75** / 0.80 / **0.86** | **0.77 · 0.77** | **0.94** / 0.84 |
**Reading (final):** with the reranker (the pipeline we actually use) **hybrid is best or tied on every metric**: Hit@5 0.97, MRR 0.857 vs recursive 0.864 (half a question), nDCG and recall clearly best (+0.07 over the runner-up). Part of that can be the multi-page overlap effect, but fixed chunks are more often multi-page (67%) and still score lower, so it isn't only that. Without a reranker the four stay within 1–3 questions. Page-level is the best *keyword* (BM25) layout, because small page chunks keep term density high, but it ranks worst after reranking (nDCG 0.66). → **D12 confirmed.** Chunking still matters much less than the embedding model (+15 points) or the reranker.

### 7.7 Local engine smoke test (D22, qwen3:14b + hybrid RRF + local rerank, $0, S3)
| question | result | time | words | citation errors |
|---|---|---|---|---|
| Feistel decryption (InfoSec) | answered, correct, cited | 20 s | 150 | 0 |
| CAPM beta vs Sharpe ratio (cross-course) | answered, correct, cited | 26 s | 118 | 0 |
| pizza dough recipe (off-topic) | refused at the confidence gate (0.003 < 0.05), no LLM call | 1.5 s | 6 | – |
Before the P-43 fix: Feistel looped for **181 s** and crashed. Style is looser than Opus: repeated bullets, an intro line, a missing "Review:" line (P-45). Full quality → `run_ragas local`.

---

## 8. Session log
- **2026-11-18 · S1**: Read RAG notes. Scanned the corpus (§2). Explained the cos/Euclid identity. 8 questions → D1–D8. Created logbook, CLAUDE.md, .gitignore, .env placeholder, key-check script. **Next:** Max creates the key → Phase 0 venv setup.
- **2026-11-18 · S1 (cont.)**: Key valid + $5 credits. venv set up. Local git (D9). Package skeleton. Manifest built and reviewed (D10). OCR tested (D11). **Next:** text extraction + cleaning (incl. OCR + quality gate), then ⚖️ chunking comparison.
- **2026-11-19 · S2**: Ollama started. **D21** (Max): guidelines final, Haiku planner kept, thresholds calibrated from data. **P-33 fixed** (judge sees the tutor's `<source>` blocks) → re-judged stored answers, faithfulness 0.89 → 0.95–0.96 ($0.06). Guardrail suite + calibration (§7.4): 23/26, threshold 6 tried and reverted to 3 (P-37), ambiguous questions collapse onto one sense (P-38). Notebooks 02–06 (07 written). README. **Multi-query + HyDE** built and measured (§7.5): no gain once a reranker runs, but multi-query round-robin helps ambiguous questions; RRF fusion favours the majority meaning (P-41). Chunking-comparison script ready. P-36…P-41 logged. **D22** (local engine by default) was built in a parallel session with Max. Spend this session ≈ $0.60 → total ≈ $3.7. **Next:** §0.1.
- **2026-11-19 · S2 side session (D22)**: Max asked how to save API money → options explained (local Ollama LLM / Claude subscription via `claude -p` / cheaper API: Batch −50%, Sonnet, Haiku) → **D22 hybrid, then "local by default"**. Pulled `qwen3:14b` (9.3 GB). Added the `ollama:` backend in `llm.py` (num_ctx 12,288, think off, JSON schema, $0, no API key), `LOCAL_MODEL`, tutor default = local + `LOCAL_RETRIEVAL`, app Engine switch (Local first; agent mode Claude only), eval scripts pinned to `CLAUDE_MODEL`, new RAGAS pipeline `local`. CLAUDE.md + `HANDOFF.md` written. $0 spent. **Not done:** smoke test (interrupted, local mode unverified), pytest, P-42 entry, SUMMARY line, commit.
- **2026-11-19 · S3**: Verified D22 local mode. The first local answer **looped for 181 s and crashed** (greedy decoding) → Qwen's recommended sampling + seed 42 + 1,500-token cap → 20 s, valid answer (P-43). Smoke test 3/3 (§7.7). pytest import fix (`pytest.ini`) → 31 pass (P-44). P-42…P-47 logged. CLAUDE.md: "local first, API only when needed" section. `HANDOFF.md` was lost in the parallel-session merge (P-46), so §0.1 is the handoff. Committed both sessions' work. Notebook 07 executed after P-48 (FAISS/PyTorch OpenMP clash → own process) and P-49 (HNSW speed-up was batch-vs-single timing, really 1–8×). **Max's 15 questions** → `manual_questions.md` (gold from the corpus), spot-check 14 OK/2 BAD, `eval/manual.py`, retrieval re-run (§7.2b: BM25 drops on real questions, rerankers best) + calibration (thresholds unchanged). Found **P-50** (`.md` section labels off by one, 139 fixed) and **P-51** (Python lectures are excluded `.py` files). Spend ≈ **$0.13** → total ≈ $3.85. Then D23 (Python stays excluded), D24 → **final RAGAS** (§7.3b: local 70 $0.065 + Claude 12 $0.452, two smoke runs $0.004). Found P-52 (loops random, penalties ignored → retry) and P-53 (local answers without citations). Vocabulary (VOCABULARY.md/.pdf, git-ignored). **Session spend ≈ $0.65 → total ≈ $4.36.** **Next:** ⚖️ P11.
- **2026-11-20 · S4**: ⚖️ **P11 → D25** regression gate (F ≥ .92, CR ≥ .90, CP ≥ .80, AR ≥ .65 soft, 0 errors, ≤ 3% uncited; `run_ragas gate`). ⚖️ **P-53 → retry once** with `CITE_REMINDER` (6/8 had cited only in the Review line); `run_ragas patch` re-answered the 8 rows on the same evidence → 8/8 cited, gate FAIL → PASS ($0.008). 5 new tests (38 pass). Notebook 06 finished and executed. Chunking re-run on 70 q ($0) → D12 confirmed (§7.6). Multi-query/HyDE on 70 q ($0.003) → identical with the reranker (§7.5) → ⚖️ **P12 → D26** (default unchanged, rest opt-in). ⚖️ **P13 → D27** (recruiters, ~8 slides) → deck published (private). `DECISIONS.md` written. CLAUDE.md budget line corrected ($1.3 → LOGBOOK tally). **Spend $0.011 → total ≈ $4.37.** **Next:** §0.1 (Max reviews, optional items).
- **2026-11-20 · S4 (release prep)**: ⚖️ **D29** GitHub release: fresh repo, test set out, notebook outputs cleared, docs scrubbed. Audit: no secrets in history, but the 24 commits hold the test set and the private email → fresh history. Found a **classmate's name** in PROBLEMS_LOG + tests and real corpus file names in tests/discover.py → corpus rules moved to `corpus_rules.local.json` (classifier identical on all 497 files), fixtures neutral. `scripts/export_public.py` + leak check (emails, private terms, corpus names, copied passages) → PASS. Max asked to probe the tutor for contact details → **1/6 leaked** an instructor's email (P-54) → ⚖️ **D28** G12 masking → **0/11**; then Max: never write `@`, `.com`, `.ch` → word filter → **0/14** (1 domain skipped). 41 tests pass. $0. **Final double-check:** staff-name tokens (from university email addresses) 0 real hits, no API key, no-reply author/committer, links OK; found a **lecture title** used as an example in 4 places (judge prompt, notebook 06) → generic; export scan now also checks two-word phrases from corpus file names. Public copy: 1 commit, 66 files, 41 tests pass, leak check PASS, **not pushed**.
- **2026-11-20 · S4 (rename, D30)**: Max: no university name. Package renamed to `ragtraining` (46 files), titles → "RAG Training" / "RAG Tutor", `CORPUS_DIR` → `corpus/`, an off-topic test question's city changed. BM25 pickle re-saved under the new module path (backup `data/bm25.pkl.bak`), search + 41 tests OK. Max then **deleted the course folder** (safety). The tutor runs from `data/` and doesn't need it; `data/` still holds the chunk text (git-ignored, never published, also needed by the leak check). To re-run ingestion: `ln -s "/path/to/course folder" corpus`. Fresh public copy in `../ragtraining`.
- **2026-11-18 · S1 (Phases 1–8)**: Extraction + hybrid chunking (D12). 5 embedding models smoke-tested → Qwen3 (D13), nomic raw for D1 (D14), BM25 stemmed (D15). Rerankers (D16). 60-question synthetic test set (Opus, $0.64, D17). Full retrieval eval (§7.2): **D1 answered** (Euclid = cosine quality). Claude rerank best → default (D19). Tutor + guardrails + agent loop + RAGAS-lite (§7.3) + HNSW test + Streamlit app (localhost only, P-35). Concise bold style (D20). 35 problems logged. Spend ≈ $3.1. **Next:** §0.1.

---

## 9. File map
| path | purpose |
|---|---|
| `LOGBOOK.md` | this file: plan, decisions, progress, results |
| `CLAUDE.md` | auto-loaded rules for Claude (points here) |
| `PROBLEMS_LOG.md` | every problem hit: symptom → cause → fix → lesson |
| `SUMMARY.md` | concise plain-language summary of what's been done (update per phase) |
| `ragtraining/index/embed.py` | Ollama embeddings (normalized batch / raw legacy), model prefixes, safe truncation |
| `ragtraining/retrieve/search.py` | all retrieval methods + fusion + filters + MMR + rerank hook |
| `ragtraining/retrieve/rerank.py` | local cross-encoder + Claude reranker |
| `ragtraining/llm.py` | LLM wrapper: Claude (structured output, fallback, cost tracking) + local `ollama:` backend (D22, $0) |
| `ragtraining/generate/tutor.py` | the tutor |
| `ragtraining/guard/rules.py` · `GUIDELINES.md` | guardrails + the rule list |
| `ragtraining/agent/loop.py` | agentic corrective-RAG loop |
| `ragtraining/eval/retrieval.py` · `ragas_lite.py` · `run_ragas.py` · `testset.py` | evaluation (`run_ragas rejudge <name>` = re-score stored answers, Haiku only · `patch <name>` = re-answer only flagged rows · `gate <name>` = D25 pass/fail) |
| `ragtraining/eval/guardrails.py` | guardrail suite (26 cases) + refusal-threshold calibration → `data/guard_suite.json`, `data/guard_scores.csv` |
| `scripts/build_notebooks.py` | generates notebooks 02+ from code (inspection only, $0) |
| `ragtraining/retrieve/expand.py` | multi-query (RRF / round-robin fusion) + HyDE, Haiku, cached in `data/expansions.json` |
| `DECISIONS.md` | curated one-page decision record: 6 decisions + gate + 3 problem stories (S4) |
| `ragtraining/eval/manual.py` · `hnsw.py` | Max's questions (parser, many gold refs, spot-check verdicts) · HNSW benchmark in its own process (P-48) |
| `ragtraining/eval/advanced.py` · `chunking.py` | §7.5 expansion eval + sense coverage (`advanced retrieval` = 70-q retrieval part only) · §7.6 chunking comparison |
| `README.md` · `pytest.ini` | setup, run, architecture, eval commands · test config (`.venv/bin/pytest`, P-44) |
| `scripts/export_public.py` | builds the public copy + leak check (D29); private inputs: `export_scrub.local.json`, `export_private_terms.local.txt` (git-ignored) |
| `corpus_rules.local.json` / `.example.json` | corpus-specific discovery rules (own name, misplaced files, excluded course), local file git-ignored |
| `ragtraining/eval/pii.py` | privacy probe: 14 contact-detail questions + PGP control, local engine, $0 (D28) |
| `tests/test_tutor.py` | citation retry (P-53) + D25 gate, with a fake LLM ($0) |
| results deck (claude.ai, private) | 8 slides for recruiters (D27): (private link) |
| `testset/` | synthetic.jsonl (60 Q), synthetic_review.md, manual_questions.md (Max) |
| `ragtraining/eval/smoke.py` | 12-question smoke test used to pick the embedding model |
| `ragtraining/ingest/discover.py` | classify files → `data/manifest.csv` |
| `ragtraining/ingest/extract.py` | clean text per page/section → `data/pages.parquet` |
| `ragtraining/ingest/chunk.py` | 4 chunking strategies → `data/chunks_*.parquet` |
| `rag_and_vector_embeddings_summary.md` | Max's RAG notes (background) |
| `corpus/` | corpus, read-only, git-ignored |
| `.env` / `.env.example` | secrets (ignored) / template |
| `scripts/check_api_key.py` | verifies key (free call) + billing (≈$0.001 call) |

---

## 10. Concepts glossary (what Max has learned, for quick review)
- **Cosine similarity** = cos of the angle between vectors (direction = meaning). **Euclidean distance** = straight-line distance (direction + length). For unit vectors: ‖a−b‖² = 2 − 2·cos θ → identical rankings.
- **Dense retrieval** = embeddings, good at meaning and paraphrase. **Sparse retrieval (BM25)** = weighted keyword overlap, good at exact terms, formulas, names (e.g. "Feistel", "CAPM").
- **RRF (Reciprocal Rank Fusion)** = merges rankings using ranks only: score = Σ 1/(k + rank), k≈60. Needs no score calibration.
- **Reranker** = a slower, more accurate model that re-scores the top ~50 candidates jointly with the query.
- **HNSW** = approximate nearest-neighbour graph index: much faster at scale, but may miss some true neighbours (measured as recall loss).
- **RAGAS**: *faithfulness* (are the answer's claims supported by the retrieved context?), *answer relevancy* (does it answer the question?), *context precision* (are the retrieved chunks relevant, ranked high?), *context recall* (did we retrieve everything needed?).
- **Regression gate** = fixed pass/fail bars per metric that every change must keep (D25). Set each bar about one noise step below the current score: too loose misses real drops, too tight fails on judge noise alone. **A check that only logs is a measurement; a guardrail acts** (P-53: retry when citations are missing).
