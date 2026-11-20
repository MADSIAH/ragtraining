# RAG Training: an exam-prep tutor over my own course material

Ask a question about any of my university courses and get a **concise answer that cites course · file · page**,
or an honest *"This isn't covered in your materials."* It runs **locally for $0** (Ollama), with Claude as an optional
higher-quality engine. I built it to **learn RAG end to end**: dense vs. sparse retrieval, fusion, reranking, guardrails,
RAGAS evaluation and agentic RAG, every choice measured on real numbers.

> Built with **Claude Code as a pair programmer**: I made the design decisions, designed the evaluation and judged the
> trade-offs; much of the code was written with Claude. `CLAUDE.md` shows the instructions it worked under.

**Start here:** [`DECISIONS.md`](DECISIONS.md) (one page: key decisions + evidence) · [`PROBLEMS_LOG.md`](PROBLEMS_LOG.md)
(54 problems, each symptom → cause → fix → lesson) · [`LOGBOOK.md`](LOGBOOK.md) (full history, all numbers in §7) ·
[`GUIDELINES.md`](GUIDELINES.md) (the tutor's rules).

## Results (70 test questions: 58 synthetic + 12 written by me)

| finding | number |
|---|---|
| Cosine + Euclidean combined vs. cosine alone (raw vectors) | MRR 0.59 = 0.59: no gain, as ‖a−b‖² = 2 − 2cos θ predicts |
| Embedding model (Qwen3 vs. nomic) | +0.22 MRR: worth far more than the distance metric |
| Reranker on top of hybrid search | MRR 0.80 → 0.86 (local) / 0.90 (Claude) |
| Keyword search on synthetic vs. my real questions | BM25 Hit@5 0.88 → 0.75: synthetic questions flatter it |
| Local engine, RAGAS on all 70 | faithfulness 0.953 · context recall 0.97 · context precision 0.87 · **$0** |
| Local vs. Claude engine on my 12 questions | faithfulness 0.96 vs 0.99 · context precision 0.75 vs 0.95 |
| Guardrail suite (26 cases) | 23/26: off-topic 6/6, not in material 6/6, prompt injection 8/8, ambiguous 1/4 |
| Contact-detail probes (14 adversarial questions) | 0/14 leak an email, phone number or `@`/`.com`/`.ch` word (1/6 before) |
| Multi-query / HyDE once a reranker runs | MRR 0.857 vs 0.858: no gain, so they stay opt-in |

## Architecture

```
 course folder (not included)
     │  ingest/discover.py   classify + exclude (own work, classmates', duplicates, admin)  → data/manifest.csv
     │  ingest/extract.py    text + OCR + quality gate + cleaning                          → data/pages.parquet
     │  ingest/chunk.py      hybrid chunking: page-level slides/exams, ~400-token book chunks → data/chunks.parquet
     ▼
 index/build.py   Qwen3-Embedding 0.6B (dense) · BM25 with stemming (sparse) · nomic raw vectors (Euclid experiment)
     ▼
 question ──► guard/rules.py      input filter (prompt injection, length)
                 ▼
            retrieve/search.py   hybrid = dense ⊕ BM25 via Reciprocal Rank Fusion   [opt-in: multi-query, HyDE]
                 ▼
            retrieve/rerank.py   local bge cross-encoder (default) or Claude Haiku → top 8
                 ▼
            confidence gate      top score < 0.05 → "not covered", no LLM call
                 ▼
            generate/tutor.py    qwen3:14b via Ollama (default, $0) or Claude Opus; sources as <source> data blocks
                 ▼
            citation check       every [S#] must exist; an uncited answer is re-asked once
            contact mask         emails / phones → [contact removed] in sources; no word with @, .com, .ch in output (G12)
   agent/loop.py (opt-in, Claude engine): plan queries → search → grade evidence → re-search (≤ 2 rounds) → tutor
   eval/: retrieval metrics · RAGAS-style judge + regression gate · guardrail suite · chunking · expansion
```

## Use it with your own material

The course material, the index and the test set are **not included** (copyrighted course content).

1. Put your files under `corpus/<Semester N>/<Course>/…` (or change `CORPUS_DIR` in `ragtraining/config.py`).
2. Copy `corpus_rules.example.json` to `corpus_rules.local.json` (git-ignored) and set your own name, so your own
   homework is excluded, plus any corpus-specific rules.
3. Build the data: `python -m ragtraining.ingest.discover`, `…ingest.extract`, `…ingest.chunk`, then
   `python -m ragtraining.index.build` (about 13 min for ~11k chunks on an M4 Pro).
4. For the evaluation, write a test set in the format described in [`testset/README.md`](testset/README.md).

## Setup (macOS, Apple Silicon)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt
ollama pull qwen3-embedding:0.6b && ollama pull qwen3:14b      # local engine: no API key needed
cp .env.example .env          # optional: an Anthropic API key for the Claude engine and the RAGAS judge
```

## Run

```bash
ollama serve                                   # if it isn't running already
.venv/bin/streamlit run app.py                 # → http://localhost:8501 (bound to localhost only)
```

The sidebar switches between the local and the Claude engine, tutor and agent mode, retrieval method, reranker and
course filter. Each answer shows its sources and the retrieved chunks.

## Evaluate

| command | what | cost |
|---|---|---|
| `.venv/bin/python -m pytest -q` | offline tests (no models, no data) | $0 |
| `python -m ragtraining.eval.retrieval [--claude]` | every retrieval method on the test set | $0 / ~$0.10 |
| `python -m ragtraining.eval.guardrails calibrate` / `suite` | refusal threshold · 26 guardrail cases | ~$0.11 / ~$0.25 |
| `python -m ragtraining.eval.run_ragas local all` | RAGAS on every question: local answers ($0), Claude Haiku judge | ~$0.07 |
| `python -m ragtraining.eval.run_ragas gate local_all` | regression gate: PASS/FAIL per metric, exit code 1 on FAIL | $0 |
| `python -m ragtraining.eval.run_ragas patch local_all` | re-answer only rows with citation problems, same evidence | judge only |
| `python -m ragtraining.eval.pii` | privacy probe: does the tutor reveal contact details? | $0 |
| `python -m ragtraining.eval.chunking` · `…eval.advanced retrieval` | chunking comparison · multi-query / HyDE | $0 · ~$0.01 |
| `python scripts/build_notebooks.py` | regenerate the notebooks (outputs are cleared in this repo) | $0 |

Claude prices are in `ragtraining/llm.py`. The whole project cost about $4.40 in API calls.

## License

MIT, see [`LICENSE`](LICENSE). This covers the code and docs only, not any course material.
