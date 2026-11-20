# RAG Training: instructions for Claude

**Start every session by reading `LOGBOOK.md`.** It holds the goal, corpus facts, decisions, the checklist and results. Continue from the first unticked box in its §5.

Rules:
- **Max decides every main design choice** (boxes marked ⚖️). Explain the concept first, spell out what each option implies, recommend one, ask with AskUserQuestion, and check that Max understands. Small mechanical choices: decide, then mention them.
- **Don't pause between steps.** Work straight through mechanical steps and stop ONLY at ⚖️ decisions. Keep chat answers concise.
- Do **not** re-scan `corpus/` to rediscover facts (they're in LOGBOOK §2). That folder is read-only: never modify it.
- **Every problem we hit goes into `PROBLEMS_LOG.md`** (symptom → cause → fix → lesson), so Max learns what can go wrong.
- Keep `SUMMARY.md` (concise, plain language) updated at the end of each phase/major step.
- After each completed step: tick the box, log the numbers in §7, add a line to §8, update the header.
- Secrets live in `.env`: never print, read aloud or commit it. Load it with python-dotenv.
- Keep chat explanations concise and educational. This is a learning project.

## Local models first, the API only when needed (D22, LOGBOOK §3/§4)
- **Default = local, $0, no API key.** Tutor answers: `qwen3:14b` via Ollama (`config.LOCAL_MODEL`). Retrieval: hybrid RRF + local cross-encoder rerank (`tutor.LOCAL_RETRIEVAL`). Embeddings are always local (Qwen3-Embedding, D2/D13). The app's default engine is **Local**. Start `ollama serve` first.
- **New code, smoke tests and demos use the local engine.** No new code path may call the Claude API by default.
- **Claude (API) only when** Max asks for it explicitly, or for **measurement**: the Claude engine (`Tutor(model=CLAUDE_MODEL)`, D19), agent mode (Haiku planner), eval scripts pinned to `CLAUDE_MODEL` (retrieval `--claude`, guardrails, `run_ragas prod/baseline/agent`), the RAGAS judge (Haiku, D17) and test-set generation.
- **Before any paid run:** estimate the cost, tell Max, then log the real spend in LOGBOOK §8 (the budget was small: $5 for the whole project).
- Claude calls use model `claude-opus-5-5` (Haiku 5.5 for judge/rerank/planner, D17), Python `anthropic` SDK, wrapper `ragtraining/llm.py`.
- Local backend settings live in `ragtraining/llm.py`, so don't change them casually. `num_ctx` 12,288 keeps the 8 sources from being cut off (P-42). Sampling follows Qwen's recommendation (temp 0.7, top_p 0.8, top_k 20, seed 42), because greedy decoding loops (P-43). Ollama ignores the repeat/presence penalties (P-52). When the 1,500-token output cap is hit (a loop), the call retries with the next seed, up to 3 tries, and counts the retries.
