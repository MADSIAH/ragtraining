"""Thin wrapper around the Claude API used by the reranker, test-set generator, judge and tutor.

- Default model: claude-opus-5-5 (config.CLAUDE_MODEL). Thinking can't be disabled on it; `effort`
  controls depth and cost (low for simple scoring, higher for generation/judging).
- Server-side refusal fallback is enabled (if the model declines, the API retries on a fallback model).
- `ask_json` uses structured outputs: the reply is guaranteed to match the JSON schema.
- Every call's token usage is accumulated in USAGE so we can report the real cost of each run.
- Local backend (D22, hybrid cost mode): a model named "ollama:<name>" (e.g. "ollama:qwen3:14b") runs on
  this Mac via Ollama instead: $0, no API key needed (the Claude client is only created on a Claude call).
  Same JSON-schema output (Ollama constrains generation to the schema). `effort` is ignored there.
"""
from __future__ import annotations

import json
from collections import Counter

import anthropic

from ragtraining.config import CLAUDE_MODEL

LOCAL_PREFIX = "ollama:"
LOCAL_CTX = 12288      # tokens; Ollama's default (2-4K) would silently cut off the 8 sources (P-42)
LOCAL_MAX_OUT = 1500   # output-token cap; a concise answer is ~250-300 tokens, so hitting it = runaway loop (P-43)
# Qwen3's recommended non-thinking sampling. temperature=0 (greedy) made it repeat bullets until the cap (P-43).
# A fixed seed makes runs mostly reproducible (not bit-exact across runs, P-52).
LOCAL_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, seed=42)   # presence: ignored by Ollama (P-52)

PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}
USAGE: Counter = Counter()
_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _record(model: str, usage) -> None:
    USAGE[(model, "in")] += usage.input_tokens
    USAGE[(model, "out")] += usage.output_tokens


def cost_so_far() -> float:
    return sum(n * PRICES.get(m, (0, 0))[0 if kind == "in" else 1] / 1e6 for (m, kind), n in USAGE.items())


def _ask_local(system: str, user: str, model: str, schema: dict | None, max_tokens: int, tries: int = 3):
    import ollama
    cap = min(max_tokens, LOCAL_MAX_OUT)
    for attempt in range(tries):
        # Loops are random even with a fixed seed (Ollama isn't bit-exact across runs), and Ollama ignores the
        # repeat/presence penalties for qwen3 (same output with or without them, P-52). So: retry with a new seed.
        opts = {**LOCAL_SAMPLING, "seed": LOCAL_SAMPLING["seed"] + attempt}
        resp = ollama.chat(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                           format=schema, think=False, options={"num_ctx": LOCAL_CTX, "num_predict": cap, **opts})
        USAGE[(LOCAL_PREFIX + model, "in")] += resp.prompt_eval_count or 0      # tracked, priced at $0
        USAGE[(LOCAL_PREFIX + model, "out")] += resp.eval_count or 0
        if (resp.prompt_eval_count or 0) >= LOCAL_CTX:
            raise RuntimeError(f"prompt filled the whole {LOCAL_CTX}-token local context; sources were cut off")
        if resp.done_reason != "length":
            break
        USAGE[(LOCAL_PREFIX + model, "loop_retries")] += 1                       # measured: how often it loops
    else:
        raise RuntimeError(f"local model hit the {cap}-token output cap {tries}x (repetition loop, P-43); "
                           "answer discarded")
    text = resp.message.content
    return json.loads(text) if schema else text


def ask(system: str, user: str, *, model: str = CLAUDE_MODEL, effort: str = "medium",
        max_tokens: int = 16000, schema: dict | None = None):
    if model.startswith(LOCAL_PREFIX):
        return _ask_local(system, user, model[len(LOCAL_PREFIX):], schema, max_tokens)
    kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": user}],
                  output_config={"effort": effort, **({"format": {"type": "json_schema", "schema": schema}}
                                                      if schema else {})})
    if model in {"claude-opus-5-5", "claude-sonnet-5-5"}:
        resp = client().beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        resp = client().messages.create(**kwargs)
    _record(model, resp.usage)
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined the request: {resp.stop_details}")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text) if schema else text


def ask_json(system: str, user: str, schema: dict, **kw) -> dict:
    return ask(system, user, schema=schema, **kw)
