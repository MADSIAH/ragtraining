"""Phase 2 tests (no Ollama calls needed)."""
import numpy as np

from ragtraining.index.embed import MODELS, _with_prefix
from ragtraining.index.sparse import SparseIndex, tokenize


def test_tokenizer_keeps_technical_terms():
    toks = tokenize("2PL and 3NF use the bid-ask spread", stem=False)
    assert {"2pl", "3nf", "bid-ask", "spread"} <= set(toks)
    assert "the" not in toks and "and" not in toks          # stopwords removed


def test_stemming_merges_word_forms():
    assert tokenize("locking", stem=True) == tokenize("locks", stem=True)


def test_bm25_prefers_rare_matching_term():
    idx = SparseIndex(["feistel cipher rounds", "cipher overview", "market microstructure"])
    s = idx.scores("feistel")
    assert int(np.argmax(s)) == 0


def test_model_prefixes():                                   # wrong prefixes silently hurt quality
    assert _with_prefix("nomic-embed-text", ["x"], "query") == ["search_query: x"]
    assert _with_prefix("nomic-embed-text", ["x"], "doc") == ["search_document: x"]
    assert _with_prefix("bge-m3", ["x"], "query") == ["x"]
    assert {"nomic-embed-text", "qwen3-embedding:0.6b"} <= set(MODELS)
    assert _with_prefix("qwen3-embedding:0.6b", ["x"], "query")[0].endswith("Query: x")
