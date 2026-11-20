"""Phase 3: tests for the retrieval math (no indexes or Ollama needed)."""
import numpy as np

from ragtraining.retrieve.search import minmax, mmr, ranks_of, rrf, weighted


def test_cosine_equals_euclid_ranking_for_unit_vectors():
    """The D1 identity: ||a-b||^2 = 2 - 2cos for unit vectors -> identical rankings."""
    rng = np.random.default_rng(0)
    D = rng.normal(size=(200, 32)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    q = rng.normal(size=32); q /= np.linalg.norm(q)
    assert (np.argsort(-(D @ q)) == np.argsort(np.linalg.norm(D - q, axis=1))).all()


def test_rankings_can_differ_for_raw_vectors():
    """With different vector lengths, Euclid and cosine disagree, which is what D1 tests."""
    D = np.array([[1.0, 0.0], [3.0, 0.3]])        # same direction-ish, different length
    q = np.array([1.0, 0.05])
    cos = (D @ q) / np.linalg.norm(D, axis=1) / np.linalg.norm(q)
    euc = -np.linalg.norm(D - q, axis=1)
    assert np.argmax(cos) != np.argmax(euc)


def test_ranks_and_rrf():
    a, b = np.array([3.0, 2.0, 1.0]), np.array([1.0, 3.0, 2.0])
    assert list(ranks_of(a)) == [0, 1, 2]
    fused = rrf([a, b])
    assert np.argmax(fused) == 1            # 2nd + 1st beats 1st + 3rd


def test_weighted_uses_normalized_scores():
    a, b = np.array([100.0, 0.0]), np.array([0.0, 0.2])   # very different scales
    assert np.allclose(weighted([a, b], [0.5, 0.5]), [0.5, 0.5])
    assert np.allclose(minmax(np.array([5.0, 5.0])), [0, 0])


def test_mmr_avoids_duplicates():
    V = np.array([[1, 0], [1, 0], [0, 1.0]])               # chunks 0 and 1 identical
    rel = np.array([1.0, 0.99, 0.5])
    picked = mmr(np.array([0, 1, 2]), rel, V, k=2, lam=0.5)
    assert picked == [0, 2]
