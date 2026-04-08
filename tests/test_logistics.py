"""Tests for src/envs/logistics.py.

Covers:
    1-5: analytical / edge-case correctness of calc_score
    6:   distance constraint respected by random generation
    7:   tokenizer encode/decode roundtrip
    8-9: local_search invariants (monotone, repairs disconnection)

Note: tests 1-6, 8-9 require Axplorer base classes (DataPoint,
BaseEnvironment). Test 7 additionally requires the Axplorer tokenizer.
Install Axplorer first: pip install -e /path/to/axplorer
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.envs.logistics import (
    LogisticsDataPoint,
    _compute_allowed_mask,
    _sample_unit_sphere,
    _sphere_distance,
)
from src.envs.tokenizers import SparseTokenizerSingleInteger


def _install_positions(N: int, r_max: float, seed: int = 0) -> np.ndarray:
    """Populate LogisticsDataPoint class variables with a fresh geometry."""
    rng = np.random.default_rng(seed)
    positions = _sample_unit_sphere(N, rng)
    mask = _compute_allowed_mask(positions, r_max)
    LogisticsDataPoint.POSITIONS = positions
    LogisticsDataPoint.ALLOWED_MASK = mask
    LogisticsDataPoint.R_MAX = r_max
    LogisticsDataPoint.MAKE_OBJECT_CANONICAL = False
    return positions


# ---------------------------------------------------------------------------
# Tests 1-5: analytical / edge cases
# ---------------------------------------------------------------------------

def test_complete_graph_fiedler_value() -> None:
    """K_N has lambda_2 = N, so score = 2N/(N-1).

    Derivation. For the complete graph K_N:
        A = J - I, D = (N-1)I, L = NI - J.
    Spectrum of L: {0 (mult. 1), N (mult. N-1)}.
    Hence lambda_2 = N, |E| = N(N-1)/2, score = 2N/(N-1).
    """
    N = 6
    _install_positions(N, r_max=math.pi)
    dp = LogisticsDataPoint(N=N, init=False)
    dp.data[:] = 1
    np.fill_diagonal(dp.data, 0)
    dp.calc_score()
    expected = 2.0 * N / (N - 1)
    assert dp.score == pytest.approx(expected, abs=1e-6)


def test_path_graph_fiedler_value() -> None:
    """P_N has lambda_2 = 2(1 - cos(pi/N)).

    Eigenvalues: lambda_k = 2(1 - cos(k*pi/N)), k = 0, ..., N-1.
    With |E| = N-1, score = lambda_2 * N / (N-1).
    """
    N = 7
    _install_positions(N, r_max=math.pi)
    dp = LogisticsDataPoint(N=N, init=False)
    for i in range(N - 1):
        dp.data[i, i + 1] = 1
        dp.data[i + 1, i] = 1
    dp.calc_score()
    lambda_2 = 2.0 * (1.0 - math.cos(math.pi / N))
    expected = lambda_2 * N / (N - 1)
    assert dp.score == pytest.approx(expected, abs=1e-6)


def test_disconnected_graph_scores_minus_one() -> None:
    """Disconnected graphs must score -1."""
    N = 4
    _install_positions(N, r_max=math.pi)
    dp = LogisticsDataPoint(N=N, init=False)
    dp.data[0, 1] = dp.data[1, 0] = 1
    dp.data[2, 3] = dp.data[3, 2] = 1
    dp.calc_score()
    assert dp.score == -1


def test_empty_graph_scores_minus_one() -> None:
    """|E| = 0 is invalid: score must be -1."""
    N = 5
    _install_positions(N, r_max=math.pi)
    dp = LogisticsDataPoint(N=N, init=False)
    dp.calc_score()
    assert dp.score == -1


def test_single_node_scores_minus_one() -> None:
    """N = 1 has no edges possible, score -1."""
    N = 1
    _install_positions(N, r_max=math.pi)
    dp = LogisticsDataPoint(N=N, init=False)
    dp.calc_score()
    assert dp.score == -1


# ---------------------------------------------------------------------------
# Test 6: distance constraint
# ---------------------------------------------------------------------------

def test_distance_constraint_respected() -> None:
    """Every edge emitted by random generation must satisfy d <= R_MAX."""
    N = 20
    r_max = 0.9
    positions = None
    for seed in range(20):
        positions = _install_positions(N, r_max, seed=seed)
        mask = LogisticsDataPoint.ALLOWED_MASK
        assert mask is not None
        n_comp, _ = connected_components(
            csgraph=csr_matrix(mask.astype(np.uint8)), directed=False
        )
        if n_comp == 1:
            break
    else:
        pytest.skip("no seed produced a connected allowed-edge geometry")
    assert positions is not None
    found_valid = False
    for _ in range(10):
        dp = LogisticsDataPoint(N=N, init=True)
        if dp.score < 0:
            continue
        found_valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if dp.data[i, j] == 1:
                    d = _sphere_distance(positions[i], positions[j])
                    assert d <= r_max + 1e-12, (
                        f"Edge ({i},{j}) has distance {d:.6f} > R_MAX={r_max}"
                    )
    assert found_valid, "No valid random instance generated in 10 tries"

    # Verify calc_score rejects an illegal edge.
    mask = LogisticsDataPoint.ALLOWED_MASK
    assert mask is not None
    illegal_pair = None
    for i in range(N):
        for j in range(i + 1, N):
            if not mask[i, j]:
                illegal_pair = (i, j)
                break
        if illegal_pair is not None:
            break
    assert illegal_pair is not None, "R_MAX=0.9 should leave some pairs forbidden"
    star = LogisticsDataPoint(N=N, init=False)
    for v in range(1, N):
        if mask[0, v]:
            star.data[0, v] = star.data[v, 0] = 1
    i, j = illegal_pair
    star.data[i, j] = star.data[j, i] = 1
    star.calc_score()
    assert star.score == -1, "calc_score should reject graphs with illegal edges"


# ---------------------------------------------------------------------------
# Test 7: tokenizer roundtrip
# ---------------------------------------------------------------------------

def test_encode_decode_roundtrip() -> None:
    """encode -> decode must reproduce the adjacency matrix bit-for-bit."""
    N = 12
    _install_positions(N, r_max=math.pi)
    tokenizer = SparseTokenizerSingleInteger(
        dataclass=LogisticsDataPoint,
        N=N,
        k=2,
        are_coordinates_symmetric=True,
        extra_symbols=["SEP", "EOS", "PAD", "BOS"],
    )
    n_checked = 0
    for _ in range(10):
        dp = LogisticsDataPoint(N=N, init=True)
        if dp.score < 0:
            continue
        n_checked += 1
        tokens = tokenizer.encode(dp)
        decoded = tokenizer.decode(tokens)
        assert decoded is not None
        assert np.array_equal(decoded.data, dp.data), "roundtrip mismatch"
    assert n_checked > 0


# ---------------------------------------------------------------------------
# Test 8: local_search monotone
# ---------------------------------------------------------------------------

def test_local_search_never_worsens_score() -> None:
    """Phase-2 swap commits only when strictly improving, so monotone."""
    N = 12
    _install_positions(N, r_max=math.pi)
    n_checked = 0
    for _ in range(10):
        dp = LogisticsDataPoint(N=N, init=True)
        if dp.score < 0:
            continue
        pre = dp.score
        dp.local_search(improve_with_local_search=True)
        assert dp.score >= pre - 1e-12, f"score worsened: {pre} -> {dp.score}"
        n_checked += 1
    assert n_checked > 0


# ---------------------------------------------------------------------------
# Test 9: local_search repairs disconnection
# ---------------------------------------------------------------------------

def test_local_search_repairs_disconnected() -> None:
    """Phase-1 repair must connect the graph when the geometry allows."""
    N = 8
    _install_positions(N, r_max=math.pi)
    dp = LogisticsDataPoint(N=N, init=False)
    for i in range(3):
        dp.data[i, i + 1] = dp.data[i + 1, i] = 1
    for i in range(4, 7):
        dp.data[i, i + 1] = dp.data[i + 1, i] = 1
    dp.calc_score()
    assert dp.score == -1
    dp.local_search(improve_with_local_search=False)
    assert dp.score >= 0, f"repair failed: score={dp.score}"
