"""Unit tests for surrogate.scorer.

All tests run at N=10 with <=100 samples for speed. Each test has a
single, narrow assertion goal described in its docstring.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from surrogate.scorer import (
    GraphData,
    SurrogateGNN,
    SurrogateScorer,
    collate,
)


# ---------------------------------------------------------------------------
# Small fixtures
# ---------------------------------------------------------------------------

def _make_random_graphdata(
    N: int = 10, p_edge: float = 0.4, rng: np.random.Generator | None = None
) -> GraphData:
    """Build a random GraphData on N nodes with Bernoulli(p_edge) edges."""
    rng = rng or np.random.default_rng(0)
    z = rng.uniform(-1.0, 1.0, size=N)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=N)
    r = np.sqrt(1.0 - z * z)
    positions = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1).astype(
        np.float32
    )

    adj = (rng.random((N, N)) < p_edge).astype(np.int64)
    adj = np.triu(adj, k=1)
    adj = adj + adj.T
    if adj.sum() == 0:
        adj[0, 1] = adj[1, 0] = 1

    degree = adj.sum(axis=1).astype(np.float32)
    node_features = np.concatenate([positions, degree[:, None]], axis=1)

    src_dst = np.argwhere(adj == 1)
    edge_index = src_dst.T.astype(np.int64)
    dots = np.clip(
        (positions[edge_index[0]] * positions[edge_index[1]]).sum(axis=1), -1.0, 1.0
    )
    edge_features = np.arccos(dots).astype(np.float32)[:, None]

    num_edges = int(adj.sum()) // 2
    max_e = N * (N - 1) // 2
    graph_features = np.array(
        [num_edges / max_e, num_edges / max_e], dtype=np.float32
    )

    target = float(N) / float(num_edges)

    return GraphData(
        node_features=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index),
        edge_features=torch.from_numpy(edge_features),
        graph_features=torch.from_numpy(graph_features),
        target_score=target,
        num_nodes=N,
    )


def _make_dataset(n: int, N: int = 10, seed: int = 0) -> list[GraphData]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = float(rng.uniform(0.2, 0.7))
        out.append(_make_random_graphdata(N=N, p_edge=p, rng=rng))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_shape() -> None:
    """GNN forward pass returns exactly one scalar prediction per graph."""
    graphs = _make_dataset(8, N=10, seed=1)
    batch = collate(graphs)
    model = SurrogateGNN(hidden_dim=32, n_layers=3)
    out = model(batch)
    assert out.shape == (8,), f"expected (8,), got {tuple(out.shape)}"
    assert model.count_parameters() < 100_000


def test_training_reduces_loss() -> None:
    """Training for 10 epochs strictly decreases average training loss."""
    train = _make_dataset(100, N=10, seed=2)
    val = _make_dataset(20, N=10, seed=3)
    scorer = SurrogateScorer(hidden_dim=32, n_layers=3, device="cpu")
    log = scorer.train(train, val, epochs=10, batch_size=16, verbose=False)
    assert log["train_loss"][-1] < log["train_loss"][0], (
        f"loss did not decrease: {log['train_loss'][0]:.4f} -> "
        f"{log['train_loss'][-1]:.4f}"
    )


def test_spearman_positive_after_training() -> None:
    """After 30 epochs the model achieves positive Spearman rho on val."""
    train = _make_dataset(100, N=10, seed=4)
    val = _make_dataset(40, N=10, seed=5)
    scorer = SurrogateScorer(hidden_dim=32, n_layers=3, device="cpu")
    scorer.train(train, val, epochs=30, batch_size=16, verbose=False)
    metrics = scorer.evaluate(val, ks=(10, 20))
    assert metrics["spearman"] > 0.0, (
        f"Spearman not positive: {metrics['spearman']:.3f}"
    )


def test_predict_batch_count() -> None:
    """`predict_batch` returns one score per datapoint."""
    N = 10
    scorer = SurrogateScorer(hidden_dim=16, n_layers=2, device="cpu")
    rng = np.random.default_rng(6)
    positions = rng.normal(size=(N, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    allowed_mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(allowed_mask, False)
    scorer._positions = positions
    scorer._allowed_mask = allowed_mask
    scorer._N = N
    scorer._r_max = 3.14

    class _FakeDP:
        def __init__(self, data: np.ndarray) -> None:
            self.N = N
            self.data = data

    dps = []
    for _ in range(7):
        a = (rng.random((N, N)) < 0.4).astype(np.uint8)
        a = np.triu(a, k=1)
        a = a + a.T
        a[0, 1] = a[1, 0] = 1
        dps.append(_FakeDP(a.astype(np.uint8)))

    preds = scorer.predict_batch(dps)
    assert preds.shape == (7,)


def test_save_load_roundtrip() -> None:
    """save/load yields identical predictions to the original model."""
    graphs = _make_dataset(12, N=10, seed=7)
    scorer_a = SurrogateScorer(hidden_dim=32, n_layers=3, device="cpu")
    scorer_a._positions = np.zeros((10, 3), dtype=np.float64)
    scorer_a._allowed_mask = np.ones((10, 10), dtype=bool)
    scorer_a._N = 10
    scorer_a._r_max = 0.8
    pred_a, _ = scorer_a._predict_on_graphdata(graphs)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.pt")
        scorer_a.save(path)
        scorer_b = SurrogateScorer(hidden_dim=8, n_layers=1, device="cpu")
        scorer_b.load(path)
        pred_b, _ = scorer_b._predict_on_graphdata(graphs)

    assert np.allclose(pred_a, pred_b, atol=1e-6), (
        f"roundtrip mismatch: max diff {np.max(np.abs(pred_a - pred_b)):.2e}"
    )
    assert scorer_b.hidden_dim == 32 and scorer_b.n_layers == 3


def test_manual_scatter_add() -> None:
    """`index_add_` aggregation matches a hand-computed sum on a triangle."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.int64
    )
    x = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=torch.float32
    )
    messages = x.index_select(0, edge_index[0])
    agg = torch.zeros_like(x)
    agg.index_add_(0, edge_index[1], messages)

    expected = torch.tensor([[3.0, 2.0], [4.0, 0.0], [1.0, 2.0]])
    assert torch.allclose(agg, expected), f"got {agg}, expected {expected}"
