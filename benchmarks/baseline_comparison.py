"""Baseline comparison: classical random-graph families vs. logistics target.

Generates N-vertex graphs from several classical families (Barabási–Albert,
Watts–Strogatz, k-nearest-neighbours on the sphere, Erdős–Rényi), enforces
the logistics distance constraint R_MAX against a fixed hub geometry, and
scores each graph by the logistics score λ₂(L)·N/|E|.

Outputs:
    benchmarks/baseline_results.json — per-family, per-variant statistics.
    Prints a summary table to stdout.

Usage:
    python benchmarks/baseline_comparison.py --N 30 --r_max 0.8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import igraph as ig
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.envs.logistics import (  # noqa: E402
    LogisticsDataPoint,
    _compute_allowed_mask,
    _sample_connected_geometry,
)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_adjacency(
    adjacency: np.ndarray, mask: np.ndarray, N: int
) -> "tuple[float | None, int, int, str]":
    """Enforce R_MAX, check connectivity, compute logistics score.

    Args:
        adjacency: (N, N) uint8 symmetric matrix (may contain illegal edges).
        mask: (N, N) bool ALLOWED_MASK.
        N: Number of nodes.

    Returns:
        (score | None, pre_filter_edges, post_filter_edges, status) where
        status is one of "ok", "empty", "disconnected", "degenerate".
    """
    pre_edges = int(adjacency.sum()) // 2
    # Enforce distance constraint: zero out edges outside the budget.
    adjacency = adjacency & mask.astype(np.uint8)

    post_edges = int(adjacency.sum()) // 2
    if post_edges == 0:
        return None, pre_edges, post_edges, "empty"

    n_comp, _ = connected_components(
        csgraph=csr_matrix(adjacency), directed=False
    )
    if n_comp > 1:
        return None, pre_edges, post_edges, "disconnected"

    A = adjacency.astype(np.float64)
    L = np.diag(A.sum(axis=1)) - A
    eigenvalues = np.linalg.eigvalsh(L)
    lambda_2 = float(eigenvalues[1])
    if lambda_2 <= 1e-9:
        return None, pre_edges, post_edges, "degenerate"
    return lambda_2 * N / post_edges, pre_edges, post_edges, "ok"


def _igraph_to_adjacency(g: ig.Graph, N: int) -> np.ndarray:
    """Convert an igraph graph to an (N, N) uint8 symmetric adjacency matrix."""
    A = np.zeros((N, N), dtype=np.uint8)
    for e in g.es:
        i, j = e.tuple
        A[i, j] = 1
        A[j, i] = 1
    return A


# ---------------------------------------------------------------------------
# Graph family generators
# ---------------------------------------------------------------------------

def _gen_barabasi_albert(
    N: int, m: int, rng: np.random.Generator
) -> np.ndarray:
    """Preferential-attachment graph with m new edges per added node."""
    g = ig.Graph.Barabasi(n=N, m=m, directed=False)
    return _igraph_to_adjacency(g, N)


def _gen_watts_strogatz(
    N: int, K: int, beta: float, rng: np.random.Generator
) -> np.ndarray:
    """Small-world graph: ring of degree K rewired with probability beta.

    `dim=1, size=N, nei=K//2` makes a ring where each node connects to its
    K/2 nearest neighbours on each side. `p=beta` rewires each edge with
    that probability. Self-loops and multi-edges are removed.
    """
    g = ig.Graph.Watts_Strogatz(1, N, K // 2, beta)
    g.simplify()  # drop self-loops and parallel edges
    return _igraph_to_adjacency(g, N)


def _gen_knn_sphere(
    N: int, k: int, positions: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """k-nearest-neighbours on the sphere (symmetrised).

    For each node, add an edge to its k nearest neighbours by great-circle
    distance. The resulting graph is symmetric because we symmetrise after
    top-k selection — if i has j in its top-k or vice versa, edge (i, j)
    is added.
    """
    # Cosine similarity: larger = closer.
    cosines = positions @ positions.T
    np.fill_diagonal(cosines, -np.inf)
    A = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        top_k = np.argpartition(-cosines[i], k)[:k]
        for j in top_k:
            A[i, j] = 1
            A[j, i] = 1
    return A


def _gen_erdos_renyi(
    N: int, p: float, rng: np.random.Generator
) -> np.ndarray:
    """Each pair connected independently with probability p."""
    upper = rng.random((N, N)) < p
    upper = np.triu(upper, k=1)
    A = (upper | upper.T).astype(np.uint8)
    return A


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def _run_family(
    name: str,
    variant: str,
    generator: Callable[[np.random.Generator], np.ndarray],
    mask: np.ndarray,
    N: int,
    num_graphs: int,
    rng: np.random.Generator,
) -> dict:
    """Generate num_graphs instances, score them, aggregate statistics."""
    scores: list[float] = []
    edges_per_graph: list[int] = []
    pre_edges_all: list[int] = []
    post_edges_all: list[int] = []
    status_counts: dict[str, int] = {"ok": 0, "empty": 0, "disconnected": 0, "degenerate": 0}
    best_score = -1.0
    best_edges = 0
    for _ in range(num_graphs):
        A = generator(rng)
        score, pre_e, post_e, status = _score_adjacency(A, mask, N)
        pre_edges_all.append(pre_e)
        post_edges_all.append(post_e)
        status_counts[status] += 1
        if score is None:
            continue
        scores.append(score)
        edges_per_graph.append(post_e)
        if score > best_score:
            best_score = score
            best_edges = post_e
    arr = np.array(scores) if scores else np.array([-1.0])
    return {
        "num_valid": len(scores),
        "num_attempted": num_graphs,
        "scores": scores,
        "mean": float(arr.mean()) if scores else float("nan"),
        "median": float(np.median(arr)) if scores else float("nan"),
        "max": float(arr.max()) if scores else float("nan"),
        "min": float(arr.min()) if scores else float("nan"),
        "std": float(arr.std()) if scores else float("nan"),
        "mean_edges": float(np.mean(edges_per_graph)) if edges_per_graph else float("nan"),
        "best_graph_edges": best_edges,
        "mean_pre_filter_edges": float(np.mean(pre_edges_all)),
        "mean_post_filter_edges": float(np.mean(post_edges_all)),
        "status_counts": status_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare classical random-graph families on the logistics score."
    )
    parser.add_argument("--N", type=int, default=30, help="Number of hubs")
    parser.add_argument(
        "--r_max", type=float, default=0.8, help="Max edge distance (radians)"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--num_graphs", type=int, default=500, help="Graphs per variant"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Sample a connected geometry using the logistics helper, so this
    # benchmark uses exactly the same family of positions the trainer sees.
    positions, mask = _sample_connected_geometry(args.N, args.r_max, rng)
    print(
        f"Sampled N={args.N} positions with R_MAX={args.r_max} "
        f"(allowed pairs: {int(mask.sum()) // 2}/{args.N * (args.N - 1) // 2})"
    )

    families: "list[tuple[str, str, Callable[[np.random.Generator], np.ndarray]]]" = [
        ("barabasi_albert", "m=2", lambda r: _gen_barabasi_albert(args.N, 2, r)),
        ("barabasi_albert", "m=3", lambda r: _gen_barabasi_albert(args.N, 3, r)),
        ("barabasi_albert", "m=4", lambda r: _gen_barabasi_albert(args.N, 4, r)),
        ("watts_strogatz", "K=4_beta=0.3", lambda r: _gen_watts_strogatz(args.N, 4, 0.3, r)),
        ("watts_strogatz", "K=6_beta=0.5", lambda r: _gen_watts_strogatz(args.N, 6, 0.5, r)),
        ("knn_sphere", "k=3", lambda r: _gen_knn_sphere(args.N, 3, positions, r)),
        ("knn_sphere", "k=4", lambda r: _gen_knn_sphere(args.N, 4, positions, r)),
        ("knn_sphere", "k=5", lambda r: _gen_knn_sphere(args.N, 5, positions, r)),
        ("erdos_renyi", "p=0.10", lambda r: _gen_erdos_renyi(args.N, 0.10, r)),
        ("erdos_renyi", "p=0.15", lambda r: _gen_erdos_renyi(args.N, 0.15, r)),
        ("erdos_renyi", "p=0.20", lambda r: _gen_erdos_renyi(args.N, 0.20, r)),
    ]

    results: dict = {}
    for family, variant, gen in families:
        stats = _run_family(family, variant, gen, mask, args.N, args.num_graphs, rng)
        results.setdefault(family, {})[variant] = stats
        sc = stats["status_counts"]
        print(
            f"  {family:20s} {variant:16s} "
            f"valid={stats['num_valid']:4d}/{stats['num_attempted']} "
            f"mean={stats['mean']:.4f} max={stats['max']:.4f} "
            f"|E|pre={stats['mean_pre_filter_edges']:.1f}→post={stats['mean_post_filter_edges']:.1f} "
            f"[disc={sc['disconnected']} empty={sc['empty']}]"
        )

    # Summary table sorted by mean score.
    print("\n" + "=" * 72)
    print(f"{'family':<20} {'variant':<16} {'mean':>8} {'max':>8} {'median':>8} {'|E|':>6}")
    print("-" * 72)
    flat: list = []
    for family, variants in results.items():
        for variant, stats in variants.items():
            flat.append((stats["mean"], family, variant, stats))
    flat.sort(key=lambda row: (float("-inf") if np.isnan(row[0]) else row[0]), reverse=True)
    for _, family, variant, stats in flat:
        print(
            f"{family:<20} {variant:<16} "
            f"{stats['mean']:>8.4f} {stats['max']:>8.4f} "
            f"{stats['median']:>8.4f} {stats['mean_edges']:>6.1f}"
        )
    print("=" * 72)

    out_path = Path(__file__).resolve().parent / "baseline_results.json"
    payload = {
        "config": {
            "N": args.N,
            "r_max": args.r_max,
            "seed": args.seed,
            "num_graphs": args.num_graphs,
            "allowed_pairs": int(mask.sum()) // 2,
        },
        "results": results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
