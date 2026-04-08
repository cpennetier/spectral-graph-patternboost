"""Simulated annealing baseline for logistics network design.

Compares random local search, simulated annealing, and KNN
on the lambda_2 * N / |E| objective under geographic constraints.

Usage:
    PYTHONPATH=. python benchmarks/simulated_annealing.py \
        --N 30 --r_max 0.8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ---------------------------------------------------------------------------
# Graph utilities (standalone, no dependency on logistics.py at import)
# ---------------------------------------------------------------------------


def _sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points uniformly on S^2."""
    z = rng.uniform(-1, 1, size=n)
    phi = rng.uniform(0, 2 * np.pi, size=n)
    r = np.sqrt(1 - z ** 2)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _compute_allowed_mask(positions: np.ndarray, r_max: float) -> np.ndarray:
    """Compute boolean mask of allowed edges (great-circle distance <= r_max)."""
    cos_thresh = math.cos(r_max)
    dots = positions @ positions.T
    np.clip(dots, -1.0, 1.0, out=dots)
    mask = dots >= cos_thresh
    np.fill_diagonal(mask, False)
    return mask


def _sample_connected_geometry(
    n: int, r_max: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample positions whose allowed-edge graph is connected."""
    for _ in range(200):
        positions = _sample_unit_sphere(n, rng)
        allowed = _compute_allowed_mask(positions, r_max)
        n_comp, _ = connected_components(
            csgraph=csr_matrix(allowed.astype(np.uint8)), directed=False
        )
        if n_comp == 1:
            return positions, allowed
    raise ValueError(f"Could not sample connected geometry for N={n}, r_max={r_max}")


def _random_spanning_tree(
    n: int, allowed: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Build a random spanning tree via randomized Prim's algorithm."""
    adj = np.zeros((n, n), dtype=bool)
    in_tree = np.zeros(n, dtype=bool)
    start = rng.integers(n)
    in_tree[start] = True
    for _ in range(n - 1):
        candidates = []
        for u in np.where(in_tree)[0]:
            for v in np.where(allowed[u] & ~in_tree)[0]:
                candidates.append((u, v))
        if not candidates:
            break
        u, v = candidates[rng.integers(len(candidates))]
        adj[u, v] = adj[v, u] = True
        in_tree[v] = True
    return adj


def _random_connected_graph(
    n: int, allowed: np.ndarray, rng: np.random.Generator,
    extra_edge_prob: float = 0.3
) -> np.ndarray:
    """Build a random connected graph: spanning tree + random extra edges."""
    adj = _random_spanning_tree(n, allowed, rng)
    for i in range(n):
        for j in range(i + 1, n):
            if allowed[i, j] and not adj[i, j]:
                if rng.random() < extra_edge_prob:
                    adj[i, j] = adj[j, i] = True
    return adj


def _is_connected(adj: np.ndarray) -> bool:
    """Check if graph is connected via BFS."""
    n = adj.shape[0]
    if n == 0:
        return False
    visited = np.zeros(n, dtype=bool)
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop(0)
        for v in np.where(adj[u])[0]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == n


def _compute_score(adj: np.ndarray) -> float:
    """Compute lambda_2 * N / |E|."""
    n = adj.shape[0]
    num_edges = np.count_nonzero(adj) // 2
    if num_edges == 0 or n < 2:
        return -1.0
    degree = adj.sum(axis=1).astype(float)
    laplacian = np.diag(degree) - adj.astype(float)
    eigvals = np.linalg.eigvalsh(laplacian)
    lam2 = eigvals[1]
    if lam2 <= 1e-9:
        return -1.0
    return float(lam2 * n / num_edges)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _get_edges(adj: np.ndarray) -> np.ndarray:
    """Return (M, 2) array of upper-triangle edges."""
    rows, cols = np.where(np.triu(adj))
    return np.column_stack([rows, cols])


def _get_non_edges(adj: np.ndarray, allowed_pairs: np.ndarray) -> np.ndarray:
    """Return allowed non-edges as (K, 2) array."""
    mask = ~adj[allowed_pairs[:, 0], allowed_pairs[:, 1]]
    return allowed_pairs[mask]


def _allowed_upper_pairs(allowed: np.ndarray) -> np.ndarray:
    """Precompute allowed upper-triangle pairs as (K, 2) array."""
    rows, cols = np.where(np.triu(allowed))
    return np.column_stack([rows, cols])


def run_local_search_only(
    n: int, allowed: np.ndarray, rng: np.random.Generator,
    num_runs: int, num_swaps: int = 60
) -> dict:
    """Random init + greedy local search."""
    allowed_pairs = _allowed_upper_pairs(allowed)
    scores = []
    for _ in range(num_runs):
        adj = _random_connected_graph(n, allowed, rng)
        score = _compute_score(adj)
        if score < 0:
            continue
        for _ in range(num_swaps):
            edges = _get_edges(adj)
            if len(edges) == 0:
                break
            idx = rng.integers(len(edges))
            ei, ej = int(edges[idx, 0]), int(edges[idx, 1])
            adj[ei, ej] = adj[ej, ei] = False
            if not _is_connected(adj):
                adj[ei, ej] = adj[ej, ei] = True
                continue
            ne = _get_non_edges(adj, allowed_pairs)
            if len(ne) == 0:
                adj[ei, ej] = adj[ej, ei] = True
                continue
            nidx = rng.integers(len(ne))
            ni, nj = int(ne[nidx, 0]), int(ne[nidx, 1])
            adj[ni, nj] = adj[nj, ni] = True
            new_score = _compute_score(adj)
            if new_score > score:
                score = new_score
            else:
                adj[ni, nj] = adj[nj, ni] = False
                adj[ei, ej] = adj[ej, ei] = True
        scores.append(score)
    return {
        "mean": float(np.mean(scores)),
        "max": float(np.max(scores)),
        "std": float(np.std(scores)),
        "num_valid": len(scores),
    }


def run_simulated_annealing(
    n: int, allowed: np.ndarray, rng: np.random.Generator,
    num_runs: int, num_iterations: int,
    t_start: float = 0.01, t_end: float = 1e-6
) -> dict:
    """Simulated annealing on lambda_2 * N / |E|."""
    allowed_pairs = _allowed_upper_pairs(allowed)
    all_best_scores = []
    for run_idx in range(num_runs):
        adj = _random_connected_graph(n, allowed, rng)
        current_score = _compute_score(adj)
        if current_score < 0:
            continue
        best_score = current_score

        for t in range(num_iterations):
            temperature = t_start * (1.0 - t / num_iterations)
            if temperature < t_end:
                temperature = t_end

            edges = _get_edges(adj)
            if len(edges) == 0:
                break
            idx = rng.integers(len(edges))
            ei, ej = int(edges[idx, 0]), int(edges[idx, 1])
            adj[ei, ej] = adj[ej, ei] = False

            if not _is_connected(adj):
                adj[ei, ej] = adj[ej, ei] = True
                continue

            ne = _get_non_edges(adj, allowed_pairs)
            if len(ne) == 0:
                adj[ei, ej] = adj[ej, ei] = True
                continue

            nidx = rng.integers(len(ne))
            ni, nj = int(ne[nidx, 0]), int(ne[nidx, 1])
            adj[ni, nj] = adj[nj, ni] = True

            new_score = _compute_score(adj)
            if new_score < 0:
                adj[ni, nj] = adj[nj, ni] = False
                adj[ei, ej] = adj[ej, ei] = True
                continue

            delta = new_score - current_score
            if delta > 0 or rng.random() < math.exp(delta / temperature):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
            else:
                adj[ni, nj] = adj[nj, ni] = False
                adj[ei, ej] = adj[ej, ei] = True

        all_best_scores.append(best_score)
        if (run_idx + 1) % 10 == 0:
            print(f"  SA run {run_idx + 1}/{num_runs}: "
                  f"best={best_score:.4f}, "
                  f"running_max={max(all_best_scores):.4f}")

    return {
        "mean": float(np.mean(all_best_scores)),
        "max": float(np.max(all_best_scores)),
        "std": float(np.std(all_best_scores)),
        "num_runs": len(all_best_scores),
    }


def _knn_score(
    n: int, positions: np.ndarray, allowed: np.ndarray, k: int = 5
) -> float:
    """Compute KNN k=5 baseline score."""
    dots = positions @ positions.T
    np.fill_diagonal(dots, -2.0)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        neighbors = np.argsort(-dots[i])
        count = 0
        for j in neighbors:
            if j != i and allowed[i, j]:
                adj[i, j] = adj[j, i] = True
                count += 1
                if count >= k:
                    break
    return _compute_score(adj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all baselines and print comparison."""
    parser = argparse.ArgumentParser(description="SA baseline for logistics")
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--r_max", type=float, default=0.8)
    parser.add_argument("--sa_runs", type=int, default=100)
    parser.add_argument("--sa_iterations", type=int, default=50000)
    parser.add_argument("--ls_runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Sampling geometry: N={args.N}, r_max={args.r_max}")
    positions, allowed = _sample_connected_geometry(args.N, args.r_max, rng)
    n_allowed = np.count_nonzero(np.triu(allowed))
    print(f"  Allowed edges: {n_allowed} "
          f"({100 * n_allowed / (args.N * (args.N - 1) // 2):.1f}%)")

    # KNN baseline
    knn_score = _knn_score(args.N, positions, allowed, k=5)
    print(f"\nKNN k=5 baseline: {knn_score:.4f}")

    # Local search only
    print(f"\nRunning local search only ({args.ls_runs} runs)...")
    t0 = time.time()
    ls_results = run_local_search_only(
        args.N, allowed, rng, args.ls_runs, num_swaps=2 * args.N
    )
    ls_time = time.time() - t0
    print(f"  Done in {ls_time:.1f}s")
    print(f"  Mean: {ls_results['mean']:.4f}, "
          f"Max: {ls_results['max']:.4f}, "
          f"Std: {ls_results['std']:.4f}")

    # Simulated annealing
    print(f"\nRunning simulated annealing ({args.sa_runs} runs, "
          f"{args.sa_iterations} iterations each)...")
    t0 = time.time()
    sa_results = run_simulated_annealing(
        args.N, allowed, rng, args.sa_runs, args.sa_iterations
    )
    sa_time = time.time() - t0
    print(f"  Done in {sa_time:.1f}s")
    print(f"  Mean: {sa_results['mean']:.4f}, "
          f"Max: {sa_results['max']:.4f}, "
          f"Std: {sa_results['std']:.4f}")

    # Print comparison table
    print(f"\n{'=' * 65}")
    print(f"  Comparison Table -- N={args.N}, r_max={args.r_max}")
    print(f"{'=' * 65}")
    print(f"  {'Method':<35} {'Mean':>8} {'Max':>8}")
    print(f"  {'-' * 55}")
    print(f"  {'Local search (random + greedy)':<35} "
          f"{ls_results['mean']:>8.4f} {ls_results['max']:>8.4f}")
    print(f"  {'Simulated annealing':<35} "
          f"{sa_results['mean']:>8.4f} {sa_results['max']:>8.4f}")
    print(f"  {'KNN k=5':<35} {'--':>8} {knn_score:>8.4f}")
    print(f"{'=' * 65}")

    # Save results
    output = {
        "N": args.N,
        "r_max": args.r_max,
        "seed": args.seed,
        "n_allowed_edges": int(n_allowed),
        "knn_k5_score": knn_score,
        "local_search": ls_results,
        "simulated_annealing": sa_results,
    }
    out_path = Path(f"benchmarks/sa_results_rmax{args.r_max:.1f}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
