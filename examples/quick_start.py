"""Quick-start demo: spectral graph design on the unit sphere.

Self-contained -- does NOT require Axplorer. Uses only numpy and scipy.

Demonstrates:
    1. Sample N hub positions uniformly on the unit sphere
    2. Compute the allowed-edge mask for a distance budget r_max
    3. Build a random connected graph satisfying the constraint
    4. Score it: lambda_2(Laplacian) * N / |E|
    5. Improve it with greedy edge swaps
    6. Compare against a KNN baseline

Usage:
    python examples/quick_start.py --N 20 --r_max 1.0 --seed 42
"""

from __future__ import annotations

import argparse
import math
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points uniformly on the unit sphere in R^3.

    Uses the cylindrical-projection (Archimedes' hat-box) construction:
    z ~ U(-1, 1), theta ~ U(0, 2*pi), then
    (x, y, z) = (sqrt(1 - z^2) * cos(theta), sqrt(1 - z^2) * sin(theta), z).
    """
    z = rng.uniform(-1.0, 1.0, size=n)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def compute_allowed_mask(positions: np.ndarray, r_max: float) -> np.ndarray:
    """Boolean mask: (i, j) is True iff great-circle distance <= r_max."""
    cosines = np.clip(positions @ positions.T, -1.0, 1.0)
    distances = np.arccos(cosines)
    mask = distances <= r_max
    np.fill_diagonal(mask, False)
    return mask


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(adj: np.ndarray, N: int) -> float:
    """Compute lambda_2(L) * N / |E|.

    Returns -1.0 for invalid graphs (disconnected, empty, N < 2).
    """
    if N < 2:
        return -1.0
    num_edges = int(adj.sum()) // 2
    if num_edges == 0:
        return -1.0

    n_comp, _ = connected_components(
        csgraph=csr_matrix(adj.astype(np.uint8)), directed=False
    )
    if n_comp > 1:
        return -1.0

    degree = adj.sum(axis=1).astype(np.float64)
    laplacian = np.diag(degree) - adj.astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    lambda_2 = float(eigenvalues[1])
    if lambda_2 <= 1e-9:
        return -1.0
    return lambda_2 * N / num_edges


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def random_connected_graph(
    N: int, mask: np.ndarray, rng: np.random.Generator, extra_prob: float = 0.3
) -> np.ndarray:
    """Build a random connected graph respecting the allowed-edge mask.

    Phase 1: random spanning tree via multi-pass random Prim.
    Phase 2: add each non-tree allowed edge with probability extra_prob.
    """
    adj = np.zeros((N, N), dtype=np.uint8)
    in_tree = np.zeros(N, dtype=bool)
    in_tree[0] = True
    remaining = list(rng.permutation(np.arange(1, N)))
    progress = True
    while remaining and progress:
        progress = False
        still_waiting = []
        for v in remaining:
            tree_nodes = np.flatnonzero(in_tree)
            allowed_parents = tree_nodes[mask[v, tree_nodes]]
            if allowed_parents.size == 0:
                still_waiting.append(v)
                continue
            u = int(rng.choice(allowed_parents))
            adj[u, v] = adj[v, u] = 1
            in_tree[v] = True
            progress = True
        remaining = still_waiting
    if remaining:
        return adj  # geometry infeasible

    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] == 0 and mask[i, j] and rng.random() < extra_prob:
                adj[i, j] = adj[j, i] = 1
    return adj


def is_connected(adj: np.ndarray) -> bool:
    """BFS connectivity check."""
    N = adj.shape[0]
    if N == 0:
        return True
    visited = np.zeros(N, dtype=bool)
    visited[0] = True
    queue: deque[int] = deque([0])
    while queue:
        u = queue.popleft()
        for v in np.flatnonzero(adj[u]):
            if not visited[v]:
                visited[v] = True
                queue.append(v)
    return bool(visited.all())


# ---------------------------------------------------------------------------
# Greedy improvement
# ---------------------------------------------------------------------------

def greedy_improve(
    adj: np.ndarray, mask: np.ndarray, N: int, num_attempts: int
) -> tuple[np.ndarray, float]:
    """Greedy edge-swap: remove a random edge (if still connected),
    add a random allowed non-edge, keep if score improves."""
    rng = np.random.default_rng()
    current_score = compute_score(adj, N)
    if current_score < 0:
        return adj, current_score

    for _ in range(num_attempts):
        edges = np.argwhere(np.triu(adj, k=1) == 1)
        if edges.shape[0] == 0:
            break
        e_idx = int(rng.integers(edges.shape[0]))
        i, j = int(edges[e_idx, 0]), int(edges[e_idx, 1])

        adj[i, j] = adj[j, i] = 0
        if not is_connected(adj):
            adj[i, j] = adj[j, i] = 1
            continue

        non_edges = np.argwhere(np.triu(mask, k=1) & (adj == 0))
        if non_edges.shape[0] == 0:
            adj[i, j] = adj[j, i] = 1
            continue
        ne_idx = int(rng.integers(non_edges.shape[0]))
        k, l = int(non_edges[ne_idx, 0]), int(non_edges[ne_idx, 1])
        adj[k, l] = adj[l, k] = 1

        new_score = compute_score(adj, N)
        if new_score > current_score:
            current_score = new_score
        else:
            adj[k, l] = adj[l, k] = 0
            adj[i, j] = adj[j, i] = 1

    return adj, current_score


# ---------------------------------------------------------------------------
# KNN baseline
# ---------------------------------------------------------------------------

def knn_graph(
    N: int, k: int, positions: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Build a k-nearest-neighbour graph on the sphere (symmetrised)."""
    cosines = positions @ positions.T
    np.fill_diagonal(cosines, -2.0)
    adj = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        neighbors = np.argsort(-cosines[i])
        count = 0
        for j in neighbors:
            if j != i and mask[i, j]:
                adj[i, j] = adj[j, i] = 1
                count += 1
                if count >= k:
                    break
    return adj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick demo: spectral graph design on the unit sphere"
    )
    parser.add_argument("--N", type=int, default=20, help="Number of hubs")
    parser.add_argument("--r_max", type=float, default=1.0, help="Max edge distance (radians)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    N, r_max, seed = args.N, args.r_max, args.seed
    rng = np.random.default_rng(seed)

    # 1. Sample hub positions
    for attempt in range(200):
        positions = sample_unit_sphere(N, rng)
        mask = compute_allowed_mask(positions, r_max)
        n_comp, _ = connected_components(
            csgraph=csr_matrix(mask.astype(np.uint8)), directed=False
        )
        if n_comp == 1:
            break
    else:
        print(f"ERROR: Could not find connected geometry for N={N}, r_max={r_max}")
        return

    n_allowed = int(np.triu(mask, k=1).sum())
    n_total = N * (N - 1) // 2

    print("=" * 60)
    print("  Spectral Graph Design -- Quick Start Demo")
    print("=" * 60)
    print(f"  N = {N} hubs on the unit sphere")
    print(f"  r_max = {r_max:.2f} rad ({math.degrees(r_max):.1f} deg)")
    print(f"  Allowed edges: {n_allowed}/{n_total} "
          f"({100 * n_allowed / n_total:.1f}%)")
    print()

    # 2. Random connected graph
    adj = random_connected_graph(N, mask, rng)
    initial_score = compute_score(adj, N)
    initial_edges = int(adj.sum()) // 2
    print(f"  Random graph:    score = {initial_score:.4f}  "
          f"(|E| = {initial_edges})")

    # 3. Greedy improvement
    adj_improved, improved_score = greedy_improve(
        adj.copy(), mask, N, num_attempts=2 * N
    )
    improved_edges = int(adj_improved.sum()) // 2
    print(f"  After greedy:    score = {improved_score:.4f}  "
          f"(|E| = {improved_edges})")

    # 4. KNN baselines
    for k in [3, 5]:
        knn_adj = knn_graph(N, k, positions, mask)
        knn_sc = compute_score(knn_adj, N)
        knn_e = int(knn_adj.sum()) // 2
        status = f"score = {knn_sc:.4f}  (|E| = {knn_e})" if knn_sc > 0 else "disconnected"
        print(f"  KNN k={k}:         {status}")

    # 5. Summary
    if improved_score > 0 and initial_score > 0:
        pct = 100 * (improved_score - initial_score) / initial_score
        print(f"\n  Greedy improvement: {pct:+.1f}% over random init")
    print()
    print("  Score = lambda_2(Laplacian) * N / |E|")
    print("  Higher = better spectral efficiency")
    print("=" * 60)


if __name__ == "__main__":
    main()
