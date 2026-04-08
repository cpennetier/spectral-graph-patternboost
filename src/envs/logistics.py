"""Logistics network design environment.

Geometric spectral graph design: given N hubs at fixed positions on the unit
sphere, find the edge set that maximises algebraic connectivity per edge
subject to a geographic distance constraint.

Object: undirected graph G = (V, E) with V = {0, ..., N-1}, encoded as a
symmetric N×N uint8 adjacency matrix.

Score:
                 λ₂(L) · N
    score(G) = -------------
                    |E|

where L = D - A is the combinatorial Laplacian and λ₂ is its second
smallest eigenvalue (the Fiedler value). Hard constraints:
    - G must be connected (λ₂ > 0)
    - |E| ≥ 1
    - every edge (i, j) must satisfy d_sphere(p_i, p_j) ≤ R_MAX

Main exports:
    LogisticsDataPoint: DataPoint subclass implementing the problem.
    LogisticsEnvironment: Environment wrapper that samples hub positions
        and the allowed-edge mask, and builds the tokenizer.

Usage:
    python train.py --env_name logistics --N 30 --r_max 0.8 ...
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import (
    SparseTokenizerSequenceKTokens,
    SparseTokenizerSingleInteger,
)
from src.envs.utils import random_symmetry_adj_matrix, sort_graph_based_on_degree
from src.utils import bool_flag

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# For integer-valued adjacency matrices with N ≲ 1000, numpy.linalg.eigvalsh
# returns eigenvalues accurate to ~1e-13 relative to the spectral radius.
# 1e-9 gives a generous margin to declare an eigenvalue "effectively zero"
# while still rejecting every numerically disconnected graph. Used as a
# sanity guard — the primary connectivity test is scipy.sparse.csgraph.
_CONNECTIVITY_EIG_TOL: float = 1e-9


# ---------------------------------------------------------------------------
# Pure helpers (module-level, no class state)
# ---------------------------------------------------------------------------

def _sphere_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Great-circle distance between two points on the unit sphere.

    d(p, q) = arccos(clip(p · q, -1, 1)). The clip guards against
    floating-point overshoot (|dot| slightly > 1) that would return NaN.

    Args:
        p: 3-vector on the unit sphere.
        q: 3-vector on the unit sphere.

    Returns:
        Distance in radians, in [0, π].
    """
    return float(np.arccos(np.clip(np.dot(p, q), -1.0, 1.0)))


def _compute_allowed_mask(positions: np.ndarray, r_max: float) -> np.ndarray:
    """Precompute which pairs of hubs are within the distance budget.

    Args:
        positions: (N, 3) float64 array of unit vectors.
        r_max: Maximum allowed great-circle distance in radians.

    Returns:
        (N, N) boolean array. Entry (i, j) is True iff i ≠ j and
        d_sphere(positions[i], positions[j]) ≤ r_max. Symmetric with
        False on the diagonal.
    """
    cosines = np.clip(positions @ positions.T, -1.0, 1.0)
    distances = np.arccos(cosines)
    mask = distances <= r_max
    np.fill_diagonal(mask, False)
    return mask


_MAX_GEOMETRY_RESAMPLES: int = 200


def _sample_connected_geometry(
    n: int, r_max: float, rng: np.random.Generator
) -> "tuple[np.ndarray, np.ndarray]":
    """Sample positions whose allowed-edge graph G' is connected.

    Retries up to _MAX_GEOMETRY_RESAMPLES times; raises on failure with
    a diagnostic that tells the user how to widen the constraint.

    Args:
        n: Number of hubs.
        r_max: Maximum allowed great-circle distance (radians).
        rng: NumPy generator.

    Returns:
        (positions, allowed_mask): positions is (n, 3) float64, mask is
        (n, n) bool.

    Raises:
        ValueError: If no connected geometry was found within the retry
            budget.
    """
    for _ in range(_MAX_GEOMETRY_RESAMPLES):
        positions = _sample_unit_sphere(n, rng)
        allowed = _compute_allowed_mask(positions, r_max)
        n_comp, _ = connected_components(
            csgraph=csr_matrix(allowed.astype(np.uint8)), directed=False
        )
        if n_comp == 1:
            return positions, allowed
    raise ValueError(
        f"Could not sample a connected allowed-edge geometry for "
        f"N={n}, R_MAX={r_max} after {_MAX_GEOMETRY_RESAMPLES} attempts. "
        f"The geometric random graph on S^2 has edge probability "
        f"p=(1-cos(R_MAX))/2 and is connected above E[deg] > ln(N). "
        f"Increase --r_max or --N."
    )


def _sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points uniformly on the unit sphere in R^3.

    Uses the cylindrical-projection construction: sample z ∼ U(-1, 1) and
    θ ∼ U(0, 2π), then (x, y, z) = (√(1-z²)·cos θ, √(1-z²)·sin θ, z).
    This is exactly uniform on the sphere because the area of a spherical
    zone {z ∈ [a, b]} is 2π(b - a), linear in z (Archimedes' hat-box).

    Args:
        n: Number of points to sample.
        rng: NumPy random generator.

    Returns:
        (n, 3) float64 array of unit vectors.
    """
    z = rng.uniform(-1.0, 1.0, size=n)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    r = np.sqrt(1.0 - z * z)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# DataPoint
# ---------------------------------------------------------------------------

class LogisticsDataPoint(DataPoint):
    """Undirected graph on N hubs on the unit sphere with spectral score.

    Class attributes (populated by LogisticsEnvironment before any DataPoint
    is instantiated — including in worker processes via the _*_class_params
    plumbing described in src/envs/environment.py):
        R_MAX: Maximum allowed great-circle distance per edge (radians).
        POSITIONS: (N, 3) float64 array of hub positions on the unit sphere.
        ALLOWED_MASK: (N, N) bool array; True where an edge is permitted.
        MAKE_OBJECT_CANONICAL: Whether to sort nodes by degree before
            tokenisation. NOTE: degree-sorting permutes node indices and
            therefore breaks the correspondence with POSITIONS. For
            logistics this means the distance constraint is no longer
            interpretable post-canonicalisation. Leave False unless you
            know what you are doing.

    Instance attributes:
        N: Number of nodes.
        data: (N, N) uint8 symmetric adjacency matrix.
        score: λ₂(L) · N / |E| for valid instances, −1 otherwise.
        features: Canonical comma-separated upper-triangle string.
    """

    R_MAX: float = 0.8
    POSITIONS: Optional[np.ndarray] = None
    ALLOWED_MASK: Optional[np.ndarray] = None
    MAKE_OBJECT_CANONICAL: bool = False

    # Probability of including each non-tree allowed edge in random
    # generation. 0.3 yields sparse but not-minimal initial graphs
    # (|E| ≈ 0.3·(allowed non-tree edges) + (N-1)).
    _RANDOM_EXTRA_EDGE_PROB: float = 0.3

    def __init__(self, N: int, init: bool = False) -> None:
        """Build an empty graph, optionally populating it with a random instance.

        Args:
            N: Number of nodes.
            init: If True, generate a random connected graph respecting the
                distance constraint, then call calc_features() and
                calc_score(). If False (the default), leave self.data zeroed
                — used by tokenizer.decode() which fills the matrix
                externally.

        Raises:
            RuntimeError: If init=True but the class-level POSITIONS /
                ALLOWED_MASK have not been initialised by the Environment.
        """
        super().__init__()
        self.N: int = N
        self.data: np.ndarray = np.zeros((self.N, self.N), dtype=np.uint8)

        if init:
            if (
                LogisticsDataPoint.POSITIONS is None
                or LogisticsDataPoint.ALLOWED_MASK is None
            ):
                raise RuntimeError(
                    "LogisticsDataPoint.POSITIONS / ALLOWED_MASK must be set "
                    "by LogisticsEnvironment before init=True instances are "
                    "created."
                )
            connected = self._random_connected_graph()
            if not connected:
                # Geometry is infeasible under R_MAX — leave score=-1 so the
                # dataset pipeline drops this instance.
                self.score = -1
                self.features = ""
                return
            if self.MAKE_OBJECT_CANONICAL:
                self.data = sort_graph_based_on_degree(self.data)
            self.calc_features()
            self.calc_score()

    # ---------------------------------------------------------------- score

    def calc_score(self) -> None:
        """Set self.score to the algebraic-connectivity-per-edge ratio.

        Computes L = D − A and its eigenvalues via numpy.linalg.eigvalsh
        (ascending order). The Fiedler value λ₂ = eigvalsh(L)[1] is zero
        iff G is disconnected, and is the spectral witness of graph
        robustness via Cheeger's inequality:

            λ₂ / 2  ≤  h(G)  ≤  √(2 λ₂)

        where h(G) = min_{S⊂V, 0<|S|≤|V|/2} |∂S| / |S| is the edge-
        expansion (Cheeger) constant. A larger λ₂ forces a larger h(G),
        meaning every non-trivial cut of G must sever a large fraction
        of its vertices worth of edges — a direct measure of topological
        resilience. Normalising by |E| expresses the trade-off against
        infrastructure cost: we reward graphs that achieve strong cut
        resistance with few lanes.

        Invalid cases (set score = −1):
            - N < 2
            - |E| == 0
            - any edge (i, j) violates d_sphere(p_i, p_j) ≤ R_MAX
            - G is disconnected (exact check via scipy.sparse.csgraph)

        Returns:
            None. Mutates self.score.
        """
        if self.N < 2:
            self.score = -1
            return

        num_edges = int(self.data.sum()) // 2
        if num_edges == 0:
            self.score = -1
            return

        # Hard distance constraint: every edge in G must lie within the
        # R_MAX budget. By analogy with cycle.py's forbidden-4-cycle check
        # this belongs in calc_score — otherwise the model could learn to
        # exploit edges it is not allowed to build.
        mask = LogisticsDataPoint.ALLOWED_MASK
        if mask is not None:
            illegal = (self.data > 0) & (~mask)
            if illegal.any():
                self.score = -1
                return

        # Exact integer-BFS connectivity check. We run this BEFORE the
        # eigensolve so we never divide by a spuriously-small λ₂ of a
        # disconnected graph.
        n_components, _ = connected_components(
            csgraph=csr_matrix(self.data), directed=False, return_labels=True
        )
        if n_components > 1:
            self.score = -1
            return

        adjacency = self.data.astype(np.float64)
        degree = adjacency.sum(axis=1)
        laplacian = np.diag(degree) - adjacency
        # eigvalsh exploits symmetry: O(N^3), numerically stable, ascending.
        eigenvalues = np.linalg.eigvalsh(laplacian)
        lambda_2 = float(eigenvalues[1])

        # Numerical guard: connected_components already ruled out
        # disconnection, but a tiny or negative eigenvalue here would flag
        # a pathology (e.g. an adjacency matrix with unexpected content).
        if lambda_2 <= _CONNECTIVITY_EIG_TOL:
            self.score = -1
            return

        self.score = lambda_2 * self.N / num_edges

    # ------------------------------------------------------------- features

    def calc_features(self) -> None:
        """Set self.features to the comma-joined upper-triangle of self.data.

        Used only for deduplication in the dataset pipeline: two graphs
        with identical upper-triangles produce identical feature strings
        and are treated as duplicates.

        Returns:
            None. Mutates self.features.
        """
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w.append(self.data[i, j])
        self.features = ",".join(map(str, w))

    # -------------------------------------------------------- local search

    def local_search(self, improve_with_local_search: bool) -> None:
        """Repair disconnectedness, then optionally improve the score.

        Phase 1 — repair (always). While the graph has more than one
        component, add the *shortest* allowed cross-component edge. Each
        such addition strictly decreases the component count. If the
        geometry does not admit a spanning tree (some component has no
        allowed edge to any other), the loop exits with G still
        disconnected and calc_score will return −1.

        Phase 2 — improve (only if improve_with_local_search=True). For
        up to 2N attempts:
            - pick a random existing edge (i, j);
            - tentatively remove it; if the graph is now disconnected,
              restore and continue;
            - pick a random allowed non-edge (k, l);
            - tentatively add it and re-score;
            - commit the swap only if the new score is strictly better,
              otherwise roll back both changes.
        The swap criterion `new > current` is what makes this phase
        monotone non-decreasing on valid inputs.

        Args:
            improve_with_local_search: Whether to run phase 2.

        Returns:
            None. Mutates self.data, self.features, self.score.
        """
        self._repair_to_connected()
        if improve_with_local_search:
            self._improve_by_edge_swap()
        if self.MAKE_OBJECT_CANONICAL:
            self.data = sort_graph_based_on_degree(self.data)
        self.calc_features()
        self.calc_score()

    # -------------------------------------------- class params (pickling)

    @classmethod
    def _update_class_params(
        cls, pars: "tuple[float, np.ndarray, np.ndarray, bool]"
    ) -> None:
        """Restore class-level state in a worker process.

        Counterpart of _save_class_params; invoked by
        DataPoint._batch_generate_and_score (and by do_score, decode_batch)
        before any instance is created in the worker. Each worker boots a
        fresh interpreter and re-imports the module, so without this
        restore the class variables would sit at their module-defined
        defaults (R_MAX=0.8, POSITIONS=None, ALLOWED_MASK=None).

        Args:
            pars: 4-tuple (R_MAX, POSITIONS, ALLOWED_MASK,
                MAKE_OBJECT_CANONICAL).
        """
        cls.R_MAX, cls.POSITIONS, cls.ALLOWED_MASK, cls.MAKE_OBJECT_CANONICAL = pars

    @classmethod
    def _save_class_params(
        cls,
    ) -> "tuple[float, Optional[np.ndarray], Optional[np.ndarray], bool]":
        """Snapshot class-level state for pickling to worker processes.

        Tuples containing numpy arrays are picklable by default; the
        serialisation cost is dominated by ALLOWED_MASK at one byte per
        cell (N² bytes) and by POSITIONS at 24·N bytes.

        Returns:
            (R_MAX, POSITIONS, ALLOWED_MASK, MAKE_OBJECT_CANONICAL).
        """
        return (cls.R_MAX, cls.POSITIONS, cls.ALLOWED_MASK, cls.MAKE_OBJECT_CANONICAL)

    # ----------------------------------------------------------- internals

    def _random_connected_graph(self) -> bool:
        """Build a random connected graph respecting ALLOWED_MASK.

        Two phases:
            1. Random spanning tree. Iterate over the non-root nodes in
               random order, attaching each to a uniformly random already-
               in-tree neighbour permitted by ALLOWED_MASK. Nodes with no
               allowed tree neighbour on a given pass are deferred; the
               loop exits when either every node is in the tree (success)
               or a full pass makes no progress (failure — the allowed-
               edge graph on V is disconnected under R_MAX).
            2. Extra edges. For every non-tree allowed pair, add it with
               probability _RANDOM_EXTRA_EDGE_PROB.

        Side effects:
            On success, self.data holds the adjacency. On failure,
            self.data is zeroed.

        Returns:
            True on success; False if no spanning tree exists.
        """
        N = self.N
        mask = LogisticsDataPoint.ALLOWED_MASK
        assert mask is not None and mask.shape == (N, N)

        rng = np.random.default_rng()

        # Phase 1 — random spanning tree via multi-pass random Prim.
        in_tree = np.zeros(N, dtype=bool)
        in_tree[0] = True
        remaining = [int(v) for v in rng.permutation(np.arange(1, N))]
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
                self.data[u, v] = 1
                self.data[v, u] = 1
                in_tree[v] = True
                progress = True
            remaining = still_waiting
        if remaining:
            # Allowed-edge graph is disconnected; caller sets score=-1.
            self.data.fill(0)
            return False

        # Phase 2 — sprinkle non-tree allowed edges.
        for i in range(N):
            for j in range(i + 1, N):
                if self.data[i, j] == 1 or not mask[i, j]:
                    continue
                if rng.random() < self._RANDOM_EXTRA_EDGE_PROB:
                    self.data[i, j] = 1
                    self.data[j, i] = 1
        return True

    def _repair_to_connected(self) -> None:
        """Add shortest allowed cross-component edges until G is connected.

        Vectorised cross-component search: for each iteration, build a
        boolean (N, N) mask of allowed inter-component pairs, then pick
        the shortest such pair by maximising the cosine similarity (which
        is a monotone decreasing transform of arccos-distance, so we avoid
        the transcendental call entirely).

        Side effects:
            Mutates self.data. When no cross-component allowed edge
            exists, leaves G disconnected.
        """
        mask = LogisticsDataPoint.ALLOWED_MASK
        positions = LogisticsDataPoint.POSITIONS
        assert mask is not None and positions is not None

        cosines = positions @ positions.T
        while True:
            n_comp, labels = connected_components(
                csgraph=csr_matrix(self.data), directed=False, return_labels=True
            )
            if n_comp <= 1:
                return
            cross = labels[:, None] != labels[None, :]
            candidates = cross & mask
            if not candidates.any():
                return  # geometry precludes a spanning tree
            cosines_masked = np.where(candidates, cosines, -np.inf)
            flat_idx = int(np.argmax(cosines_masked))
            i, j = divmod(flat_idx, self.N)
            self.data[i, j] = 1
            self.data[j, i] = 1

    def _improve_by_edge_swap(self) -> None:
        """Greedy edge-swap to raise score while preserving connectivity.

        For up to 2N attempts: drop a random edge, verify the graph is
        still connected (else restore and continue), then add a random
        allowed non-edge. Commit iff the new score strictly exceeds the
        current one.
        """
        mask = LogisticsDataPoint.ALLOWED_MASK
        assert mask is not None

        rng = np.random.default_rng()
        self.calc_score()
        current = self.score
        if current < 0:
            return

        num_attempts = max(1, 2 * self.N)
        for _ in range(num_attempts):
            edges = np.argwhere(np.triu(self.data, k=1) == 1)
            if edges.shape[0] == 0:
                return
            e_idx = int(rng.integers(edges.shape[0]))
            i, j = int(edges[e_idx, 0]), int(edges[e_idx, 1])

            # Tentative removal.
            self.data[i, j] = 0
            self.data[j, i] = 0
            if not self._is_connected():
                self.data[i, j] = 1
                self.data[j, i] = 1
                continue

            # Pick a random allowed non-edge (excluding the one we just dropped).
            non_edges = np.argwhere(np.triu(mask, k=1) & (self.data == 0))
            if non_edges.shape[0] == 0:
                self.data[i, j] = 1
                self.data[j, i] = 1
                continue
            ne_idx = int(rng.integers(non_edges.shape[0]))
            k, l = int(non_edges[ne_idx, 0]), int(non_edges[ne_idx, 1])

            self.data[k, l] = 1
            self.data[l, k] = 1

            # Score the candidate (graph post-swap is connected: we removed
            # an edge and verified connectivity, then only added an edge).
            self.calc_score()
            if self.score > current:
                current = self.score
            else:
                # Roll back both edits.
                self.data[k, l] = 0
                self.data[l, k] = 0
                self.data[i, j] = 1
                self.data[j, i] = 1
                self.score = current

    def _is_connected(self) -> bool:
        """Return True iff self.data is a single connected component.

        BFS from node 0 using a deque — O(N + |E|).
        """
        if self.N == 0:
            return True
        visited = np.zeros(self.N, dtype=bool)
        visited[0] = True
        queue: "deque[int]" = deque([0])
        while queue:
            u = queue.popleft()
            for v in np.flatnonzero(self.data[u]):
                iv = int(v)
                if not visited[iv]:
                    visited[iv] = True
                    queue.append(iv)
        return bool(visited.all())


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class LogisticsEnvironment(BaseEnvironment):
    """Environment wrapper for the logistics / spectral graph design problem.

    Responsibilities:
        - Sample N hub positions uniformly on the unit sphere.
        - Precompute ALLOWED_MASK from R_MAX.
        - Install R_MAX, POSITIONS, ALLOWED_MASK, MAKE_OBJECT_CANONICAL as
          class attributes on LogisticsDataPoint so every instance
          (including those created in worker processes) sees consistent
          geometry.
        - Build the tokenizer.
    """

    k: int = 2
    are_coordinates_symmetric: bool = True
    data_class = LogisticsDataPoint

    def __init__(self, params: argparse.Namespace) -> None:
        """Initialise positions, allowed-mask, and tokenizer.

        Args:
            params: argparse namespace. Required attributes: N, r_max,
                encoding_tokens, make_object_canonical,
                augment_data_representation. Optional: seed (reproducible
                positions when seed ≥ 0).

        Raises:
            ValueError: If params.encoding_tokens is neither
                "single_integer" nor "sequence_k_tokens".
        """
        super().__init__(params)

        seed = getattr(params, "seed", -1)
        rng_seed: Optional[int] = seed if seed is not None and seed >= 0 else None
        rng = np.random.default_rng(rng_seed)

        # The random-Prim spanning-tree builder requires the allowed-edge
        # graph G' = (V, {(i,j) : d(p_i,p_j) ≤ R_MAX}) to be connected;
        # otherwise every DataPoint(init=True) fails. For a random
        # geometric graph on S^2 with edge probability
        # p = (1-cos(R_MAX))/2, this happens above the expected-degree
        # threshold E[deg] > ln(N). For tight (N, R_MAX) we resample until
        # connectivity holds, then bail out with a clear diagnostic.
        positions, allowed = _sample_connected_geometry(
            params.N, float(params.r_max), rng
        )

        self.data_class.R_MAX = float(params.r_max)
        self.data_class.POSITIONS = positions
        self.data_class.ALLOWED_MASK = allowed
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical

        encoding_augmentation = (
            random_symmetry_adj_matrix if params.augment_data_representation else None
        )
        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizerSingleInteger(
                self.data_class,
                params.N,
                self.k,
                self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS,
                encoding_augmentation=encoding_augmentation,
            )
        elif params.encoding_tokens == "sequence_k_tokens":
            self.tokenizer = SparseTokenizerSequenceKTokens(
                self.data_class,
                params.N,
                self.k,
                self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS,
                encoding_augmentation=encoding_augmentation,
            )
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")

    @staticmethod
    def register_args(parser: argparse.ArgumentParser) -> None:
        """Register CLI arguments specific to the logistics environment.

        Args:
            parser: An argparse parser (already instantiated by train.py).
        """
        parser.add_argument(
            "--N", type=int, default=30, help="Number of hubs"
        )
        parser.add_argument(
            "--r_max",
            type=float,
            default=0.8,
            help="Maximum allowed great-circle distance per edge (radians)",
        )
        parser.add_argument(
            "--encoding_tokens",
            type=str,
            default="single_integer",
            help="single_integer/sequence_k_tokens",
        )
        parser.add_argument(
            "--make_object_canonical",
            type=bool_flag,
            default="false",
            help="sort graph nodes by degree (breaks position correspondence)",
        )
        parser.add_argument(
            "--augment_data_representation",
            type=bool_flag,
            default="false",
            help="apply a random node permutation at encoding time",
        )
