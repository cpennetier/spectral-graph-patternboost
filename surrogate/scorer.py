"""GNN-based surrogate scorer for the logistics environment.

A lightweight Graph Neural Network that approximates the Fiedler-based
exact score computed by :meth:`LogisticsDataPoint.calc_score` -- that is,

                     lambda_2(L_G) * N
         score(G) = -------------------
                          |E|

where L_G = D - A is the combinatorial Laplacian of the undirected graph
G = (V, E) on N hubs and lambda_2(L_G) is its second smallest eigenvalue (the
Fiedler value / algebraic connectivity).

The surrogate is used as a **ranking filter**: given tens of thousands of
candidate graphs sampled from the transformer, the GNN predicts approximate
scores in milliseconds so that only the top-K candidates are forwarded to
the exact eigensolve. The correctness bar is therefore ranking -- Spearman
rank correlation and Recall@K -- not point-wise regression accuracy.

Main exports:
    GraphData: Dataclass carrying one graph's tensors.
    Batch: Packed batch of GraphData instances with a `batch_index`
        mapping each node to its graph.
    SurrogateGNN: The neural network (message passing + mean readout).
    SurrogateScorer: High-level API (train / predict / evaluate / IO).

Dependencies: pure PyTorch + numpy + scipy. No PyTorch Geometric.

Usage:
    from surrogate.scorer import SurrogateScorer
    scorer = SurrogateScorer(hidden_dim=64, n_layers=4)
    train_data = scorer.generate_training_data(N=30, r_max=0.8,
                                               num_samples=10_000, seed=42)
    val = train_data[8000:]
    log = scorer.train(train_data[:8000], val, epochs=200)
    scorer.save("models/surrogate.pt")
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_HIDDEN_DIM: int = 64
_DEFAULT_N_LAYERS: int = 4
_NODE_FEAT_DIM: int = 4            # (x, y, z, degree)
_EDGE_FEAT_DIM: int = 1            # (great-circle distance)
_GRAPH_FEAT_DIM: int = 2           # (|E|/max_E_allowed, density)
_DEFAULT_ALPHA: float = 0.1        # ranking-loss weight
_DEFAULT_MARGIN: float = 0.05      # ranking-loss margin
_DEFAULT_LR: float = 1e-3
_DEFAULT_WEIGHT_DECAY: float = 1e-4
_DEFAULT_BATCH_SIZE: int = 64


# ---------------------------------------------------------------------------
# Graph tensor containers
# ---------------------------------------------------------------------------

@dataclass
class GraphData:
    """Tensor representation of a single scored graph.

    Edges are stored **bidirectionally** (both (i,j) and (j,i)) so that
    message passing propagates information in both directions with a
    single scatter-add.

    Attributes:
        node_features: (N, 4) float32 -- columns (x, y, z, degree).
        edge_index: (2, 2|E|) int64 -- row 0 = source, row 1 = destination.
        edge_features: (2|E|, 1) float32 -- great-circle distance per edge.
        graph_features: (2,) float32 -- (|E|/max_E_allowed, density).
        target_score: float -- exact lambda_2 * N / |E|.
        num_nodes: N.
    """

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    graph_features: torch.Tensor
    target_score: float
    num_nodes: int


@dataclass
class Batch:
    """Packed batch of graphs.

    All per-node and per-edge tensors are concatenated along dim 0; the
    `batch_index` vector maps every node to its graph id so that the
    readout layer can mean-pool per graph with a single index_add_.

    Attributes:
        node_features: (sum(N_i), 4) float32.
        edge_index: (2, sum(2|E_i|)) int64 -- offsets already applied per graph.
        edge_features: (sum(2|E_i|), 1) float32.
        graph_features: (B, 2) float32.
        batch_index: (sum(N_i),) int64 -- graph id per node, values in [0, B).
        targets: (B,) float32 -- exact scores.
        num_graphs: B.
    """

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    graph_features: torch.Tensor
    batch_index: torch.Tensor
    targets: torch.Tensor
    num_graphs: int

    def to(self, device: torch.device) -> "Batch":
        """Move every tensor in the batch to `device`. Returns self."""
        self.node_features = self.node_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_features = self.edge_features.to(device)
        self.graph_features = self.graph_features.to(device)
        self.batch_index = self.batch_index.to(device)
        self.targets = self.targets.to(device)
        return self


def collate(graphs: list[GraphData]) -> Batch:
    """Pack a list of GraphData into a single Batch.

    Shifts every graph's edge_index by the cumulative node offset so that
    indices remain valid in the concatenated node tensor.

    Args:
        graphs: Non-empty list of GraphData.

    Returns:
        Batch with concatenated tensors.
    """
    assert len(graphs) > 0, "collate received empty graph list"
    node_features = []
    edge_index_parts = []
    edge_features = []
    graph_features = []
    batch_index_parts = []
    targets = []

    node_offset = 0
    for gi, g in enumerate(graphs):
        node_features.append(g.node_features)
        edge_index_parts.append(g.edge_index + node_offset)
        edge_features.append(g.edge_features)
        graph_features.append(g.graph_features)
        batch_index_parts.append(
            torch.full((g.num_nodes,), gi, dtype=torch.int64)
        )
        targets.append(g.target_score)
        node_offset += g.num_nodes

    return Batch(
        node_features=torch.cat(node_features, dim=0),
        edge_index=torch.cat(edge_index_parts, dim=1),
        edge_features=torch.cat(edge_features, dim=0),
        graph_features=torch.stack(graph_features, dim=0),
        batch_index=torch.cat(batch_index_parts, dim=0),
        targets=torch.tensor(targets, dtype=torch.float32),
        num_graphs=len(graphs),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MessagePassingLayer(nn.Module):
    """One round of edge-conditioned message passing.

    For each directed edge (u -> v) the message is

        m_{u->v} = W_n * x_u + W_e * e_{uv}

    and the update is

        x_v <- x_v + ReLU(LayerNorm(sum_{u in N(v)} m_{u->v})).

    Aggregation uses `torch.Tensor.index_add_` as a pure-PyTorch stand-in
    for torch_scatter's scatter_add -- correct even with repeated
    destination indices.
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int) -> None:
        super().__init__()
        self.node_lin = nn.Linear(hidden_dim, hidden_dim)
        self.edge_lin = nn.Linear(edge_feat_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate one round of messages.

        Args:
            x: (sum(N), H) current node hidden states.
            edge_index: (2, sum(E_dir)) sources (row 0) and destinations (row 1).
            edge_features: (sum(E_dir), F_e) per-edge features.

        Returns:
            (sum(N), H) updated node hidden states.
        """
        src = edge_index[0]
        dst = edge_index[1]
        messages = self.node_lin(x.index_select(0, src)) + self.edge_lin(edge_features)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)
        return x + F.relu(self.norm(agg))


class SurrogateGNN(nn.Module):
    """GCN-style surrogate for lambda_2(L) * N / |E|.

    Architecture:
        x_0 = Linear(node_feat_dim, H)(node_features)
        for k in 1..K:  x_k = MessagePassingLayer(x_{k-1})
        g  = mean_pool_per_graph(x_K)            (B, H)
        g  = concat(g, graph_features)           (B, H + F_g)
        y  = Linear(H, 1) o ReLU o Linear(H+F_g, H)(g)

    Parameter count (defaults H=64, K=4, node_feat_dim=4, edge_feat_dim=1,
    graph_feat_dim=2): ~36K parameters, well under the 100K budget.
    """

    def __init__(
        self,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        n_layers: int = _DEFAULT_N_LAYERS,
        node_feat_dim: int = _NODE_FEAT_DIM,
        edge_feat_dim: int = _EDGE_FEAT_DIM,
        graph_feat_dim: int = _GRAPH_FEAT_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.graph_feat_dim = graph_feat_dim

        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim, edge_feat_dim) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + graph_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Predict scores for every graph in `batch`.

        Args:
            batch: Packed Batch of B graphs with sum(N) total nodes.

        Returns:
            (B,) tensor of predicted scores.
        """
        x = self.node_encoder(batch.node_features)
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_features)

        # Mean pool per graph via index_add_ over batch_index.
        B = batch.num_graphs
        H = x.shape[1]
        summed = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        summed.index_add_(0, batch.batch_index, x)
        counts = torch.zeros(B, device=x.device, dtype=x.dtype)
        counts.index_add_(
            0, batch.batch_index, torch.ones_like(batch.batch_index, dtype=x.dtype)
        )
        pooled = summed / counts.unsqueeze(1).clamp(min=1.0)

        features = torch.cat([pooled, batch.graph_features], dim=1)
        return self.head(features).squeeze(-1)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def pairwise_margin_ranking_loss(
    predicted: torch.Tensor, target: torch.Tensor, margin: float
) -> torch.Tensor:
    """Pairwise margin ranking loss over all ordered pairs in a batch.

    For every pair (i, j) with target[i] > target[j] we penalise
        max(0, margin - (predicted[i] - predicted[j])).
    Pairs with ties (target[i] == target[j]) contribute zero. The mean is
    taken over pairs with strictly positive target difference; if no such
    pair exists the loss is zero.

    Args:
        predicted: (B,) predicted scores.
        target: (B,) exact scores.
        margin: Positive scalar margin.

    Returns:
        Scalar tensor.
    """
    diff_t = target.unsqueeze(1) - target.unsqueeze(0)        # (B, B)
    diff_p = predicted.unsqueeze(1) - predicted.unsqueeze(0)  # (B, B)
    mask = diff_t > 0
    if not mask.any():
        return predicted.new_zeros(())
    losses = torch.clamp(margin - diff_p, min=0.0)
    return losses[mask].mean()


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def _datapoint_to_graphdata(dp, positions: np.ndarray, allowed_mask: np.ndarray) -> GraphData:
    """Convert a scored LogisticsDataPoint into a GraphData.

    Edges are emitted in both directions. `max_E_allowed` is taken as the
    number of True entries in the upper triangle of ALLOWED_MASK; density
    is 2|E| / (N(N-1)).

    Args:
        dp: A LogisticsDataPoint with dp.data set and dp.score > 0.
        positions: (N, 3) hub positions on the unit sphere.
        allowed_mask: (N, N) boolean allowed-edge mask.

    Returns:
        GraphData.
    """
    N = dp.N
    adjacency = dp.data.astype(np.int64)
    degree = adjacency.sum(axis=1).astype(np.float32)
    node_features = np.concatenate(
        [positions.astype(np.float32), degree[:, None]], axis=1
    )  # (N, 4)

    # Undirected edges -> both directions for symmetric message passing.
    src_dst = np.argwhere(adjacency == 1)  # (2|E|, 2)
    if src_dst.shape[0] == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, 1), dtype=np.float32)
    else:
        edge_index = src_dst.T.astype(np.int64)  # (2, 2|E|)
        # Great-circle distance = arccos(clip(p_i * p_j, -1, 1)).
        dots = np.clip(
            (positions[edge_index[0]] * positions[edge_index[1]]).sum(axis=1),
            -1.0,
            1.0,
        )
        edge_features = np.arccos(dots).astype(np.float32)[:, None]

    num_edges = int(adjacency.sum()) // 2
    max_e_allowed = int(np.triu(allowed_mask, k=1).sum())
    max_e_allowed = max(max_e_allowed, 1)
    max_e_total = N * (N - 1) // 2
    graph_features = np.array(
        [num_edges / max_e_allowed, num_edges / max(max_e_total, 1)],
        dtype=np.float32,
    )

    return GraphData(
        node_features=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index),
        edge_features=torch.from_numpy(edge_features),
        graph_features=torch.from_numpy(graph_features),
        target_score=float(getattr(dp, "score", 0.0) or 0.0),
        num_nodes=N,
    )


def _build_logistics_env(N: int, r_max: float, seed: int):
    """Instantiate a LogisticsEnvironment with a minimal argparse Namespace.

    The environment constructor samples hub positions and allowed-edge mask
    and installs them as class attributes on LogisticsDataPoint, which is
    exactly what we need before instantiating scored datapoints.

    Args:
        N: Number of hubs.
        r_max: Distance budget in radians.
        seed: Seed for reproducible positions (>= 0 for determinism, -1 for
            fresh randomness).

    Returns:
        (env, positions, allowed_mask).
    """
    from src.envs.logistics import LogisticsDataPoint, LogisticsEnvironment

    ns = argparse.Namespace(
        N=N,
        r_max=r_max,
        encoding_tokens="single_integer",
        make_object_canonical=False,
        augment_data_representation=False,
        seed=seed,
    )
    env = LogisticsEnvironment(ns)
    positions = LogisticsDataPoint.POSITIONS.copy()
    allowed_mask = LogisticsDataPoint.ALLOWED_MASK.copy()
    return env, positions, allowed_mask


class SurrogateScorer:
    """High-level interface around SurrogateGNN.

    Wraps data generation, training, prediction, evaluation, and I/O. The
    scorer owns the torch model and remembers the (positions, mask) used
    at data-generation time so that `predict_batch` can re-featurise fresh
    datapoints consistently.
    """

    def __init__(
        self,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        n_layers: int = _DEFAULT_N_LAYERS,
        device: str = "cpu",
    ) -> None:
        """Create a fresh scorer with an untrained GNN.

        Args:
            hidden_dim: GNN hidden dimension.
            n_layers: Number of message passing layers.
            device: Torch device spec, e.g. "cpu", "mps", "cuda".
        """
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = torch.device(device)
        self.model = SurrogateGNN(hidden_dim=hidden_dim, n_layers=n_layers).to(
            self.device
        )
        self._positions: Optional[np.ndarray] = None
        self._allowed_mask: Optional[np.ndarray] = None
        self._N: Optional[int] = None
        self._r_max: Optional[float] = None

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_training_data(
        self,
        N: int,
        r_max: float,
        num_samples: int,
        seed: int = 42,
        progress_every: int = 500,
    ) -> list[GraphData]:
        """Generate `num_samples` scored graphs for training.

        Each sample is a random connected LogisticsDataPoint run through
        `local_search(improve_with_local_search=True)`. Datapoints with
        score <= 0 (disconnected / geometry-infeasible) are dropped. The
        expected yield is ~80-90% of the requested count.

        Args:
            N: Number of hubs.
            r_max: Distance budget in radians.
            num_samples: Number of raw samples to attempt.
            seed: Seed for reproducible hub positions and Python RNG.
            progress_every: Print a progress line every `progress_every`
                attempted samples. Set to 0 to disable.

        Returns:
            List of GraphData with score > 0.
        """
        from src.envs.logistics import LogisticsDataPoint

        _, positions, allowed_mask = _build_logistics_env(N, r_max, seed)
        self._positions = positions
        self._allowed_mask = allowed_mask
        self._N = N
        self._r_max = r_max

        np.random.seed(seed)

        graphs: list[GraphData] = []
        t0 = time.time()
        for i in range(num_samples):
            dp = LogisticsDataPoint(N=N, init=True)
            if dp.score < 0:
                continue
            dp.local_search(improve_with_local_search=True)
            if dp.score <= 0:
                continue
            graphs.append(_datapoint_to_graphdata(dp, positions, allowed_mask))
            if progress_every and (i + 1) % progress_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  [gen] {i + 1}/{num_samples} attempted, "
                    f"{len(graphs)} valid, {elapsed:.1f}s elapsed"
                )
        return graphs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: list[GraphData],
        val_data: list[GraphData],
        epochs: int = 200,
        lr: float = _DEFAULT_LR,
        weight_decay: float = _DEFAULT_WEIGHT_DECAY,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        alpha: float = _DEFAULT_ALPHA,
        margin: float = _DEFAULT_MARGIN,
        verbose: bool = True,
        lr_schedule: str = "none",
    ) -> dict[str, list[float]]:
        """Train the GNN with MSE + alpha * ranking loss.

        Args:
            train_data: Training set.
            val_data: Validation set (used for per-epoch metrics).
            epochs: Number of epochs.
            lr: Adam learning rate.
            weight_decay: Adam weight decay.
            batch_size: Number of graphs per batch.
            alpha: Weight on the pairwise ranking loss.
            margin: Margin for the ranking loss.
            verbose: Whether to print per-epoch metrics.
            lr_schedule: Learning-rate schedule. One of:
                "none" -- constant lr;
                "step" -- x0.1 at epoch `epochs // 2`;
                "cosine" -- torch.optim.lr_scheduler.CosineAnnealingLR
                over the full run (lr -> 0 at the last epoch).

        Returns:
            Dict with keys "train_loss", "val_mse", "val_spearman", "lr",
            each a list of length `epochs`.
        """
        assert len(train_data) > 0, "train_data is empty"
        assert len(val_data) > 0, "val_data is empty"

        optim = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
        if lr_schedule == "none":
            scheduler = None
        elif lr_schedule == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optim, step_size=max(1, epochs // 2), gamma=0.1
            )
        elif lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=epochs, eta_min=lr * 1e-2
            )
        else:
            raise ValueError(
                f"Unknown lr_schedule={lr_schedule!r}; "
                f"expected 'none', 'step', or 'cosine'."
            )

        log: dict[str, list[float]] = {
            "train_loss": [],
            "val_mse": [],
            "val_spearman": [],
            "lr": [],
        }

        rng = np.random.default_rng(0)
        n_train = len(train_data)

        for epoch in range(epochs):
            self.model.train()
            perm = rng.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                if len(idx) < 2:
                    continue  # ranking loss needs >= 2
                batch = collate([train_data[int(j)] for j in idx]).to(self.device)
                pred = self.model(batch)
                mse = F.mse_loss(pred, batch.targets)
                rank = pairwise_margin_ranking_loss(pred, batch.targets, margin)
                loss = mse + alpha * rank

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            val_mse, val_rho = self._val_metrics(val_data, batch_size)
            current_lr = optim.param_groups[0]["lr"]

            log["train_loss"].append(train_loss)
            log["val_mse"].append(val_mse)
            log["val_spearman"].append(val_rho)
            log["lr"].append(current_lr)

            if scheduler is not None:
                scheduler.step()

            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
                print(
                    f"epoch {epoch + 1:4d}/{epochs} | "
                    f"lr={current_lr:.2e} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_mse={val_mse:.4f} | "
                    f"val_spearman={val_rho:.4f}"
                )

        return log

    def _val_metrics(
        self, val_data: list[GraphData], batch_size: int
    ) -> tuple[float, float]:
        """Compute (MSE, Spearman rho) on the validation set."""
        preds, targets = self._predict_on_graphdata(val_data, batch_size)
        mse = float(np.mean((preds - targets) ** 2))
        if np.std(preds) == 0 or np.std(targets) == 0:
            rho = 0.0
        else:
            rho_res = spearmanr(preds, targets)
            rho = float(rho_res.correlation) if not math.isnan(rho_res.correlation) else 0.0
        return mse, rho

    def _predict_on_graphdata(
        self, graphs: list[GraphData], batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on a list of GraphData. Returns (preds, targets)."""
        self.model.eval()
        preds = np.empty(len(graphs), dtype=np.float32)
        targets = np.empty(len(graphs), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(graphs), batch_size):
                chunk = graphs[start : start + batch_size]
                batch = collate(chunk).to(self.device)
                out = self.model(batch).cpu().numpy()
                preds[start : start + len(chunk)] = out
                targets[start : start + len(chunk)] = batch.targets.cpu().numpy()
        return preds, targets

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        datapoints: list,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        """Predict scores for a list of LogisticsDataPoints.

        Re-uses the scorer's cached (positions, allowed_mask). You must
        call `generate_training_data` first (or `load` a trained model
        whose metadata includes those arrays) so the featuriser has the
        geometry it needs.

        Args:
            datapoints: List of LogisticsDataPoint with `.data` set.
            batch_size: Inference batch size.

        Returns:
            (len(datapoints),) float32 array of predicted scores.
        """
        if self._positions is None or self._allowed_mask is None:
            raise RuntimeError(
                "SurrogateScorer has no cached geometry. Call "
                "generate_training_data(...) or load(...) first."
            )
        graphs = [
            _datapoint_to_graphdata(dp, self._positions, self._allowed_mask)
            for dp in datapoints
        ]
        preds, _ = self._predict_on_graphdata(graphs, batch_size)
        return preds

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        val_data: list[GraphData],
        ks: tuple[int, ...] = (100, 500, 1000),
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> dict[str, float]:
        """Compute MSE, Spearman rho, and Recall@K on `val_data`.

        Recall@K is defined as:
            |top_K(predicted) intersect top_K(exact)| / K.
        K values that exceed |val_data| are skipped.

        Args:
            val_data: List of GraphData with target_score set.
            ks: Tuple of K values.
            batch_size: Inference batch size.

        Returns:
            Dict with keys "mse", "spearman", and f"recall@{k}" for each k.
        """
        preds, targets = self._predict_on_graphdata(val_data, batch_size)
        mse = float(np.mean((preds - targets) ** 2))
        rho_res = spearmanr(preds, targets)
        rho = float(rho_res.correlation) if not math.isnan(rho_res.correlation) else 0.0
        out: dict[str, float] = {"mse": mse, "spearman": rho}
        for k in ks:
            if k > len(val_data):
                continue
            top_pred = set(np.argsort(-preds)[:k].tolist())
            top_exact = set(np.argsort(-targets)[:k].tolist())
            out[f"recall@{k}"] = len(top_pred & top_exact) / k
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights + geometry + config to `path` (torch.save).

        Args:
            path: Destination file path.
        """
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "hidden_dim": self.hidden_dim,
                    "n_layers": self.n_layers,
                },
                "geometry": {
                    "N": self._N,
                    "r_max": self._r_max,
                    "positions": self._positions,
                    "allowed_mask": self._allowed_mask,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load weights + geometry + config from `path`.

        Rebuilds the model if the stored hidden_dim / n_layers differ from
        the current instance.

        Args:
            path: Source file path.
        """
        blob = torch.load(path, map_location=self.device, weights_only=False)
        cfg = blob["config"]
        if cfg["hidden_dim"] != self.hidden_dim or cfg["n_layers"] != self.n_layers:
            self.hidden_dim = cfg["hidden_dim"]
            self.n_layers = cfg["n_layers"]
            self.model = SurrogateGNN(
                hidden_dim=self.hidden_dim, n_layers=self.n_layers
            ).to(self.device)
        self.model.load_state_dict(blob["state_dict"])
        geom = blob.get("geometry", {})
        self._N = geom.get("N")
        self._r_max = geom.get("r_max")
        self._positions = geom.get("positions")
        self._allowed_mask = geom.get("allowed_mask")
