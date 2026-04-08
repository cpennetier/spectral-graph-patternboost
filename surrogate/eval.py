"""CLI: evaluate a trained surrogate GNN on fresh logistics graphs.

Reports MSE, Spearman rho, Recall@{100,500,1000}, and speedup factor
(exact-eigensolve wall time vs GNN forward-pass wall time).

Usage:
    PYTHONPATH=. python surrogate/eval.py --model models/surrogate.pt \
        --N 30 --r_max 0.8 --num_test 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Project root on sys.path so `import src.*` works when run as a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from surrogate.scorer import SurrogateScorer, _datapoint_to_graphdata  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the surrogate GNN.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--r_max", type=float, default=0.8)
    p.add_argument("--num_test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--figure_path",
        type=str,
        default="figures/surrogate_scatter.png",
    )
    return p.parse_args()


def _time_exact_scoring(num_graphs: int, N: int) -> float:
    """Time `num_graphs` calls to numpy.linalg.eigvalsh on random NxN
    Laplacians, as a proxy for the cost the surrogate replaces.

    Returns:
        Wall-clock time in seconds for the whole loop.
    """
    rng = np.random.default_rng(0)
    t0 = time.time()
    for _ in range(num_graphs):
        a = (rng.random((N, N)) < 0.3).astype(np.float64)
        a = np.triu(a, k=1)
        a = a + a.T
        d = a.sum(axis=1)
        L = np.diag(d) - a
        _ = np.linalg.eigvalsh(L)
    return time.time() - t0


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[eval_surrogate] loading {args.model}")
    scorer = SurrogateScorer(device=args.device)
    scorer.load(args.model)
    print(f"[eval_surrogate] GNN parameters: {scorer.model.count_parameters():,}")

    from src.envs.logistics import LogisticsDataPoint

    if scorer._positions is None:
        raise RuntimeError("loaded model has no geometry cached")
    LogisticsDataPoint.POSITIONS = scorer._positions
    LogisticsDataPoint.ALLOWED_MASK = scorer._allowed_mask
    LogisticsDataPoint.R_MAX = scorer._r_max
    LogisticsDataPoint.MAKE_OBJECT_CANONICAL = False

    # Generate fresh test data (use a different seed than training).
    print(f"[eval_surrogate] generating {args.num_test} test graphs")
    t0 = time.time()

    test_graphs = []
    n_attempts = 0
    while len(test_graphs) < args.num_test and n_attempts < args.num_test * 3:
        dp = LogisticsDataPoint(N=args.N, init=True)
        n_attempts += 1
        if dp.score < 0:
            continue
        dp.local_search(improve_with_local_search=True)
        if dp.score <= 0:
            continue
        test_graphs.append(
            _datapoint_to_graphdata(dp, scorer._positions, scorer._allowed_mask)
        )
    gen_time = time.time() - t0
    print(
        f"[eval_surrogate] generated {len(test_graphs)} valid graphs in "
        f"{gen_time:.1f}s"
    )

    # Metrics.
    ks = tuple(k for k in (100, 500, 1000) if k <= len(test_graphs))
    metrics = scorer.evaluate(test_graphs, ks=ks)

    # Timing: wall time for surrogate vs wall time for exact eigensolves.
    t0 = time.time()
    preds, targets = scorer._predict_on_graphdata(test_graphs)
    surrogate_time = time.time() - t0
    exact_time = _time_exact_scoring(len(test_graphs), args.N)
    speedup = exact_time / max(surrogate_time, 1e-9)

    metrics["surrogate_time_s"] = surrogate_time
    metrics["exact_time_s"] = exact_time
    metrics["speedup"] = speedup

    print(f"[eval_surrogate] metrics:\n{json.dumps(metrics, indent=2)}")

    # Scatter plot.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(args.figure_path) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(targets, preds, s=6, alpha=0.4)
        lo = float(min(targets.min(), preds.min()))
        hi = float(max(targets.max(), preds.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax.set_xlabel("exact score")
        ax.set_ylabel("predicted score")
        ax.set_title(
            f"Surrogate vs exact (N={args.N}, n={len(test_graphs)}, "
            f"rho={metrics['spearman']:.3f})"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.figure_path, dpi=150)
        print(f"[eval_surrogate] saved scatter plot to {args.figure_path}")
    except Exception as e:
        print(f"[eval_surrogate] could not render scatter plot: {e!r}")


if __name__ == "__main__":
    main()
