"""CLI: generate training data, train the surrogate GNN, and save it.

Usage:
    PYTHONPATH=. python surrogate/train.py --N 30 --r_max 0.8 \
        --num_samples 10000 --epochs 200 --output models/surrogate.pt
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

from surrogate.scorer import SurrogateScorer  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the surrogate GNN.")
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--r_max", type=float, default=0.8)
    p.add_argument("--num_samples", type=int, default=10_000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=str, default="models/surrogate.pt")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["none", "step", "cosine"],
        help="Learning rate schedule: none|step|cosine (default cosine).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[train_surrogate] device={args.device}")
    print(
        f"[train_surrogate] N={args.N} r_max={args.r_max} "
        f"num_samples={args.num_samples} epochs={args.epochs}"
    )

    scorer = SurrogateScorer(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        device=args.device,
    )
    print(
        f"[train_surrogate] GNN parameters: "
        f"{scorer.model.count_parameters():,}"
    )

    t0 = time.time()
    data = scorer.generate_training_data(
        N=args.N, r_max=args.r_max, num_samples=args.num_samples, seed=args.seed
    )
    print(
        f"[train_surrogate] generated {len(data)} valid graphs in "
        f"{time.time() - t0:.1f}s (yield "
        f"{100 * len(data) / args.num_samples:.1f}%)"
    )

    # 80/20 split (configurable).
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(data))
    n_val = int(args.val_frac * len(data))
    val = [data[int(i)] for i in perm[:n_val]]
    train = [data[int(i)] for i in perm[n_val:]]
    print(f"[train_surrogate] train={len(train)} val={len(val)}")

    t0 = time.time()
    log = scorer.train(
        train,
        val,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        alpha=args.alpha,
        margin=args.margin,
        verbose=True,
        lr_schedule=args.lr_schedule,
    )
    print(f"[train_surrogate] training took {time.time() - t0:.1f}s")

    # Final evaluation.
    ks = tuple(k for k in (100, 500, 1000) if k <= len(val))
    metrics = scorer.evaluate(val, ks=ks)
    print(f"[train_surrogate] final metrics: {json.dumps(metrics, indent=2)}")

    # Save model.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    scorer.save(args.output)
    print(f"[train_surrogate] saved model to {args.output}")

    # Save training log alongside model.
    log_path = args.output.replace(".pt", "_log.json")
    with open(log_path, "w") as f:
        json.dump(
            {"args": vars(args), "log": log, "final_metrics": metrics}, f, indent=2
        )
    print(f"[train_surrogate] saved log to {log_path}")


if __name__ == "__main__":
    main()
