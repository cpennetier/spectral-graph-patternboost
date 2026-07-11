# Geometric Spectral Graph Design via PatternBoost

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Population-based search for sparse, sphere-embedded graphs with high algebraic connectivity under hard distance constraints.**

> **Research program — verifiable learning and decision systems.**  
> Proposal mechanisms may be heuristic or learned; the structures they produce, their feasibility, and their value remain independently measurable.
>
> **Role of this repository:** search-based proposal generation for constrained graph structures.  
> [PatternBoost](https://github.com/cpennetier/spectral-graph-patternboost) ·
> [Graph diffusion](https://github.com/cpennetier/spectral-graph-diffusion) ·
> [Ephemeris Kernel](https://github.com/cpennetier/ephemeris-kernel) ·
> [Epure Arena](https://github.com/cpennetier/epure-arena)

**[Research note (PDF)](paper/paper.pdf)** · 14 pages · approximately $20 of compute · one weekend

## Research question

Can population-based generative search discover sparse graph structures with high spectral efficiency when the admissible edges are fixed by geometry—and where does classical search remain better?

The repository treats this as an empirical research question rather than assuming that the learned or population-based method must win.

## Problem

Given $N$ vertices with fixed positions on the unit sphere, find a connected graph $G=(V,E)$ that maximizes

$$
\Phi(G)=\frac{\lambda_2(L_G)\,N}{|E|},
$$

subject to the great-circle constraint

$$
d(\mathbf p_i,\mathbf p_j)\le r_{\max}
\qquad \forall (i,j)\in E.
$$

Here:

- $\lambda_2(L_G)$ is the Fiedler value, the second-smallest eigenvalue of the graph Laplacian;
- $|E|$ is the number of edges;
- $r_{\max}$ defines the admissible-edge mask induced by the spherical embedding.

The objective rewards **spectral connectivity per edge**. The geometric mask removes the symmetry available to unconstrained extremal constructions: the feasible graph family depends on the sampled vertex positions.

## What this artifact contains

The repository contributes four connected pieces:

1. **A constrained graph-design environment** for Axplorer's PatternBoost framework.
2. **An exact spectral objective** computed from the graph Laplacian.
3. **A learned GNN surrogate** for ranking candidate graphs.
4. **Classical baselines**, including KNN constructions, random graph families, and simulated annealing.

The exact eigensolver remains the source of truth. The surrogate proposes or ranks; it does not redefine the objective.

## Results

| Setting | KNN $k=5$ | PatternBoost best | Improvement |
|---|---:|---:|---:|
| Tight constraint ($r_{\max}=0.8$, 70 admissible edges) | 0.072 | 0.087 | **+20.7%** |
| Moderate constraint ($r_{\max}=1.2$, 140 admissible edges) | 0.141 | 0.459 | **+226%** |

![Learning dynamics](figures/fig5_combined_learning.png)

*Left: tight constraint, converged by epoch 14. Right: moderate constraint, still improving at epoch 25. Dashed lines show the KNN $k=5$ baseline.*

![Baseline comparison](figures/fig3_baseline_comparison.png)

The broader result is deliberately mixed:

- Classical Barabási–Albert, Watts–Strogatz, and Erdős–Rényi generators are poor constrained baselines: in the tested setup, 84% of their proposed edges violate the distance mask.
- Simulated annealing reaches higher single-instance scores at $N=30$. This is a useful negative result: at this scale, the spectral landscape remains tractable for a strong classical local-search method.
- A 22,337-parameter GNN surrogate predicts Fiedler scores with Spearman $\rho=0.963$.
- At $N=30$, the surrogate does not create a wall-clock advantage because exact eigendecomposition is already cheap. Its potential value begins only when the $O(N^3)$ spectral calculation becomes material.

## Architecture

### Constrained PatternBoost environment

The Axplorer environment in `src/envs/` implements:

- vertex positions sampled uniformly on $S^2$, with a connectivity check;
- an admissible-edge mask from great-circle distance;
- disconnected-component repair using the shortest feasible cross-component edge;
- greedy edge swaps for local improvement;
- exact scoring through `numpy.linalg.eigvalsh`;
- symmetric sparse tokenization compatible with the Axplorer search loop.

### GNN surrogate

`surrogate/scorer.py` implements a pure-PyTorch graph convolutional scorer with:

- 4 message-passing layers;
- residual connections and layer normalization;
- node features $(x,y,z,\mathrm{degree})$;
- edge features given by great-circle distance;
- MSE plus pairwise-ranking loss;
- no PyG dependency.

### Baselines

`benchmarks/` contains:

- KNN constructions;
- classical random graph generators;
- simulated annealing;
- scripts for direct score comparison.

## Quick start

### Install

```bash
git clone https://github.com/cpennetier/spectral-graph-patternboost.git
cd spectral-graph-patternboost
pip install -r requirements.txt
```

Axplorer is required only for the full PatternBoost training run:

```bash
git clone https://github.com/AxiomMath/axplorer.git
cd axplorer
pip install -e .
```

### Run the standalone artifact (no Axplorer)

```bash
# End-to-end CPU demo
python examples/quick_start.py

# Surrogate tests
pytest tests/test_surrogate.py -v
```

### Run the environment-backed pieces (Axplorer checkout required)

The constrained environment extends Axplorer base classes, so the
environment tests, the baselines, and surrogate training resolve those
modules from an Axplorer checkout: install the environment module into it
(the copy step in the reproduce section below) and put the checkout on
`PYTHONPATH`.

```bash
export PYTHONPATH=/path/to/axplorer

# Environment tests
pytest tests/ -v

# Baselines
python benchmarks/baseline_comparison.py \
  --N 30 --r_max 0.8

python benchmarks/simulated_annealing.py \
  --N 30 --r_max 0.8 --sa_runs 20

# Train the GNN surrogate on CPU
python surrogate/train.py \
  --N 30 --r_max 0.8 \
  --num_samples 5000 --epochs 100
```

### Reproduce the PatternBoost runs

<!-- IMPLEMENTATION PLACEHOLDER — DO NOT REMOVE SILENTLY.
The public snapshot currently uses a legacy application-specific module and
environment identifier for the Axplorer integration. Before committing this
README, either:
1. resolve <ENV_MODULE_PATH>, <ENV_IMPORT>, and <ENV_NAME> against neutral,
   verified identifiers in the repository; or
2. leave this comment and the placeholders visible rather than inventing a
   command.
Preserve the complete training configuration below. Verified against current
Axplorer main: every flag below exists on its train.py EXCEPT
--encoding_tokens, which is an environment-constructor parameter of the
reported run rather than a train.py flag.
-->

```bash
cd /path/to/axplorer

cp /path/to/spectral-graph-patternboost/<ENV_MODULE_PATH> src/envs/
# Register: <ENV_IMPORT>

python train.py \
  --env_name <ENV_NAME> \
  --N 30 \
  --r_max 0.8 \
  --max_epochs 25 \
  --max_steps 5000 \
  --gensize 100000 \
  --pop_size 30000 \
  --num_samples_from_model 100000 \
  --n_layer 4 \
  --n_embd 256 \
  --n_head 8 \
  --batch_size 64 \
  --temperature 0.6 \
  --inc_temp 0.1 \
  --keep_only_unique true \
  --encoding_tokens single_integer \
  --always_search true \
  --num_workers 8
```

Reported hardware: GCP `n1-standard-8` with one NVIDIA T4.  
Reported training time: approximately 10 hours per run.  
Reported cost: approximately $10 per run.

## Repository map

```text
benchmarks/    classical baselines and simulated annealing
examples/      standalone quick start
figures/       rendered result figures used in this README
paper/         research note
src/envs/      constrained PatternBoost environment
surrogate/     GNN scorer and training code
tests/         environment and scoring tests
```

## Scope and limitations

- All reported experiments use $N=30$.
- The two constraint regimes use one sampled vertex layout each; multi-layout and multi-seed variance are not reported.
- Simulated annealing outperforms PatternBoost on the tested single-instance comparison.
- The surrogate has no measured speed advantage at $N=30$.
- The Fiedler value is a spectral proxy. It is not a complete measure of the downstream behavior of a graph.
- Scaling behavior beyond the tested regime remains open.

## Companion experiment

[`spectral-graph-diffusion`](https://github.com/cpennetier/spectral-graph-diffusion) studies the same kind of externally verified spectral target using a learned generative model rather than population-based search.

The two repositories are not one software stack. They are two proposal mechanisms evaluated against an explicit spectral quantity.

## Citation

```bibtex
@misc{pennetier2026patternboost,
  author       = {Pennetier, Christophe},
  title        = {Geometric Spectral Graph Design via PatternBoost},
  year         = {2026},
  howpublished = {Research note and software artifact},
  url          = {https://github.com/cpennetier/spectral-graph-patternboost}
}
```

## License

MIT — see [LICENSE](LICENSE).
