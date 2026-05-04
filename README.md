# committor

Python implementation of **"Committor Functions via Tensor Networks"** by Chen, Hoskins, Khoo, and Lindsey (arXiv:2106.12515).

Computes committor functions for high-dimensional stochastic processes in the **overdamped Langevin** setting by solving a variational problem parametrised in a matrix product state / tensor-train (MPS/TT) format.

---

## Project layout

```
committor/             Core library
├── _types.py          Shared type aliases and CommittorResult protocol
├── basis.py           Univariate and tensor-product basis functions
├── tensor_train.py    TTTrain / MPOTrain data structures and contractions
├── assembly.py        Quadrature matrices, MPO and h^B assembly
├── als.py             Alternating least-squares (ALS) solver
├── solvers.py         High-level solver entry points and result types
├── problems.py        Double-well benchmark and error metrics
└── ginzburg_landau.py GL kernel, density TT, and GL committor solver

tests/                 Pytest test suite
├── test_basis.py
├── test_tensor_train.py
├── test_assembly.py
├── test_als.py
├── test_solvers.py
└── test_ginzburg_landau.py

notebooks/
└── 01_committor_examples.ipynb   Step-by-step walkthroughs of paper examples
```

---

## Quick start

### Install

```bash
pip install -e ".[dev,notebook]"
```

### 1-D committor (closed-form, fast)

```python
from committor.solvers import solve_committor_1d
from committor.problems import double_well_potential, exact_committor_1d
import numpy as np

result = solve_committor_1d(V=double_well_potential, beta=5.0, a=-2.0, b=2.0)

x = np.linspace(-1, 1, 100)
print(result.q(x))    # committor values
print(result.dq(x))   # derivative (for reactive-flow eq.)
```

### n-D double-well TT-ALS (scalable)

```python
from committor.problems import build_double_well_nd_problem_weighted
from committor.solvers  import solve_committor_nd_tt

d = 20  # dimensions
basis, per_dim = build_double_well_nd_problem_weighted(d=d, beta=5.0, nbasis=30)
result = solve_committor_nd_tt(per_dim, basis, rho=400.0, tt_rank=4, n_sweeps=30)

X = np.random.randn(1000, d)        # sample points, shape (n, d)
q_vals = result.q(X)               # committor approximation
```

### Ginzburg-Landau (d=50, non-product density)

```python
from committor.ginzburg_landau import solve_gl_committor

result = solve_gl_committor(d=50, lam=0.03, T=8.0, verbose=True)
```

---

## Running tests

```bash
pytest                        # all tests
pytest tests/test_solvers.py  # one module
pytest -k "1d"                # filter by name
pytest --cov=committor        # coverage report
```

The test suite covers basis orthonormality, TT/MPO algebra, ALS convergence, and solver accuracy. Most tests run in a few seconds; `test_solvers.py::TestSolveCommittorNDTT::test_nd_accuracy_d5` takes ~30 s.

---

## Reproducing paper results

Open the notebook for annotated walkthroughs:

```bash
jupyter notebook notebooks/01_committor_examples.ipynb
```

To run the full paper experiments from the command line:

| Experiment | Command |
|-----------|---------|
| Section 4.1 — double well T=0.2, d=20 | `python -m committor.problems` |
| Section 4.2 — Ginzburg-Landau T=8/16 | see notebook §4 |

---

## Algorithm overview

The library implements the variational (soft-boundary) formulation:

```
argmin_q  ∫|∇q|² p dx  +  ρ ∫q² p_A dx  +  ρ ∫(q−1)² p_B dx
```

where `p` is the Boltzmann density, and `p_A`, `p_B` are Gaussian soft-boundary measures concentrated near the metastable sets A and B.

The coefficient tensor `Q` (from the expansion q = Σ Q(i) φ(i)) is represented in MPS/TT format and optimised by **alternating least squares (ALS)**, exploiting that all integrals factorise as MPO-MPS contractions with O(d) complexity.

Key modules:

| Module | Responsibility |
|--------|---------------|
| `basis.py` | Legendre, Fourier, and density-weighted orthogonal bases |
| `assembly.py` | Per-dimension quadrature matrices → MPO and h^B TT |
| `als.py` | L→R and R→L sweeps; environment caching |
| `solvers.py` | User-facing entry points with ρ-continuation |
| `ginzburg_landau.py` | Transfer-kernel eigendecomposition → rank-J density TT |

---

## References

Chen, Y., Hoskins, J., Khoo, Y., Lindsey, M. (2021).  
*Committor Functions via Tensor Networks.* arXiv:2106.12515.
