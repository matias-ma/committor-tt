# committor_tt

Python implementation of **"Committor Functions via Tensor Networks"** by Chen, Hoskins, Khoo, and Lindsey ([arXiv:2106.12515](https://arxiv.org/abs/2106.12515)), together with an application to cryo-EM conformational dynamics via the RECOVAR latent space (see `committor_in_cryo_em.pdf` in this repository).

Computes committor functions for high-dimensional stochastic processes under overdamped Langevin dynamics by solving a variational problem whose coefficient tensor is represented and optimised in matrix product state / tensor-train (MPS/TT) format, achieving computational and memory complexity that scales *linearly* in the number of dimensions.

---

## Background and mathematics

### The committor function

Consider a particle evolving under the overdamped Langevin SDE

$$dX_t = -\nabla V(X_t)\,dt + \sqrt{2\beta^{-1}}\,dW_t, \qquad X_t \in \Omega \subset \mathbb{R}^d,$$

where $V$ is the potential energy, $\beta = 1/T$ is the inverse temperature, and $W_t$ is a standard Wiener process. The equilibrium distribution is the Boltzmann–Gibbs density $p(x) \propto e^{-\beta V(x)}$.

Given two disjoint metastable sets $A, B \subset \Omega$, the **committor function** $q : \Omega \to [0,1]$ is defined as

$$q(x) = \mathbb{P}(\tau_B < \tau_A \mid X_0 = x),$$

the probability that a trajectory initialised at $x$ reaches $B$ before $A$. It satisfies the backward Kolmogorov equation

$$-\beta^{-1}\Delta q(x) + \nabla V(x) \cdot \nabla q(x) = 0 \text{ in } \Omega\setminus(A\cup B), \qquad q|_{\partial A}=0,\quad q|_{\partial B}=1.$$

### Soft variational formulation

Rather than enforcing the boundary conditions as hard constraints (which is incompatible with a TT parametrisation), we use a penalised soft formulation:

$$\min_q \int_\Omega |\nabla q|^2\, p\,dx + \rho \int_\Omega q^2\, p_A\,dx + \rho\int_\Omega (q-1)^2\, p_B\,dx,$$

where $p_A$ and $p_B$ are smooth densities (typically Gaussians) concentrated near $A$ and $B$ respectively, and $\rho > 0$ is a penalty weight. As $\rho \to \infty$ the soft committor converges to the true committor.

### Tensor-train parametrization

We expand $q$ in a tensor-product basis $\{\varphi^{(1)}_{i_1}(x_1)\cdots\varphi^{(d)}_{i_d}(x_d)\}$ and represent the coefficient tensor $Q(i_1,\dots,i_d)$ in **MPS / tensor-train format**:

$$Q(i_1,\dots,i_d) = G_1(:,i_1,:)\, G_2(:,i_2,:)\,\cdots\, G_d(:,i_d,:),$$

where each core $G_k \in \mathbb{R}^{r_{k-1}\times n_k \times r_k}$ and the bond dimensions $r_k$ (the *TT rank*) control expressiveness vs. cost.

The variational objective becomes a quadratic form $\langle Q | W | Q \rangle - 2\rho\langle Q | h^B \rangle$ in $Q$, where the operator $W = H + \rho H^A + \rho H^B$ is assembled as a **matrix product operator (MPO)** and $h^B$ as a rank-1 TT. When the equilibrium density $p$ is itself representable in TT format (e.g. as a rank-$J$ TT via a transfer-matrix eigendecomposition), all integrals factorise into $d$ independent 1D quadratures, giving **O(d) complexity** in both time and memory.

The coefficient TT is then optimised by **alternating least squares (ALS)**: iterating over sites $k = 1,\dots,d$, holding all other cores fixed, and solving the resulting small linear system for $G_k$ exactly. Each sweep costs $O(d)$ contractions.

---

## Repository layout

```
committor/
├── _types.py              Shared type aliases (Array, ScalarFn) and CommittorResult protocol
├── basis.py               Legendre, Fourier, and density-weighted orthonormal basis functions
├── tensor_train.py        TTTrain / MPOTrain data structures; environment contractions
├── assembly.py            Per-dimension quadrature matrices; MPO and h^B assembly (rank-1 density)
├── assembly_tt_density.py MPO assembly generalised to rank-J TT-format densities; TTDensitySpec
├── als.py                 Left-to-right / right-to-left ALS sweeps; multi-sweep driver
├── solvers.py             High-level entry points: solve_committor_1d / _nd_dense / _nd_tt
├── problems.py            Double-well benchmark, exact reference committor, error metrics
├── notebook_api.py        Notebook-friendly find_committor() front end; create_mixture_tt_density()
└── ginzburg_landau.py     (Work in progress) GL transfer kernel, rank-J density TT, GL committor solver

tests/
├── test_basis.py
├── test_tensor_train.py
├── test_assembly.py
├── test_als.py
├── test_solvers.py
└── test_ginzburg_landau.py

quickstart.ipynb           Step-by-step walkthrough (1D, double-well ND, toy RECOVAR example)
committor_in_cryo_em.pdf   Accompanying paper: cryo-EM motivation and mathematical background
```


---

## Installation

```bash
# Clone the repository
git clone https://github.com/matias-ma/committor_tt.git
cd committor_tt

# Install with development and notebook dependencies
pip install -e ".[dev,notebook]"
```

**Dependencies:** `numpy >= 1.23`, `scipy >= 1.9`. Python 3.9+.

---

## Quick start

### 1D committor (closed-form, fast)

```python
from committor.solvers import solve_committor_1d
from committor.problems import double_well_potential, exact_committor_1d
import numpy as np

result = solve_committor_1d(V=double_well_potential, beta=5.0, a=-2.0, b=2.0, nbasis=30, rho=400.0)

x = np.linspace(-1, 1, 200)
print(result.q(x))    # committor values
print(result.dq(x))   # derivative (useful for computing reactive flux)

# Compare against the exact reference
q_true = exact_committor_1d(double_well_potential, beta=5.0)
```

### n-D double-well via TT-ALS (scales to large d)

```python
from committor.problems import build_double_well_nd_problem_weighted
from committor.solvers import solve_committor_nd_tt
import numpy as np

d = 20
basis, per_dim = build_double_well_nd_problem_weighted(d=d, beta=5.0, nbasis=30)
result = solve_committor_nd_tt(per_dim, basis, rho=400.0, tt_rank=4, n_sweeps=30, verbose=True)

X = np.random.randn(1000, d)  # (n_samples, d)
q_vals = result.q(X)
```

### Non-product mixture density

```python
from committor.notebook_api import find_committor, make_basis, create_mixture_tt_density, make_gaussian_product_states
import numpy as np

# Build basis
basis = make_basis(basis_kind="legendre", ns=[20]*4, intervals=[(-2.0, 2.0)]*4)

# Rank-2 mixture density: 60% near x=-1, 40% near x=+1
density_spec = create_mixture_tt_density(
    product_weight_fns_list=[
        [lambda x: np.exp(-5*(x+1)**2)] * 4,   # component 1
        [lambda x: np.exp(-5*(x-1)**2)] * 4,   # component 2
    ],
    committor_basis=basis,
    component_weights=[0.6, 0.4],
)

wA_fns, wB_fns = make_gaussian_product_states(
    a_center=[-1.0]*4,
    b_center=[+1.0]*4,
    sigma=0.1,
)

result = find_committor(
    density_spec=density_spec,
    wA_fns=wA_fns,
    wB_fns=wB_fns,
    rho=400.0,
    method="tt",
    tt_rank=4,
    verbose=True,
)
```

### WORK IN PROGRESS: Ginzburg-Landau (d = 50, non-product density)

```python
from committor.ginzburg_landau import solve_gl_committor

result = solve_gl_committor(d=50, lam=0.03, T=8.0, verbose=True)
# result.q(X) where X has shape (n_samples, 50)
```

---

## Running tests

```bash
pytest                          # full suite
pytest tests/test_solvers.py    # one module
pytest -k "1d"                  # filter by name
pytest --cov=committor          # coverage report
```

Most tests complete in a few seconds. `test_solvers.py::TestSolveCommittorNDTT::test_nd_accuracy_d5` takes around 30 s.

---

## Algorithm overview

| Module | Responsibility |
|---|---|
| `basis.py` | Legendre (plain and density-weighted), Fourier bases |
| `assembly.py` | Per-dimension quadrature → MPO and $h^B$ TT (rank-1 / product density) |
| `assembly_tt_density.py` | Same, generalised to rank-$J$ TT densities; `TTDensitySpec` container |
| `als.py` | L→R and R→L sweeps; environment caching; $\rho$-continuation driver |
| `solvers.py` | User-facing entry points; dispatches between product and TT-density paths |
| `ginzburg_landau.py` | Transfer-kernel eigendecomposition → rank-$J$ density TT; GL solver |

The MPO bond dimension is 4 for product densities and $2J+2$ for rank-$J$ TT densities.

---

## Connection to cryo-EM (RECOVAR)

The `committor_in_cryo_em.pdf` paper in this repository motivates the algorithm from the perspective of cryo-EM conformational heterogeneity analysis. RECOVAR (Gilles & Singer, PNAS 2025) estimates a conformational density in a 4–10 dimensional PCA latent space. This latent density, viewed via Boltzmann statistics as an effective free energy landscape, is a natural input for the TT committor solver: the dimensionality is too high for classical PDE solvers but well within the linear-scaling regime of the TT approach.

A sketch of the full pipeline:

1. Run RECOVAR to obtain latent coordinates and a density estimate $\hat\rho(z)$.
2. Express $\hat\rho$ in TT format (e.g. via the tensor-train density estimation algorithm at [matias-ma/tt-de](https://github.com/matias-ma/tt-de), or via `create_mixture_tt_density` for mixture models).
3. Identify metastable states $A$ and $B$ in latent space and build soft-boundary measures $p_A$, $p_B$.
4. Call `solve_committor_nd_tt` with `density_spec` to obtain the committor $q(z)$.
5. Interpret level sets of $q$ as transition-state regions of the conformational landscape.

---

## References

Chen, Y., Hoskins, J., Khoo, Y., Lindsey, M. (2023). *Committor Functions via Tensor Networks.* Journal of Computational Physics, 472:111646. ([arXiv:2106.12515](https://arxiv.org/abs/2106.12515))

Gilles, M. A., Singer, A. (2025). *Cryo-EM heterogeneity analysis using regularized covariance estimation and kernel regression.* PNAS, 122(9):e2419140122.

---

## Contact

For any questions, bug reports, or suggestions to reach out at **mati@princeton.edu**.
