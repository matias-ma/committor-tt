# committor

Committor functions for high-dimensional stochastic processes, built on top of
[chebdGE](https://github.com/matias-ma/chebdGE) (`chebttge`/`chebtuckerge`: continuous Chebyshev
tensor-train / Tucker approximation in JAX) and [functional TT-DE](https://github.com/matias-ma/tt-de) (JAX
density estimation; also the source of the Ginzburg-Landau model used as the
flagship example here). Implements the soft variational committor
formulation of Chen, Hoskins, Khoo & Lindsey,
["Committor Functions via Tensor Networks"](https://arxiv.org/abs/2106.12515),
with an eye towards the cryo-EM/RECOVAR application sketched in
[committor_in_cryo_em.pdf](papers/committor_in_cryo_em.pdf) (future work; not yet implemented here).

---

## Background and mathematics

### The committor function

Consider a particle evolving under the overdamped Langevin SDE

$$dX_t = -\nabla V(X_t)\,dt + \sqrt{2\beta^{-1}}\,dW_t, \qquad X_t \in \Omega \subset \mathbb{R}^d,$$

where $V$ is the potential energy, $\beta = 1/T$ is the inverse temperature, and $W_t$ is a standard Wiener process. The equilibrium distribution is the Boltzmann-Gibbs density $p(x) \propto e^{-\beta V(x)}$.

Given two disjoint metastable sets $A, B \subset \Omega$, the **committor function** $q : \Omega \to [0,1]$ is defined as

$$q(x) = \mathbb{P}(\tau_B < \tau_A \mid X_0 = x),$$

the probability that a trajectory initialised at $x$ reaches $B$ before $A$. It satisfies the backward Kolmogorov equation

$$-\beta^{-1}\Delta q(x) + \nabla V(x) \cdot \nabla q(x) = 0 \text{ in } \Omega\setminus(A\cup B), \qquad q|_{\partial A}=0,\quad q|_{\partial B}=1.$$

### Soft variational formulation

Hard boundary conditions are incompatible with a low-rank function-space parametrization (boundary measures aren't absolutely continuous w.r.t. Lebesgue measure), so instead we minimize the soft, penalized functional

$$\min_q \int_\Omega |\nabla q|^2\, p\,dx + \rho \int_\Omega q^2\, p_A\,dx + \rho\int_\Omega (q-1)^2\, p_B\,dx,$$

where $p_A$ and $p_B$ are smooth surrogate densities concentrated near $A$ and $B$, and $\rho > 0$ is a penalty weight; as $\rho \to \infty$ the minimizer converges to the true committor.

### How this repository solves it

chebdGE gives a differentiable, JAX-native way to *represent and evaluate* a
function in continuous Chebyshev TT/Tucker format (`ChebTT`/`ChebTucker`),
plus cross-approximation (`tt_qtensge`/`tucker_qtensge`) to fit one to a
known black-box function. It does **not** provide any PDE/variational solver
or TT/Tucker operator algebra, so solving the committor's variational
problem is this repository's own contribution, built directly on chebdGE's
representation:

* $q$ is a `CommittorAnsatz` (`committor/ansatz.py`): a `ChebTT`/`ChebTucker`
  whose Chebyshev-coefficient cores are ordinary trainable JAX arrays.
  Evaluation and its gradient reuse chebdGE's own evaluation code verbatim
  (via `jax.grad`/`jax.vmap`) -- nothing about TT/Tucker contraction is
  reimplemented here.
* Each expectation in the loss above is estimated by plain Monte Carlo over
  samples from $p$ (the potential's own equilibrium sampler, e.g.
  `ftde.sample_ginzburg_landau`) and from $p_A$/$p_B$ (the *Laplace/Gaussian
  approximation* of $p$ around each basin's energy minimizer -- see
  `committor/sampling.py` for why this, rather than an ad hoc isotropic
  Gaussian or filtering equilibrium samples, is the right general choice).
* The ansatz's cores are then trained directly against this Monte Carlo loss
  with Adam (`optax`) under a $\rho$-continuation schedule
  (`committor/training.py`), the same JAX+optax "generate samples, then
  minibatch" style ft-de itself uses for its density fits. A TT/Tucker
  ansatz evaluates as a chain of matrix products, so (like an RNN) it is
  prone to exploding gradients as $d$ grows; gradient clipping fixes this.

Every step of the pipeline -- representation, evaluation, gradients,
training, and diagnostics/plotting -- is plain JAX/equinox/optax, so it
places onto GPU automatically via JAX's normal device placement with no code
changes. The only host-side (numpy) pieces are the potentials' own MCMC
samplers (matching ft-de's own convention, since sampling doesn't need
autodiff) and a one-off scipy L-BFGS search for the Ginzburg-Landau basin
minimizers -- both cheap, one-time costs, not part of the per-step training
loop.

Per this project's convention, **TT is the primary format** (its cost is
only linear in $d$, which is the whole point of this approach at high
dimension); Tucker is also supported and preferred in the 3-7D band, where
its dense core is still cheap (`committor.domains.recommended_format`).

---

## Repository layout

```
src/committor/
├── potentials.py    Physical systems: double_well (toy, closed-form 1D oracle),
│                    ginzburg_landau (thin wrapper around ftde.ginzburg_landau)
├── ansatz.py         CommittorAnsatz: trainable ChebTT/ChebTucker + init/warm_start
├── domains.py        TT-vs-Tucker format selection by dimension
├── sampling.py        Laplace-approximation sampler for the p_A/p_B surrogates
├── variational.py     The soft variational Monte Carlo loss + rho-continuation schedule
├── training.py         Adam/optax training loop (with gradient clipping)
├── reference.py        Closed-form 1D committor oracle, for testing only
└── diagnostics.py      Relative-L2-vs-oracle metric; chebdGE plotting/summary wrappers

examples/
├── double_well.py      Sanity check across d = 1, 2, 4, 6, 10 vs. the closed-form oracle
└── ginzburg_landau.py   Flagship example: TT at d=20, Tucker at d=5

tests/                   pytest suite (potentials, ansatz, variational loss, end-to-end training)
```

---

## Installation

Follow [INSTALL.md](INSTALL.md) to install chebdGE, functional TT-DE, and this package.

## Quick start

```python
import jax
from committor import ansatz, potentials, training, diagnostics
from committor.reference import double_well_1d_committor

pot = potentials.double_well(d=4, beta=2.0)
a0 = ansatz.init(jax.random.PRNGKey(0), d=4, domain=pot.domain, rank=4, degree=16, format="auto")
a, diag = training.train_committor(a0, pot)

q_true = double_well_1d_committor(beta=2.0)
print(diagnostics.relative_l2_error(a, q_true, pot))
```

See `examples/ginzburg_landau.py` for the flagship, higher-dimensional TT
example (reusing `ftde.ginzburg_landau` for the potential), and
`examples/double_well.py` for the full sanity-check sweep.

---

## Running tests

```bash
pytest tests/
```

All tests run in well under a minute (CPU is enough at these problem sizes).

---

## Connection to cryo-EM (RECOVAR) -- future work

[committor_in_cryo_em.pdf](papers/committor_in_cryo_em.pdf) motivates this from cryo-EM conformational
heterogeneity analysis: RECOVAR (Gilles & Singer, PNAS 2025) estimates a
conformational density in a 4-10 dimensional PCA latent space, and viewed
via Boltzmann statistics as an effective free energy landscape, that latent
density is a natural input for this TT/Tucker committor solver -- too
high-dimensional for classical PDE solvers, but well within the
linear-in-$d$ regime of the TT approach. `committor.potentials.Potential` is
deliberately generic (just `log_density`/`sample`/two basin points) so a
RECOVAR-derived latent potential can be dropped in later without touching
`ansatz.py`, `variational.py`, or `training.py`; that integration itself is
not yet implemented.

---

## References

Chen, Y., Hoskins, J., Khoo, Y., Lindsey, M. (2023). *Committor Functions via Tensor Networks.* Journal of Computational Physics, 472:111646. ([arXiv:2106.12515](https://arxiv.org/abs/2106.12515))

Gilles, M. A., Singer, A. (2025). *Cryo-EM heterogeneity analysis using regularized covariance estimation and kernel regression.* PNAS, 122(9):e2419140122.

Peng, C., Khoo, Y., Wang, R. (2024). *Multivariate Density Estimation via Variance-Reduced Sketching* (arXiv:2401.11646) -- source of the Ginzburg-Landau parametrization reused here via `ftde.ginzburg_landau`.

---

## Contact

For any questions, bug reports, or suggestions to reach out at **mati@princeton.edu**.

## Research documents and earlier implementation

The original [committor paper](papers/committor_in_cryo_em.pdf) is preserved alongside the current research
write-ups and figures. The functional implementation is now the main package
in `src/`; the superseded implementation remains available in Git history.
