"""A trainable committor ansatz built directly on chebdGE's ChebTT/ChebTucker.

chebdGE's ``ChebTT``/``ChebTucker`` objects are ordinary ``equinox.Module``
pytrees whose Chebyshev-coefficient cores (``core_representation="coeff_tensor"``)
are plain JAX arrays. That means a ``ChebTT``/``ChebTucker`` built from those
cores is *already* differentiable and optax-trainable end to end through its
own evaluation code (a vectorized Clenshaw contraction) -- we do not need to
re-implement TT/Tucker evaluation, only own initialization and the trainable
parameter bookkeeping. chebdGE's own cross-approximation (``tt_qtensge`` /
``tucker_qtensge``) is reused only as an optional one-shot initializer
(:func:`warm_start`), since pivot selection inside cross-approximation is not
itself differentiable.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from chebttge import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from chebtuckerge.chebtucker import ChebTucker
from chebtuckerge.factors import ChebyshevCoeffFactor

from committor.domains import Format, resolve_format

Array = jnp.ndarray
DomainLike = Sequence[Sequence[float]]


def _as_static_domain(domain: DomainLike) -> Tuple[Tuple[float, float], ...]:
    """Normalize a domain to a hashable tuple of (lo, hi) pairs.

    Kept as plain Python floats (not a ``jnp.ndarray``) so it is a *static*
    piece of metadata rather than a differentiable pytree leaf: chebdGE's
    per-core ``ChebyshevCoeffCore``/``ChebyshevCoeffFactor`` constructors call
    ``float(domain[0])`` internally, which fails on a traced array.
    """
    domain_arr = jnp.asarray(domain, dtype=jnp.float64)
    if domain_arr.ndim != 2 or domain_arr.shape[1] != 2:
        raise ValueError(f"domain must have shape (d, 2), got {domain_arr.shape}")
    return tuple((float(lo), float(hi)) for lo, hi in domain_arr)


class CommittorAnsatz(eqx.Module):
    """A committor ``q(x)`` represented as a trainable ChebTT or ChebTucker.

    Only ``tt_cores`` / (``tucker_core``, ``tucker_factors``) are trainable
    pytree leaves; ``format`` and ``domain`` are static. Use :func:`init` or
    :func:`warm_start` to construct one rather than the constructor directly.
    """

    format: str = eqx.field(static=True)
    domain: Tuple[Tuple[float, float], ...] = eqx.field(static=True)
    tt_cores: Optional[Tuple[Array, ...]]
    tucker_core: Optional[Array]
    tucker_factors: Optional[Tuple[Array, ...]]

    def __init__(
        self,
        format: Format,
        domain: DomainLike,
        tt_cores: Optional[Sequence[Array]] = None,
        tucker_core: Optional[Array] = None,
        tucker_factors: Optional[Sequence[Array]] = None,
    ):
        if format not in ("tt", "tucker"):
            raise ValueError(f"format must be 'tt' or 'tucker', got {format!r}")
        self.format = format
        self.domain = _as_static_domain(domain)
        self.tt_cores = tuple(tt_cores) if tt_cores is not None else None
        self.tucker_core = tucker_core
        self.tucker_factors = tuple(tucker_factors) if tucker_factors is not None else None

    @property
    def d(self) -> int:
        return len(self.domain)

    def _domain_array(self) -> Array:
        return jnp.asarray(self.domain, dtype=jnp.float64)

    def _build(self):
        """Materialize the underlying chebdGE ``ChebTT``/``ChebTucker`` object."""
        if self.format == "tt":
            cores = tuple(
                ChebyshevCoeffCore(c, domain=self.domain[k])
                for k, c in enumerate(self.tt_cores)
            )
            return ChebTT(cores=cores, domain=self._domain_array())
        factors = tuple(
            ChebyshevCoeffFactor(f, domain=self.domain[k])
            for k, f in enumerate(self.tucker_factors)
        )
        return ChebTucker(self.tucker_core, factors, self._domain_array())

    def __call__(self, x: Array) -> Array:
        """Evaluate q at ``x``, shape ``(d,)`` or ``(batch, d)``."""
        return self._build()(x)

    def value_and_grad(self, x: Array) -> Tuple[Array, Array]:
        """Evaluate ``(q(x), grad_x q(x))`` for a batch ``x`` of shape ``(n, d)``.

        Uses plain JAX autodiff through chebdGE's own evaluation code (rather
        than chebdGE's symbolic ``computations.gradient``, which only supports
        the chebfun-object core representation, not ``coeff_tensor``).
        """
        model = self._build()

        def scalar(xi: Array) -> Array:
            return model(xi[None, :])[0]

        return jax.vmap(jax.value_and_grad(scalar))(x)

    def finalize(self):
        """Return the plain chebdGE ``ChebTT``/``ChebTucker``, e.g. for use with
        ``chebttge.diagnostics``/``chebttge.plotting`` (or the Tucker analogues)."""
        return self._build()


def _init_scale_zero_mode(rank: int, d: int) -> float:
    """Chebyshev T_0 coefficient making a rank-1 constant path equal 0.5.

    ``0.5 ** (1 / d)`` per site so that the product over ``d`` sites (each
    contributing this value along the ``(0, 0)`` rank index, since
    ``T_0 == 1``) multiplies out to exactly 0.5 -- a neutral prior for a
    committor before any training.
    """
    del rank  # only the per-site count matters
    return 0.5 ** (1.0 / d)


def init(
    key: jax.Array,
    d: int,
    domain: DomainLike,
    rank: int,
    degree: int,
    format: Format | str = "auto",
    init_scale: float = 0.05,
) -> CommittorAnsatz:
    """Randomly initialize a :class:`CommittorAnsatz` with q(x) ~= 0.5 everywhere.

    Parameters
    ----------
    key        : PRNG key.
    d          : number of variables.
    domain     : (d, 2) array-like of per-variable [lo, hi] bounds.
    rank       : TT bond dimension / Tucker rank (uniform across sites/modes).
    degree     : Chebyshev polynomial degree per site/mode (``degree + 1`` coefficients).
    format     : "tt", "tucker", or "auto" (resolved by dimension, see domains.py).
    init_scale : standard deviation of the random perturbation around the
                 constant-0.5 initial guess.
    """
    fmt = resolve_format(d, format)
    c0 = _init_scale_zero_mode(rank, d)
    if fmt == "tt":
        ranks = (1,) + (rank,) * (d - 1) + (1,)
        keys = jax.random.split(key, d)
        cores = []
        for k in range(d):
            shape = (ranks[k], ranks[k + 1], degree + 1)
            core = init_scale * jax.random.normal(keys[k], shape, dtype=jnp.float64)
            core = core.at[0, 0, 0].set(c0)
            cores.append(core)
        return CommittorAnsatz(format="tt", domain=domain, tt_cores=cores)

    key_core, key_factors = jax.random.split(key)
    core = init_scale * jax.random.normal(key_core, (rank,) * d, dtype=jnp.float64)
    core = core.at[(0,) * d].set(1.0)
    keys = jax.random.split(key_factors, d)
    factors = []
    for k in range(d):
        factor = init_scale * jax.random.normal(keys[k], (rank, degree + 1), dtype=jnp.float64)
        factor = factor.at[0, 0].set(c0)
        factors.append(factor)
    return CommittorAnsatz(format="tucker", domain=domain, tucker_core=core, tucker_factors=factors)


def warm_start(
    guess_fn: Callable[[Array], Array],
    d: int,
    domain: DomainLike,
    rank: int,
    format: Format | str = "auto",
    **build_options,
) -> CommittorAnsatz:
    """Initialize a :class:`CommittorAnsatz` from a chebdGE cross-approximation of ``guess_fn``.

    Cross-approximation itself is a one-shot, non-differentiable fit (pivot
    selection uses argmax), so it is used purely to produce a good starting
    point; the resulting cores/factors are then lifted out as ordinary
    trainable arrays. ``build_options`` are forwarded to
    ``ChebTTOptions``/``ChebTuckerOptions`` (e.g. ``tol``, ``n_starts``).
    """
    fmt = resolve_format(d, format)
    domain_arr = jnp.asarray(domain, dtype=jnp.float64)
    build_options.setdefault("core_representation", "coeff_tensor")

    if fmt == "tt":
        from chebttge import ChebTTOptions, tt_qtensge

        opts = ChebTTOptions(domain=domain_arr, max_rank=rank, **build_options)
        tt = tt_qtensge(guess_fn, opts)
        cores = tuple(core.coeffs for core in tt.cores)
        return CommittorAnsatz(format="tt", domain=domain, tt_cores=cores)

    from chebtuckerge import ChebTuckerOptions, tucker_qtensge

    opts = ChebTuckerOptions(domain=domain_arr, max_rank=rank, **build_options)
    tucker = tucker_qtensge(guess_fn, opts)
    factors = tuple(factor.core.coeffs[0] for factor in tucker.factors)
    return CommittorAnsatz(
        format="tucker", domain=domain, tucker_core=tucker.core, tucker_factors=factors
    )


def fit(
    ansatz: CommittorAnsatz,
    guess_fn: Callable[[Array], Array],
    x: Array,
    *,
    steps: int = 1500,
    batch_size: int = 1024,
    learning_rate: float = 1e-2,
    seed: int = 0,
) -> Tuple[CommittorAnsatz, float]:
    """Least-squares fit of ``ansatz`` to ``guess_fn`` at the points ``x`` (n, d), e.g. a warm start.

    Unlike :func:`warm_start` this keeps the ansatz's own rank and degree (cross
    approximation picks both adaptively), so the result can be trained with the
    same settings as a random start. Adam with a cosine-decayed rate over random
    minibatches of ``x``; returns the fitted ansatz and the last minibatch MSE.
    """
    y = guess_fn(x)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adam(optax.cosine_decay_schedule(learning_rate, steps, alpha=0.01)))
    opt_state = optimizer.init(eqx.filter(ansatz, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(a, opt_state, xb, yb):
        loss, grads = eqx.filter_value_and_grad(lambda m: jnp.mean((m(xb) - yb) ** 2))(a)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(a, eqx.is_inexact_array))
        return eqx.apply_updates(a, updates), opt_state, loss

    key = jax.random.PRNGKey(seed)
    for _ in range(steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch_size,), 0, x.shape[0])
        ansatz, opt_state, loss = step(ansatz, opt_state, x[idx], y[idx])
    return ansatz, float(loss)
