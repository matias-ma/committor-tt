"""Committor functions via chebdGE Chebyshev tensor-train / Tucker approximations.

This package implements the soft variational committor formulation of Chen,
Hoskins, Khoo & Lindsey (arXiv:2106.12515) on top of chebdGE's
(``chebttge``/``chebtuckerge``) continuous TT/Tucker Chebyshev machinery, and
reuses ``ftde`` (../ft-de) directly for the Ginzburg-Landau equilibrium
density and sampler.

Quickstart
----------
>>> from committor import potentials, ansatz, training, diagnostics
>>> pot = potentials.double_well(d=4)
>>> a = ansatz.init(jax.random.PRNGKey(0), d=4, domain=pot.domain, rank=4, degree=12)
>>> a, diag = training.train_committor(a, pot)
"""

from committor.ansatz import CommittorAnsatz, fit, init, warm_start
from committor.domains import recommended_format, resolve_format
from committor.potentials import Potential, double_well, find_local_maxima, ginzburg_landau, squared_tt_density
from committor.training import TrainingDiagnostics, heldout_loss, train_committor
from committor.variational import rho_schedule, soft_committor_loss

__all__ = [
    "CommittorAnsatz",
    "fit",
    "init",
    "warm_start",
    "recommended_format",
    "resolve_format",
    "Potential",
    "double_well",
    "find_local_maxima",
    "ginzburg_landau",
    "squared_tt_density",
    "TrainingDiagnostics",
    "heldout_loss",
    "train_committor",
    "rho_schedule",
    "soft_committor_loss",
]
