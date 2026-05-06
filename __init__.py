"""committor — Tensor-train committor functions (Chen, Hoskins, Khoo, Lindsey 2021).

Implements the MPS/TT-based approach to computing committor functions from
arXiv:2106.12515.  The package is organised by responsibility:

    _types          — shared type aliases and the CommittorResult protocol
    basis           — univariate and tensor-product basis functions
    tensor_train    — TT/MPO data structures and contraction helpers
    assembly        — quadrature matrices and MPO/h^B assembly
    als             — alternating least squares (ALS) solver
    solvers         — high-level solver entry points and result types
    problems        — double-well potential utilities and error metrics
    ginzburg_landau — GL-specific density TT, MPO assembly, and solver

Typical usage
-------------
1D committor (dense, closed-form):

    from committor.solvers import solve_committor_1d
    from committor.problems import double_well_potential, exact_committor_1d

    result = solve_committor_1d(V=double_well_potential, beta=5.0, a=-2.0, b=2.0)
    q_vals = result.q(x_grid)

n-D TT-ALS committor (scalable):

    from committor.problems import build_double_well_nd_problem
    from committor.solvers  import solve_committor_nd_tt

    basis, per_dim = build_double_well_nd_problem(d=20, beta=5.0, nbasis=30)
    result = solve_committor_nd_tt(per_dim, basis, rho=400.0, tt_rank=4)
    q_vals = result.q(X)   # X shape (n_samples, 20)
"""

# --------------------------------------------------------------------------
# Re-export the most commonly used public API so users can write
#   from committor import solve_committor_1d, double_well_potential
# --------------------------------------------------------------------------

from committor._types import Array, ScalarFn, CommittorResult

from committor.basis import (
    UnivariateBasis,
    TensorProductBasis,
    shifted_orthonormal_legendre_basis,
    fourier_basis,
    tensor_product_legendre_basis,
    density_weighted_orthogonal_basis,
    double_well_density_weighted_basis,
)

from committor.tensor_train import (
    TTTrain,
    MPOTrain,
    tt_from_dense,
    tt_evaluate,
    tt_inner,
    mpo_inner,
)

from committor.assembly import (
    PerDimMatrices,
    quadrature_matrices_1d,
    quadrature_matrices_nd,
    assemble_dense_nd,
    assemble_mpo_rank1,
    assemble_hb_tt,
)

from committor.solvers import (
    Committor1DResult,
    CommittorNDDenseResult,
    CommittorTTResult,
    solve_committor_1d,
    solve_committor_nd_dense,
    solve_committor_nd_tt,
)

from committor.problems import (
    double_well_potential,
    double_well_nd_measure_fns,
    build_double_well_nd_problem,
    build_double_well_nd_problem_weighted,
    exact_committor_1d,
    lift_to_d_dimensions,
    relative_error_1d,
    relative_error_nd_mc,
)

__all__ = [
    # types
    "Array", "ScalarFn", "CommittorResult",
    # basis
    "UnivariateBasis", "TensorProductBasis",
    "shifted_orthonormal_legendre_basis", "fourier_basis",
    "tensor_product_legendre_basis", "density_weighted_orthogonal_basis",
    "double_well_density_weighted_basis",
    # tensor train
    "TTTrain", "MPOTrain",
    "tt_from_dense", "tt_evaluate", "tt_inner", "mpo_inner",
    # assembly
    "PerDimMatrices",
    "quadrature_matrices_1d", "quadrature_matrices_nd",
    "assemble_dense_nd", "assemble_mpo_rank1", "assemble_hb_tt",
    # solvers
    "Committor1DResult", "CommittorNDDenseResult", "CommittorTTResult",
    "solve_committor_1d", "solve_committor_nd_dense", "solve_committor_nd_tt",
    # problems
    "double_well_potential", "double_well_nd_measure_fns",
    "build_double_well_nd_problem", "build_double_well_nd_problem_weighted",
    "exact_committor_1d", "lift_to_d_dimensions",
    "relative_error_1d", "relative_error_nd_mc",
]