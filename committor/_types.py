"""Type aliases and shared protocols used throughout the committor package."""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Array = np.ndarray
"""Shorthand for numpy ndarray."""

ScalarFn = Callable[[Array], Array]
"""A function mapping an array of x-values to an array of function values."""


# ---------------------------------------------------------------------------
# CommittorResult — minimal interface shared by every solver backend
# ---------------------------------------------------------------------------

@runtime_checkable
class CommittorResult(Protocol):
    """Protocol satisfied by all committor result objects.

    Any object with a ``q`` method satisfies this protocol, so both the
    1-D dense result (Committor1DResult), the n-D dense result
    (CommittorNDDenseResult), and the TT result (CommittorTTResult) are
    automatically compatible with code written against this interface.

    The only requirement is::

        result.q(x)  ->  committor values at the given points

    Shape convention for ``x`` is left open: the 1-D solver takes a flat
    array of shape ``(n_pts,)`` while the n-D solvers expect ``(n_samples, d)``.
    """

    def q(self, x: Array) -> Array:
        """Evaluate the committor approximation at the given points."""
        ...