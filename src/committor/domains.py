"""Format selection between chebdGE's TT and Tucker representations.

Per this project's instructions, Tucker is preferred in the 3-7 dimensional
band (its dense core is still cheap there) while TT is preferred once the
dimension grows past that, since its cost is only linear in ``d``.
"""

from __future__ import annotations

from typing import Literal

Format = Literal["tt", "tucker"]

_TUCKER_MIN_DIM = 3
_TUCKER_MAX_DIM = 7


def recommended_format(d: int) -> Format:
    """Return ``"tucker"`` for 3 <= d <= 7 and ``"tt"`` otherwise.

    Callers are always free to override this by passing an explicit
    ``format="tt"``/``format="tucker"`` to the ansatz/training helpers instead
    of ``format="auto"``.
    """
    if d < 1:
        raise ValueError(f"dimension must be >= 1, got {d}")
    if _TUCKER_MIN_DIM <= d <= _TUCKER_MAX_DIM:
        return "tucker"
    return "tt"


def resolve_format(d: int, format: Format | Literal["auto"] = "auto") -> Format:
    """Resolve ``format="auto"`` via :func:`recommended_format`, else pass through."""
    if format == "auto":
        return recommended_format(d)
    if format not in ("tt", "tucker"):
        raise ValueError(f"format must be 'auto', 'tt', or 'tucker', got {format!r}")
    return format
