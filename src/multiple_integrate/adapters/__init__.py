"""Optional backend adapters used by the integration planner."""

from .funcprops import (
    funcprops_available,
    function_domain,
    function_injective,
    function_monotonicity,
    function_range,
    function_sign,
    function_singularities,
    function_symmetry,
)
from .semialg import (
    IntrinsicPiece,
    IntrinsicReduction,
    reduce_domain,
    reduce_intrinsic_domain,
    region_condition,
    region_moment,
    semialg_available,
    semialgebraic_domain,
    semialgebraic_measure,
)

__all__ = [
    "funcprops_available",
    "function_domain",
    "function_injective",
    "function_monotonicity",
    "function_range",
    "function_sign",
    "function_singularities",
    "function_symmetry",
    "IntrinsicPiece",
    "IntrinsicReduction",
    "reduce_intrinsic_domain",
    "reduce_domain",
    "region_condition",
    "region_moment",
    "semialg_available",
    "semialgebraic_domain",
    "semialgebraic_measure",
]
