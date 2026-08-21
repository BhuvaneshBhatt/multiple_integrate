"""multiple_integrate
====================

Symbolic definite integration for Python, with an emphasis on structured
multiple integrals.

The main public entry point is ``multiple_integrate``.
"""

from multiple_integrate.core import clear_cache, multiple_integrate
from multiple_integrate.regions import (
    AffineSimplexRegion,
    AnnulusRegion,
    BallRegion,
    BoxRegion,
    DiskRegion,
    EllipsoidRegion,
    GraphRegion,
    IteratedRegion,
    Region,
    SimplexRegion,
    SphericalShellRegion,
    UnionRegion,
    boole,
    region_from_ranges,
)

__all__ = [
    "multiple_integrate",
    "clear_cache",
    "Region",
    "AffineSimplexRegion",
    "AnnulusRegion",
    "BallRegion",
    "BoxRegion",
    "DiskRegion",
    "EllipsoidRegion",
    "GraphRegion",
    "IteratedRegion",
    "SimplexRegion",
    "SphericalShellRegion",
    "UnionRegion",
    "boole",
    "region_from_ranges",
]
