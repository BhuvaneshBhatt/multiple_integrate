"""Structured region models, recognition, and Boolean restrictions."""

from multiple_integrate.region_formulas import Boole, indicator_condition, restrict_region
from multiple_integrate.region_recognition import (
    match_affine_simplex,
    match_graph_region,
    match_standard_ball,
    match_standard_disk,
    match_standard_ellipsoid,
    match_standard_simplex,
    region_from_ranges,
)

__all__ = [
    "match_standard_simplex",
    "match_affine_simplex",
    "match_graph_region",
    "match_standard_disk",
    "match_standard_ball",
    "match_standard_ellipsoid",
    "Boole",
    "indicator_condition",
    "restrict_region",
    "region_from_ranges",
]
