"""Graph reversal must retain exact boundary ordering and oriented semantics."""

import pytest
import sympy as sp

from multiple_integrate import integrate
from multiple_integrate.region_types import GraphRegion

x, y = sp.symbols("x y", real=True)


@pytest.mark.parametrize("width", [sp.Rational(1, 10**20), sp.Rational(1, 10**100), 1, 10**20])
@pytest.mark.parametrize("origin", [0, 10**40])
def test_tiny_and_translated_triangles_have_exact_reversed_moments(width, origin):
    region = GraphRegion(((y, 0, x - origin), (x, origin, origin + width)))
    pieces = region.reversed_pieces()
    assert pieces is not None
    for density in [sp.S.One, x - origin, y]:
        original = sp.integrate(density, *region.ranges)
        reversed_ = sum(sp.integrate(density, *piece) for piece in pieces)
        assert sp.simplify(original - reversed_) == 0
    assert pieces[0][0][1] == origin + y


@pytest.mark.parametrize("lo,hi", [(x, 1 - x), (2 - x, x), (2, 1), (0, x)])
def test_crossing_and_negative_bounds_decline_geometric_reversal(lo, hi):
    region = GraphRegion(((y, lo, hi), (x, -1, 1)))
    assert region.reversed_pieces() is None


@pytest.mark.parametrize("slope", [sp.Rational(1, 10**30), -sp.Rational(1, 10**30), sp.sqrt(2)])
def test_small_or_algebraic_slopes_use_exact_comparisons(slope):
    region = GraphRegion(((y, slope * x, slope * x + 1), (x, 0, 1)))
    pieces = region.reversed_pieces()
    assert pieces is not None
    for density in [sp.S.One, x, y]:
        assert (
            sp.simplify(
                sum(sp.integrate(density, *r) for r in pieces)
                - sp.integrate(density, *region.ranges)
            )
            == 0
        )


def test_crossing_graph_public_integral_preserves_signed_parts():
    # A geometric reversal would incorrectly retain only a positive triangle.
    density = y
    assert integrate(density, (y, x, 1 - x), (x, 0, 1)) == 0
