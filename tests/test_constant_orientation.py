"""Explicit orientation and geometric box measure must be applied exactly once."""

import pytest
import sympy as sp

from multiple_integrate import IntegrationPolicy, IntegrationStatus, integrate, integrate_result
from multiple_integrate.region_types import BoxRegion

x, y, a = sp.symbols("x y a", real=True)
policy = IntegrationPolicy(max_singular_queries=0)


@pytest.mark.parametrize("upper", [-2, -1, 1, 2])
def test_constant_explicit_interval_retains_signed_width(upper):
    assert integrate(3, (x, 0, upper)) == 3 * upper


@pytest.mark.parametrize("ux,uy", [(-2, 3), (2, -3), (-2, -3)])
def test_constant_product_orientation_is_not_applied_twice(ux, uy):
    assert integrate(2, (x, 0, ux), (y, 0, uy)) == 2 * ux * uy


def test_symbolic_reversed_constant_width_and_late_constant_reduction():
    assert integrate(1, (x, 0, 1 / a), assumptions=a < 0) == 1 / a
    assert integrate(1, (x, 0, a), assumptions=a < 0) == a
    density = sp.sin(x) ** 2 + sp.cos(x) ** 2
    assert integrate(density, (x, 0, -2)) == -2


@pytest.mark.parametrize(
    "density,expected", [(1, 2), (x, -2), (x * x, sp.Rational(8, 3)), (sp.sin(x), sp.cos(2) - 1)]
)
def test_native_geometric_box_has_consistent_unsigned_domain(density, expected):
    box = BoxRegion(((x, 0, -2),))
    assert box.constant_volume() == 2
    assert sp.simplify(integrate(density, box) - expected) == 0


def test_geometric_box_unknown_sign_uses_minimum_and_maximum_endpoints():
    box = BoxRegion(((x, 0, a),))
    r = integrate_result(x * x, box, policy=policy)
    assert r.value.subs(a, -2) == sp.Rational(8, 3)
    assert r.value.subs(a, 2) == sp.Rational(8, 3)
    assert sp.simplify(integrate(1, box).rewrite(sp.Abs) - sp.Abs(a)) == 0


def test_unproved_geometric_real_endpoints_cannot_return_a_contour_value():
    c = sp.Symbol("c", finite=True)
    r = integrate_result(1, BoxRegion(((x, 0, c),)), policy=policy)
    assert r.status is IntegrationStatus.UNKNOWN and r.absolute_convergent is not True
    nonreal = integrate_result(1, BoxRegion(((x, 0, sp.I),)), policy=policy)
    assert nonreal.status is IntegrationStatus.UNKNOWN
    assert integrate(1, (x, 0, sp.I)) == sp.I
