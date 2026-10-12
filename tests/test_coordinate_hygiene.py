"""Coordinate changes preserve external parameters and theorem prerequisites."""

import pytest
import sympy as sp

from multiple_integrate.analytic_region_formulas import _radial_ball_integral
from multiple_integrate.quadratic_gaussian import _integrate_quadratic
from multiple_integrate.region_types import (
    AffineSimplexRegion,
    BallRegion,
    DiskRegion,
    EllipsoidRegion,
)
from multiple_integrate.transforms import CoordinateTransform, _polar_disk_transform, _try_transform

x, y, z = sp.symbols("x y z", real=True)


def test_coordinate_swap_is_simultaneous():
    transform = CoordinateTransform((x, y), (y, x), (y, x), sp.Integer(1), ())
    assert transform.apply(x - y) == y - x


@pytest.mark.parametrize("name", ["_ball_u0", "_u0", "_u1"])
def test_ellipsoid_moment_preserves_named_parameters(name):
    parameter = sp.Symbol(name, real=True)
    region = EllipsoidRegion(variables_nd=(x, y), axes=(1, 1))
    assert region.polynomial_moment(parameter) == sp.pi * parameter
    assert region.polynomial_moment(parameter * x**2) == sp.pi * parameter / 4


def test_affine_simplex_preserves_named_parameter():
    parameter = sp.Symbol("_u0", real=True)
    region = AffineSimplexRegion(((x, 0, 1), (y, 0, 1)), shifts=(0, 0), scales=(1, 1), dimension=2)
    assert region.polynomial_moment(parameter) == parameter / 2


@pytest.mark.parametrize(
    "name,assumptions", [("_r", {"nonnegative": True}), ("_theta", {"real": True})]
)
def test_polar_transform_does_not_capture_parameters(name, assumptions):
    parameter = sp.Symbol(name, **assumptions)
    region = DiskRegion(((x, -1, 1), (y, -1, 1)), radius=1)
    transform = _polar_disk_transform(region)
    assert parameter not in transform.target_vars
    assert (
        _try_transform(transform, parameter, assumptions=None, principal_value=False)
        == parameter * sp.pi
    )


@pytest.mark.parametrize("name", ["_rho2", "_r"])
def test_radial_formula_does_not_capture_parameters(name):
    parameter = sp.Symbol(name, nonnegative=True, real=True)
    value = _radial_ball_integral(parameter * sp.exp(-x * x - y * y), (x, y), (0, 0), sp.Integer(1))
    assert sp.simplify(value - parameter * sp.pi * (1 - sp.exp(-1))) == 0


def test_ball_moment_preserves_coordinate_like_parameter():
    parameter = sp.Symbol("_ball_u0", real=True)
    region = BallRegion(((x, -1, 1), (y, -1, 1), (z, -1, 1)), radius=1, dimension=3)
    assert region.polynomial_moment(parameter) == 4 * sp.pi * parameter / 3


@pytest.mark.parametrize(
    "matrix",
    [sp.diag(sp.Symbol("a", real=True), sp.Symbol("a", real=True)), sp.diag(-1, -1), sp.diag(1, 0)],
)
def test_quadratic_shortcut_requires_positive_definiteness(matrix):
    assert (
        _integrate_quadratic(lambda v: sp.exp(-v), matrix, sp.zeros(2, 1), sp.Integer(0), 2, {})
        is None
    )


def test_positive_quadratic_shortcut_is_retained():
    a = sp.Symbol("a", positive=True)
    assert (
        _integrate_quadratic(
            lambda v: sp.exp(-v), sp.diag(a, a), sp.zeros(2, 1), sp.Integer(0), 2, {}
        )
        == sp.pi / a
    )


@pytest.mark.parametrize("sign", ["positive", "nonnegative", "negative", "nonpositive"])
def test_assumed_coordinate_bounds_are_not_dropped(sign):
    from multiple_integrate import integrate

    coordinate = sp.Symbol("signed_coordinate", **{sign: True})
    lower, upper = (0, 1) if sign in ("positive", "nonnegative") else (-1, 0)
    assert integrate(coordinate, (x, 0, 2), (coordinate, lower, upper)) == (upper**2 - lower**2)
