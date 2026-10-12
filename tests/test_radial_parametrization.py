from __future__ import annotations

import sympy as sp

from multiple_integrate.radial_parametrization import radial_parametrization
from multiple_integrate.sphere_integrals import (
    radial_sphere_reduction,
    sphere_monomial_integral,
    sphere_surface_area,
)


def test_radial_parametrization_preserves_radius_in_four_dimensions():
    variables = sp.symbols("x0:4", real=True)
    chart = radial_parametrization(variables, (0, 0, 0, 0))
    mapping = dict(chart.mapping)
    squared_radius = sp.trigsimp(sum(mapping[var] ** 2 for var in variables))
    assert sp.simplify(sp.expand_trig(squared_radius) - chart.radial_variable**2) == 0
    theta1, theta2, _theta3 = chart.angular_variables
    expected = chart.radial_variable**3 * sp.sin(theta1) ** 2 * sp.sin(theta2)
    assert sp.simplify(chart.jacobian_factor - expected) == 0


def test_sphere_moments_are_dimension_independent():
    assert sphere_surface_area(4) == 2 * sp.pi**2
    assert sphere_monomial_integral((1, 0, 0, 0)) == 0
    assert sp.simplify(sphere_monomial_integral((2, 0, 0, 0)) - sp.pi**2 / 2) == 0


def test_radial_sphere_reduction_handles_borderline_odd_pv_without_angles():
    variables = sp.symbols("x0:5", real=True)
    radial = sp.symbols("r", positive=True, real=True)
    radius_sq = sum(var**2 for var in variables)
    reduced = radial_sphere_reduction(
        variables[0] / radius_sq**3,
        variables,
        (0, 0, 0, 0, 0),
        radial,
    )
    assert reduced == 0


def test_radial_parametrization_supports_rotated_affine_ellipsoid():
    x, y = sp.symbols("x y", real=True)
    shape = sp.ImmutableMatrix([[2, 1], [1, 2]])
    affine = sp.ImmutableMatrix(shape.cholesky())
    chart = radial_parametrization((x, y), (0, 0), affine_matrix=affine)
    mapping = dict(chart.mapping)
    transformed = sp.ImmutableMatrix([mapping[x], mapping[y]])
    quadratic = sp.simplify((transformed.T * shape.inv() * transformed)[0])
    assert sp.trigsimp(quadratic - chart.radial_variable**2) == 0
    assert sp.simplify(chart.jacobian_factor - sp.sqrt(3) * chart.radial_variable) == 0
