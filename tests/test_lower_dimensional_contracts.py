"""Contracts for intrinsic integration on lower-dimensional geometry."""

import pytest
import semialg
import sympy as sp

from multiple_integrate import integrate


@pytest.mark.parametrize(
    ("region", "variables", "intrinsic_measure", "ambient_measure"),
    [
        (semialg.Point((2, 3)), sp.symbols("x y", real=True), sp.Integer(1), sp.Integer(0)),
        (
            semialg.Simplex(((0, 0), (3, 4))),
            sp.symbols("x y", real=True),
            sp.Integer(5),
            sp.Integer(0),
        ),
        (
            semialg.Simplex(((0, 0, 0), (1, 0, 0), (0, 1, 0))),
            sp.symbols("x y z", real=True),
            sp.Rational(1, 2),
            sp.Integer(0),
        ),
        (
            semialg.Sphere((0, 0), 2),
            sp.symbols("x y", real=True),
            4 * sp.pi,
            sp.Integer(0),
        ),
        (
            semialg.Sphere((0, 0, 0), 2),
            sp.symbols("x y z", real=True),
            16 * sp.pi,
            sp.Integer(0),
        ),
    ],
)
def test_measure_dimension_matrix(region, variables, intrinsic_measure, ambient_measure):
    intrinsic = integrate(
        1,
        region=region,
        variables=variables,
        measure_dimension="intrinsic",
    )
    ambient = integrate(
        1,
        region=region,
        variables=variables,
        measure_dimension="ambient",
    )
    assert sp.simplify(intrinsic - intrinsic_measure) == 0
    assert ambient == ambient_measure


def test_embedded_segment_first_moment():
    x, y = sp.symbols("x y", real=True)
    segment = semialg.Simplex(((0, 0), (3, 4)))
    value = integrate(
        x + y,
        region=segment,
        variables=(x, y),
        measure_dimension="intrinsic",
    )
    assert value == sp.Rational(35, 2)


def test_embedded_triangle_first_moment():
    x, y, z = sp.symbols("x y z", real=True)
    triangle = semialg.Simplex(((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    value = integrate(
        x + 2 * y + z,
        region=triangle,
        variables=(x, y, z),
        measure_dimension=2,
    )
    assert value == sp.Rational(1, 2)


def test_point_integral_uses_counting_measure():
    x, y = sp.symbols("x y", real=True)
    point = semialg.Point((2, 3))
    value = integrate(
        x**2 + y,
        region=point,
        variables=(x, y),
        measure_dimension=0,
    )
    assert value == 7
