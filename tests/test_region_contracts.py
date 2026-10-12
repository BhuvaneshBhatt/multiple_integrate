"""Contracts connecting geometric regions, integration, measure, and sampling."""

import math

import pytest
import semialg
import sympy as sp

from multiple_integrate import integrate, region_from_ranges


@pytest.mark.parametrize(
    ("region", "variables"),
    [
        (semialg.Box(((0, 2), (-1, 3))), sp.symbols("x y", real=True)),
        (semialg.Ball((0, 0), 2), sp.symbols("x y", real=True)),
        (
            semialg.Simplex(((0, 0), (1, 0), (0, 1))),
            sp.symbols("x y", real=True),
        ),
        (
            semialg.Ellipsoid((0, 0), ((4, 0), (0, 9))),
            sp.symbols("x y", real=True),
        ),
        (
            semialg.SphericalShell((0, 0), (1, 2)),
            sp.symbols("x y", real=True),
        ),
    ],
)
def test_measure_matches_constant_integral(region, variables):
    assert sp.simplify(integrate(1, region=region, variables=variables) - region.measure()) == 0


@pytest.mark.parametrize(
    ("region", "variables"),
    [
        (semialg.Box(((0, 2), (-1, 3))), sp.symbols("x y", real=True)),
        (semialg.Ball((1, -2), 3), sp.symbols("x y", real=True)),
        (
            semialg.Simplex(((0, 0), (2, 0), (0, 4))),
            sp.symbols("x y", real=True),
        ),
        (
            semialg.Ellipsoid((1, 2), ((4, 0), (0, 9))),
            sp.symbols("x y", real=True),
        ),
    ],
)
def test_centroid_matches_first_moments(region, variables):
    measure = region.measure()
    expected = tuple(
        sp.simplify(integrate(v, region=region, variables=variables) / measure) for v in variables
    )
    assert tuple(map(sp.simplify, region.centroid())) == expected


def test_range_recognition_matches_geometry():
    # The concrete contracts are assembled below so each expression shares the
    # same Symbol instances as its integration ranges.
    x, y = sp.symbols("x y", real=True)
    cases = [
        (
            [(y, -1, 3), (x, 0, 2)],
            semialg.Box(((0, 2), (-1, 3))),
            x + 2 * y,
        ),
        (
            [(y, -sp.sqrt(4 - x**2), sp.sqrt(4 - x**2)), (x, -2, 2)],
            semialg.Ball((0, 0), 2),
            x**2 + y**2,
        ),
        (
            [(y, 0, 1 - x), (x, 0, 1)],
            semialg.Simplex(((0, 0), (1, 0), (0, 1))),
            x + y,
        ),
    ]
    for bounds, canonical, expression in cases:
        domain = region_from_ranges(bounds)
        by_ranges = integrate(expression, region=domain)
        by_geometry = integrate(expression, region=canonical, variables=(x, y))
        assert sp.simplify(by_ranges - by_geometry) == 0


def _contains(region, point, variables):
    symbolic = region.as_semialgebraic_region(variables)
    condition = symbolic.quantifier_free_formula()
    return bool(condition.subs(dict(zip(variables, point, strict=True))))


@pytest.mark.parametrize(
    "region",
    [
        semialg.Box(((-2, 3), (1, 4))),
        semialg.Ball((0, 0), 2),
        semialg.Simplex(((0, 0), (1, 0), (0, 1))),
        semialg.Ellipsoid((0, 0), ((4, 0), (0, 9))),
        semialg.SphericalShell((0, 0), (1, 2)),
    ],
)
def test_random_sampling_is_seeded_and_inside(region):
    variables = sp.symbols(f"x0:{region.ambient_dimension()}", real=True)
    points = region.random_points(25, seed=1234)
    assert points == region.random_points(25, seed=1234)
    assert len(points) == 25
    assert all(_contains(region, point, variables) for point in points)


def test_disk_sampling_moments():
    region = semialg.Ball((0, 0), 2)
    points = region.random_points(2000, seed=20260909)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    r2 = [x * x + y * y for x, y in zip(xs, ys, strict=True)]

    assert abs(sum(xs) / len(xs)) < 0.08
    assert abs(sum(ys) / len(ys)) < 0.08
    assert math.isclose(sum(r2) / len(r2), 2.0, abs_tol=0.08)


def test_simplex_sampling_barycenter():
    region = semialg.Simplex(((0, 0), (1, 0), (0, 1)))
    points = region.random_points(2000, seed=314159)
    mean_x = sum(float(point[0]) for point in points) / len(points)
    mean_y = sum(float(point[1]) for point in points) / len(points)

    assert math.isclose(mean_x, 1 / 3, abs_tol=0.025)
    assert math.isclose(mean_y, 1 / 3, abs_tol=0.025)
