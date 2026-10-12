"""Independent integration paths must agree on shared mathematical cases."""

import semialg
import sympy as sp

from multiple_integrate import integrate, integrate_result, region_from_ranges


def test_box_ranges_geometry_formula_agree():
    x, y = sp.symbols("x y", real=True)
    expression = x**2 + x * y + 2 * y
    ranges = ((x, -1, 2), (y, 0, 3))
    geometry = semialg.Box(((-1, 2), (0, 3)))
    formula = geometry.as_semialgebraic_region((x, y)).quantifier_free_formula()

    by_ranges = integrate_result(expression, *ranges)
    by_geometry = integrate_result(expression, region=geometry, variables=(x, y))
    by_formula = integrate_result(expression, region=formula, variables=(x, y))

    assert by_ranges.method == "region_shortcut"
    assert by_geometry.method == "semialgebraic_domain_reduction"
    assert by_formula.method == "semialgebraic_domain_reduction"
    assert sp.simplify(by_ranges.value - by_geometry.value) == 0
    assert sp.simplify(by_ranges.value - by_formula.value) == 0


def test_disk_ranges_and_geometry_agree():
    x, y = sp.symbols("x y", real=True)
    expression = x**2 + y**2
    ranges = ((y, -sp.sqrt(4 - x**2), sp.sqrt(4 - x**2)), (x, -2, 2))
    recognized = region_from_ranges(ranges)
    geometry = semialg.Ball((0, 0), 2)

    by_ranges = integrate(expression, region=recognized)
    by_geometry = integrate(expression, region=geometry, variables=(x, y))

    assert by_ranges == 8 * sp.pi
    assert by_geometry == by_ranges


def test_simplex_ranges_and_geometry_agree():
    x, y = sp.symbols("x y", real=True)
    expression = x + y
    ranges = ((y, 0, 1 - x), (x, 0, 1))
    recognized = region_from_ranges(ranges)
    geometry = semialg.Simplex(((0, 0), (1, 0), (0, 1)))

    by_ranges = integrate(expression, region=recognized)
    by_geometry = integrate(expression, region=geometry, variables=(x, y))

    assert by_ranges == sp.Rational(1, 3)
    assert by_geometry == by_ranges


def test_intrinsic_circle_matches_explicit_parameter_integral():
    x, y, theta = sp.symbols("x y theta", real=True)
    circle = semialg.Sphere((0, 0), 2)

    intrinsic = integrate(
        x**2 + y**2,
        region=circle,
        variables=(x, y),
        measure_dimension="intrinsic",
    )
    parameterized = integrate(8, (theta, 0, 2 * sp.pi))

    assert intrinsic == parameterized == 16 * sp.pi
