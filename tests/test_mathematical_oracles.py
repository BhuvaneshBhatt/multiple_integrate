import sympy as sp

from multiple_integrate import integrate
from multiple_integrate.region_types import BoxRegion


def test_box_moment_matches_product_formula():
    x, y = sp.symbols("x y", real=True)
    value = integrate(x**2 * y**4, (x, -2, 3), (y, 1, 4))
    oracle = ((3**3 - (-2) ** 3) / sp.Integer(3)) * ((4**5 - 1) / sp.Integer(5))
    assert sp.simplify(value - oracle) == 0


def test_simplex_monomial_matches_dirichlet_formula():
    x, y = sp.symbols("x y", nonnegative=True)
    value = integrate(x**2 * y, (y, 0, 1 - x), (x, 0, 1))
    oracle = sp.factorial(2) * sp.factorial(1) / sp.factorial(2 + 1 + 2)
    assert value == oracle


def test_ball_second_moment_matches_isotropy_formula():
    x, y, z = sp.symbols("x y z", real=True)
    value = integrate(
        x**2,
        (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)),
        (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
        (x, -1, 1),
    )
    # By rotational symmetry, three equal coordinate moments sum to ∫_B r^2 dV = 4π/5.
    assert value == 4 * sp.pi / 15


def test_box_region_and_ranges_are_equivalent():
    x, y = sp.symbols("x y", real=True)
    ranges = ((x, -1, 2), (y, 0, 3))
    region = BoxRegion(ranges)
    expr = x**2 + x * y + 2 * y
    assert integrate(expr, *ranges) == integrate(expr, region=region)


def test_variable_permutation_preserves_product_integral():
    x, y = sp.symbols("x y", real=True)
    xy = integrate(x**2 * y**3, (x, 0, 2), (y, -1, 1))
    yx = integrate(y**2 * x**3, (y, 0, 2), (x, -1, 1))
    assert xy == yx


def test_affine_translation_matches_change_of_variables_oracle():
    x, y = sp.symbols("x y", real=True)
    translated = integrate((x - 2) ** 2 + (y + 3), (x, 2, 3), (y, -3, -1))
    u, v = sp.symbols("u v", real=True)
    reference = integrate(u**2 + v, (u, 0, 1), (v, 0, 2))
    assert translated == reference
