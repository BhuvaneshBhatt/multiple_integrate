"""Wrong-result regressions for principal values and analytic shortcuts."""

import pytest
import sympy as sp

from multiple_integrate import integrate
from multiple_integrate.region_types import AffineSimplexRegion

x, a = sp.symbols("x a", real=True)


@pytest.mark.parametrize(
    "expr,expected", [(sp.sqrt(x), sp.Rational(2, 3) * (1 + sp.I)), (sp.log(x), -2 + sp.I * sp.pi)]
)
def test_pv_preserves_branch_sensitive_complex_integrals(expr, expected):
    assert sp.simplify(integrate(expr, (x, -1, 1), principal_value=True) - expected) == 0


@pytest.mark.parametrize("singular", [False, True])
def test_pv_preserves_unknown_complex_coefficient(singular):
    coefficient = sp.Symbol("c")
    expr = coefficient / (x - sp.Rational(1, 2)) if singular else coefficient
    expected = -coefficient * sp.log(3) if singular else 2 * coefficient
    assert sp.simplify(integrate(expr, (x, -1, 1), principal_value=True) - expected) == 0


@pytest.mark.parametrize("dimension", [1, 2])
def test_affine_dirichlet_retains_internal_looking_parameter(dimension):
    parameter = sp.Symbol("_u0", real=True)
    variables = sp.symbols(f"x0:{dimension}", real=True)
    region = AffineSimplexRegion(
        tuple((v, 0, 1) for v in variables),
        shifts=(0,) * dimension,
        scales=(1,) * dimension,
        dimension=dimension,
    )
    expected = parameter * sp.gamma(sp.Rational(3, 2)) / sp.gamma(sp.Rational(3, 2) + dimension)
    assert sp.simplify(integrate(parameter * sp.sqrt(variables[0]), region=region) - expected) == 0


@pytest.mark.parametrize(
    "quadratic,constant,numerator", [(4, 1, 1), (4, 9, 1), (9, 4, -3), (-4, -1, 1), (1, 4, 2)]
)
def test_full_line_rational_scale(quadratic, constant, numerator):
    expected = sp.pi * numerator / (quadratic * sp.sqrt(sp.Rational(constant, quadratic)))
    assert integrate(numerator / (quadratic * x * x + constant), (x, -sp.oo, sp.oo)) == expected


@pytest.mark.parametrize("function", [sp.exp, sp.sin, sp.cos])
def test_zero_rate_parameter_cell(function):
    result = integrate(function(a * x + 1), (x, 0, 2))
    assert sp.simplify(result.subs(a, 0) - 2 * function(1)) == 0
    assert sp.simplify(result.subs(a, 2) - sp.integrate(function(2 * x + 1), (x, 0, 2))) == 0


@pytest.mark.parametrize("function", [sp.exp, sp.sin, sp.cos])
def test_nonzero_rate_keeps_elementary_formula(function):
    rate = sp.Symbol("rate", positive=True)
    result = integrate(function(rate * x), (x, 0, 1))
    assert not result.has(sp.Piecewise)
    assert sp.simplify(result - sp.integrate(function(rate * x), (x, 0, 1))) == 0
