"""Exact selected families checked against independent quadrature and broken assumptions."""

import mpmath as mp
import pytest
import sympy as sp

from multiple_integrate import IntegrationPolicy, integrate_result
from multiple_integrate.convergence_evidence import absolute_rule
from multiple_integrate.core import clear_cache
from multiple_integrate.selected_families import selected_interval_rule

x, y, a, p, q = sp.symbols("x y a p q", real=True)
policy = IntegrationPolicy(max_singular_queries=0)


@pytest.fixture(autouse=True)
def uncached_results():
    clear_cache()


@pytest.mark.parametrize("degree", [1, 3, 4, 6])
@pytest.mark.parametrize("power", [sp.Rational(-1, 2), 0, 2])
def test_laplace_moments_have_independent_quadrature(degree, power):
    f = x**power * sp.exp(-2 * x**degree)
    r = integrate_result(f, (x, 0, sp.oo), policy=policy)
    assert r.absolute_convergent is True
    with mp.workdps(35):
        value = mp.mpf(str(r.value.evalf(30)))
        oracle = mp.quad(sp.lambdify(x, f, "mpmath"), [0, 1, mp.inf])
        assert abs(value - oracle) < mp.mpf("1e-17")


@pytest.mark.parametrize("degree", [4, 6])
@pytest.mark.parametrize("power", [0, 1, 2, 3])
def test_full_even_laplace_proves_absolute_before_odd_cancellation(degree, power):
    f = x**power * sp.exp(-a * x**degree)
    method, value = selected_interval_rule(f, x, -sp.oo, sp.oo, a > 0)
    assert method == "generalized_laplace_moments"
    expected = (
        0
        if power % 2
        else 2
        * sp.gamma(sp.Rational(power + 1, degree))
        / (degree * a ** sp.Rational(power + 1, degree))
    )
    assert sp.simplify(value - expected) == 0
    assert absolute_rule(f, ((x, -sp.oo, sp.oo),), assumptions=a > 0)[0] is True
    assert selected_interval_rule(f, x, -sp.oo, sp.oo) is None


@pytest.mark.parametrize("degree", [1, 3, 4])
def test_complex_laplace_rate_uses_right_half_plane(degree):
    f = (1 + x) * sp.exp(-(2 + sp.I) * x**degree)
    r = integrate_result(f, (x, 0, sp.oo), policy=policy)
    assert r.absolute_convergent is True
    with mp.workdps(35):
        oracle = mp.quad(sp.lambdify(x, f, "mpmath"), [0, 1, mp.inf])
        value = complex(r.value.evalf(30))
        assert abs(value - oracle) < mp.mpf("1e-14")
    assert selected_interval_rule(sp.exp(-sp.I * x**degree), x, 0, sp.oo) is None


@pytest.mark.parametrize("lo,hi", [(-2, 3), (0, 2), (1, 4)])
@pytest.mark.parametrize(
    "powers", [(sp.Rational(-1, 2), sp.Rational(-1, 2)), (sp.Rational(1, 2), sp.Rational(-1, 3))]
)
def test_shifted_beta_polynomial_amplitudes(lo, hi, powers):
    p0, q0 = powers
    f = (1 + x * x) * (x - lo) ** p0 * (hi - x) ** q0
    r = integrate_result(f, (x, lo, hi), policy=policy)
    reverse = integrate_result(f, (x, hi, lo), policy=policy)
    assert r.method == "endpoint_beta_moments" and r.absolute_convergent is True
    assert sp.simplify(reverse.value + r.value) == 0
    with mp.workdps(35):
        oracle = mp.quad(sp.lambdify(x, f, "mpmath"), [lo, (lo + hi) / 2, hi])
        assert abs(mp.mpf(str(r.value.evalf(30))) - oracle) < mp.mpf("1e-15")


def test_symbolic_endpoint_and_laplace_prerequisites():
    f = x**p * (1 - x) ** q
    assert selected_interval_rule(f, x, 0, 1, sp.And(p > -1, q > -1)) is not None
    assert selected_interval_rule(f, x, 0, 1, p > -1) is None
    assert selected_interval_rule(f, x, 0, 1, sp.And(p > -1, q <= -1)) is None
    f = x**p * sp.exp(-a * x**3)
    assert selected_interval_rule(f, x, 0, sp.oo, sp.And(a > 0, p > -1)) is not None
    assert selected_interval_rule(f, x, 0, sp.oo, a > 0) is None
    assert selected_interval_rule(f, x, 0, sp.oo, sp.And(a >= 0, p > -1)) is None


@pytest.mark.parametrize("degree", [2, 3, 4, 6])
@pytest.mark.parametrize("power", [sp.Rational(-1, 2), 0, 1])
def test_rational_mellin_moments_and_tail_thresholds(degree, power):
    f = x**power / (2 + 3 * x**degree) ** 2
    r = integrate_result(f, (x, 0, sp.oo), policy=policy)
    assert r.absolute_convergent is True
    with mp.workdps(35):
        oracle = mp.quad(sp.lambdify(x, f, "mpmath"), [0, 1, mp.inf])
        assert abs(mp.mpf(str(r.value.evalf(30))) - oracle) < mp.mpf("1e-17")
    assert (
        selected_interval_rule(x ** (2 * degree - 1) / (2 + 3 * x**degree) ** 2, x, 0, sp.oo)
        is None
    )
    assert selected_interval_rule(x**-1 / (2 + 3 * x**degree) ** 2, x, 0, sp.oo) is None


@pytest.mark.parametrize("bad", [1 / (1 - x**3), 1 / (a + x**3), 1 / (1 + x**3), sp.exp(-(x**3))])
def test_selected_full_line_rules_decline_unproved_branches(bad):
    assert selected_interval_rule(bad, x, -sp.oo, sp.oo) is None


def test_mellin_polynomial_moments_and_orientation():
    f = (1 + x + x * x) / (1 + x**4) ** 2
    method, value = selected_interval_rule(f, x, 0, sp.oo)
    assert method == "rational_mellin_moments"
    assert selected_interval_rule(f, x, sp.oo, 0)[1] == -value
    assert absolute_rule(f, ((x, 0, sp.oo),))[0] is True


@pytest.mark.parametrize("degree", [3, 4])
def test_expansion_and_degree_budgets_precede_selected_formulas(degree):
    from multiple_integrate import StrategyBudget

    budget = IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=2))
    assert selected_interval_rule(sp.exp(-(x**degree)), x, 0, sp.oo, policy=budget) is None


@pytest.mark.parametrize("mixed", [0, x * x * y * y, x**4 * y * y])
def test_coercive_even_exponential_comparison_handles_coupling(mixed):
    f = (1 + x * x + y**4) * sp.exp(-a * x**4 - y**6 - mixed)
    ranges = ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
    assert absolute_rule(f, ranges, assumptions=a > 0)[0] is True
    assert absolute_rule(f, ranges)[0] is not True


@pytest.mark.parametrize(
    "phase", [-(x**4) + y**4, -(x**4), -(x**4) - y**4 + x * x * y * y, -(x**3) - y**4]
)
def test_coercive_comparison_declines_missing_or_broken_decay(phase):
    from multiple_integrate.selected_families import coercive_exponential_absolute

    assert (
        coercive_exponential_absolute(sp.exp(phase), ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo)))
        is None
    )


@pytest.mark.parametrize("power", [sp.Rational(-1, 3), sp.Rational(-1, 2), -1])
def test_higher_degree_tail_majorants_use_actual_growth(power):
    f = (1 + x * x + x**6) ** power
    assert absolute_rule(f, ((x, -sp.oo, sp.oo),))[0] is True
    assert absolute_rule(f, ((x, 0, sp.oo),))[0] is True


def test_coupled_positive_even_tail_and_broken_prerequisites():
    from multiple_integrate.selected_families import positive_even_tail_absolute

    ranges = ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
    f = (1 + x * x) / (1 + x**4 + y**6 + x * x * y * y) ** q
    assert absolute_rule(f, ranges, assumptions=q > sp.Rational(11, 12))[0] is True
    assert positive_even_tail_absolute(f, ranges, q <= sp.Rational(11, 12)) is None
    assert positive_even_tail_absolute(1 / (1 + x**4 + y**6 - x * x * y * y), ranges) is None
    assert positive_even_tail_absolute(1 / (x**4 + y**6), ranges) is None
    assert positive_even_tail_absolute(1 / (1 + x**4 + x * x * y * y), ranges) is None


def test_selected_recognition_respects_coordinate_budget():
    from multiple_integrate import StrategyBudget

    budget = IntegrationPolicy(semialg=StrategyBudget(max_variables=0))
    assert selected_interval_rule(sp.exp(-(x**4)), x, 0, sp.oo, policy=budget) is None
    assert absolute_rule(sp.exp(-(x**4)), ((x, 0, sp.oo),), policy=budget)[0] is None
