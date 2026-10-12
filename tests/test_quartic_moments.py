"""Shifted quartic moments have independent numeric and prerequisite oracles."""

import mpmath as mp
import pytest
import sympy as sp

from multiple_integrate import IntegrationPolicy, integrate_result
from multiple_integrate.fast_rules import quartic_rational_moment

x, a, b = sp.symbols("x a b", real=True)
policy = IntegrationPolicy(max_singular_queries=0)


@pytest.fixture(autouse=True)
def uncached_rule_results():
    from multiple_integrate.core import clear_cache

    clear_cache()


@pytest.mark.parametrize("shift", [-2, 0, 1, 3])
@pytest.mark.parametrize("numerator", [1, x, x * x + sp.I])
def test_shifted_quartic_rule_matches_quadrature(shift, numerator):
    f = numerator / (1 + (x * x + 2 * x + shift) ** 2)
    r = integrate_result(f, (x, -sp.oo, sp.oo), policy=policy)
    assert r.method == "quartic_rational_moments"
    assert r.absolute_convergent is True
    with mp.workdps(40):
        oracle = mp.quad(sp.lambdify(x, f, "mpmath"), [-mp.inf, -3, -1, 0, 2, mp.inf])
        value = mp.mpc(str(sp.re(r.value).evalf(35)), str(sp.im(r.value).evalf(35)))
        assert abs(value - oracle) < mp.mpf("1e-30")


def test_symbolic_positive_quartic_never_inherits_backend_zero():
    r = integrate_result(1 / (1 + (x * x + 2 * x + b) ** 2), (x, -sp.oo, sp.oo), policy=policy)
    assert r.method == "quartic_rational_moments" and r.absolute_convergent is True
    assert r.value != 0
    assert sp.simplify(r.value.subs(b, 1) - sp.pi / sp.sqrt(2)) == 0
    assert all(r.value.subs(b, c).is_positive is True for c in [-2, 0, 1, 3])


@pytest.mark.parametrize(
    "density", [1 / (x**4 - 1), 1 / (x**4 - 2 * x * x + 1), x**3 / (1 + x**4), 1 / (x**4 + x + 2)]
)
def test_quartic_rule_declines_poles_conditional_tails_and_non_biquadratics(density):
    assert quartic_rational_moment(density, x, -sp.oo, sp.oo) is None


def test_quartic_leading_coefficient_and_orientation_requirements():
    f = 1 / (a * x**4 + 1)
    assert quartic_rational_moment(f, x, -sp.oo, sp.oo) is None
    good = quartic_rational_moment(f, x, -sp.oo, sp.oo, assumptions=a > 0)
    assert sp.simplify(good - sp.pi / (sp.sqrt(2) * a ** sp.Rational(1, 4))) == 0
    assert quartic_rational_moment(-1 / (1 + x**4), x, sp.oo, -sp.oo) == sp.pi / sp.sqrt(2)
    assert quartic_rational_moment(1 / (1 + x**4), x, 0, sp.oo) is None
