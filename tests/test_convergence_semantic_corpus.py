"""Adversarial finite/divergent/conditional/unevaluated semantics."""

import pytest
import sympy as sp

from multiple_integrate import (
    IntegrationPolicy,
    IntegrationStatus,
    StrategyBudget,
    integrate,
    integrate_result,
)

x, y = sp.symbols("x y", real=True)


@pytest.mark.parametrize(
    "f,limits,expected",
    [
        (x ** sp.Rational(-1, 2), (x, 0, 1), 2),
        (sp.log(x), (x, 0, 1), -1),
        (sp.log(1 - x), (x, 0, 1), -1),
        (x ** sp.Rational(1, 2), (x, -1, 0), 2 * sp.I / 3),
        (sp.log(x), (x, -1, 0), -1 + sp.I * sp.pi),
        (sp.cancel((x**2 - 1) / (x - 1)), (x, 0, 2), 4),
    ],
)
def test_finite_and_branch_oracles(f, limits, expected):
    result = integrate_result(f, limits)
    assert sp.simplify(result.value - expected) == 0
    assert result.status is IntegrationStatus.EXACT


@pytest.mark.parametrize("power", [sp.Rational(-1, 2), -1, -2])
def test_boundary_convergence_transition(power):
    result = integrate_result(x**power, (x, 0, 1))
    if power > -1:
        assert result.status is IntegrationStatus.EXACT
        assert result.value == 1 / (power + 1)
    else:
        assert result.status is IntegrationStatus.DIVERGENT


def test_interior_pole_is_not_ordinary_zero():
    ordinary = integrate_result(1 / x, (x, -1, 1))
    pv = integrate_result(1 / x, (x, -1, 1), principal_value=True)
    assert ordinary.status is IntegrationStatus.DIVERGENT
    assert pv.status is IntegrationStatus.PRINCIPAL_VALUE
    assert pv.value == 0


def test_parameter_convergence_strata():
    a = sp.Symbol("a", real=True)
    result = integrate_result(sp.exp(-a * x), (x, 0, sp.oo))
    assert result.status is IntegrationStatus.CONDITIONAL
    assert result.parameter_strata
    assert integrate(sp.exp(-a * x), (x, 0, sp.oo), assumptions=a > 0) == 1 / a
    assert (
        integrate_result(sp.exp(-a * x), (x, 0, sp.oo), assumptions=a <= 0).status
        is IntegrationStatus.DIVERGENT
    )
    assert integrate_result(1, (x, 0, sp.oo)).status is IntegrationStatus.DIVERGENT


def test_budget_unknown_is_not_divergence():
    result = integrate_result(
        1,
        region=sp.And(x >= 0, x <= 1),
        variables=(x,),
        policy=IntegrationPolicy(semialg=StrategyBudget(max_variables=0)),
    )
    assert result.status is IntegrationStatus.UNEVALUATED
    assert result.value != 0


@pytest.mark.parametrize(
    "valid,broken,ranges,expected",
    [
        (sp.exp(-(x**2)), sp.exp(x**2), ((x, -sp.oo, sp.oo),), sp.sqrt(sp.pi)),
        (1 / x**2, 1 / x, ((x, 1, sp.oo),), 1),
        (1 / sp.sqrt(x), 1 / x, ((x, 0, 1),), 2),
        (1 / (x + y), 1 / (x + y) ** 2, ((x, 0, 1), (y, 0, 1)), 2 * sp.log(2)),
    ],
)
def test_exact_route_prerequisite_pairs(valid, broken, ranges, expected):
    good = integrate_result(valid, *ranges)
    bad = integrate_result(broken, *ranges)
    assert good.status is IntegrationStatus.EXACT
    assert sp.simplify(good.value - expected) == 0
    assert bad.status is IntegrationStatus.DIVERGENT


@pytest.mark.slow
def test_conditional_vs_absolute_convergence():
    conditional = integrate_result(sp.sin(x) / x, (x, 0, sp.oo))
    absolute = integrate_result(sp.Abs(sp.sin(x)) / x, (x, 1, sp.oo))
    assert conditional.value == sp.pi / 2
    assert conditional.status is IntegrationStatus.EXACT
    assert absolute.status is IntegrationStatus.DIVERGENT
    assert absolute.complete is True
    assert absolute.ordinary_convergent is False
    assert absolute.absolute_convergent is False


@pytest.mark.parametrize("exponent", [sp.Rational(-1, 2), -1])
def test_simplex_dirichlet_prerequisite(exponent):
    result = integrate_result(x**exponent, (y, 0, 1 - x), (x, 0, 1))
    if exponent > -1:
        assert result.value == sp.Rational(4, 3)
        assert result.status is IntegrationStatus.EXACT
    else:
        assert result.status is IntegrationStatus.DIVERGENT


def test_quadratic_gaussian_rank_prerequisite():
    full_rank = integrate_result(sp.exp(-(x**2) - y**2), (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
    rank_drop = integrate_result(sp.exp(-(x**2)), (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
    assert full_rank.value == sp.pi
    assert full_rank.status is IntegrationStatus.EXACT
    assert rank_drop.status is IntegrationStatus.DIVERGENT


@pytest.mark.parametrize("power,expected", [(sp.Rational(-1, 2), 2 * sp.pi), (-1, sp.oo)])
def test_radial_convergence_prerequisite(power, expected):
    import semialg

    result = integrate_result(
        (x**2 + y**2) ** power, region=semialg.Ball((0, 0), 1), variables=(x, y)
    )
    assert result.value == expected
    assert result.status is (IntegrationStatus.EXACT if power > -1 else IntegrationStatus.DIVERGENT)
