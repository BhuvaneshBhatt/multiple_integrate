"""Reproductions of result-certification and domain-boundary bugs."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import semialg
import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    IntegrationPolicy,
    IntegrationStatus,
    StrategyBudget,
    UnevaluatedRegionIntegral,
    integrate,
    integrate_result,
    regularized_result,
)
from multiple_integrate.adapters.semialg import (
    SingularStratum,
    reduce_intrinsic_domain,
    semialgebraic_domain,
)
from multiple_integrate.conditional import conditional_status
from multiple_integrate.convergence import conditional_cases
from tests.test_intrinsic_semialgebraic import _regular_reduction

x, y, a, b, c = sp.symbols("x y a b c", real=True)


@pytest.mark.parametrize(
    "f,ranges",
    [
        (x**a * (1 - x) ** b, ((x, 0, 1),)),
        (x**a * sp.exp(-b * x), ((x, 0, sp.oo),)),
    ],
)
def test_partial_theorem_does_not_claim_complete(f, ranges):
    result = integrate_result(f, *ranges)
    assert result.status is IntegrationStatus.CONDITIONAL
    assert result.complete is False
    assert any(case.status is IntegrationStatus.UNKNOWN for case in result.cases)
    assert sp.simplify_logic(sp.Or(*(case.condition for case in result.cases))) is sp.true


@pytest.mark.parametrize("lo,hi", [(0, 1), (1, sp.oo)])
def test_zero_coefficient_is_not_divergent(lo, hi):
    result = integrate_result(c * x**a, (x, lo, hi))
    selected = [case for case in result.cases if case.condition.subs({c: 0, a: -2}) is sp.true]
    assert len(selected) == 1
    assert selected[0].status is IntegrationStatus.EXACT
    assert selected[0].value == 0
    assert result.complete is True
    assert integrate(c * x**a, (x, lo, hi), assumptions=sp.Eq(c, 0)) == 0


@pytest.mark.parametrize("value", [sp.oo + sp.I * sp.pi, -sp.oo + sp.I, sp.I * sp.oo])
def test_complex_infinity_not_finite_exact(value):
    status, _, convergent, complete = conditional_status(value)
    assert status is IntegrationStatus.DIVERGENT
    assert convergent is False and complete is True


@pytest.mark.parametrize(
    "value", [sp.Limit(sp.sin(x), x, sp.oo), sp.Integral(sp.sin(x), (x, 0, sp.oo))]
)
def test_unevaluated_limits_are_unknown(value):
    status, _, convergent, complete = conditional_status(value)
    assert status is IntegrationStatus.UNKNOWN
    assert convergent is None and complete is False


def test_partial_piecewise_retains_uncovered_cell():
    status, cases, convergent, complete = conditional_status(sp.Piecewise((1, a > 0)))
    assert status is IntegrationStatus.CONDITIONAL
    assert complete is False and convergent is None
    assert len(cases) == 2
    assert cases[1].status is IntegrationStatus.UNKNOWN
    assert cases[1].condition == (a <= 0)


def test_real_order_theorems_do_not_accept_complex_parameters():
    k = sp.Symbol("k")
    assert conditional_cases(sp.exp(-k * x), ((x, 0, sp.oo),)) == ()
    assert conditional_cases(x**k, ((x, 1, sp.oo),)) == ()
    assert conditional_cases(x**k * sp.exp(-x), ((x, 0, sp.oo),)) == ()


@pytest.mark.parametrize("positional", [False, True])
def test_region_budget_preserves_domain_for_both_call_forms(positional):
    box = semialg.Box(((0, 1),))
    policy = IntegrationPolicy(semialg=StrategyBudget(max_variables=0))
    result = (
        integrate_result(1, box, variables=(x,), policy=policy)
        if positional
        else integrate_result(1, region=box, variables=(x,), policy=policy)
    )
    assert result.status is IntegrationStatus.UNEVALUATED
    assert result.complete is False
    assert isinstance(result.value, UnevaluatedRegionIntegral)
    assert result.value.variables == (x,)


def test_owned_region_rejects_conflicting_coordinate_order():
    region = semialg.Box(((0, 1),)).as_semialgebraic_region((x,))
    with pytest.raises(ValueError, match="coordinate order"):
        semialgebraic_domain(region, (y,))
    assert integrate(1, region=region, variables=(x,)) == 1


@pytest.mark.parametrize(
    "region", [semialg.Box(((0, 1), (0, 1))), SimpleNamespace(formula=sp.true, variables=(x, x))]
)
def test_all_domain_forms_validate_distinct_coordinates(region):
    with pytest.raises(ValueError, match="distinct"):
        semialgebraic_domain(region, (x, x))


def test_unknown_intrinsic_dimension_is_not_assumed_ambient(monkeypatch):
    import multiple_integrate.adapters.semialg as adapter

    cover = SimpleNamespace(
        variables=(x, y), certified_dimension=lambda: None, charts=(), formula=sp.Eq(y, x)
    )
    monkeypatch.setattr(adapter, "_intrinsic_parametric_cover", lambda *args, **kwargs: cover)
    assert reduce_intrinsic_domain(1, sp.Eq(y, x), variables=(x, y)) is None


def test_typed_singular_stratum_has_meaningful_error(monkeypatch):
    import multiple_integrate.core as core

    reduced = replace(
        _regular_reduction(x), singular_strata=(SingularStratum((1, 2), 1, ("unverified chart",)),)
    )
    monkeypatch.setattr(core, "reduce_intrinsic_domain", lambda *args, **kwargs: reduced)
    with pytest.raises(NotImplementedError, match="unverified chart"):
        integrate(1, region=sp.Eq(y, x), variables=(x, y), measure_dimension=1)


def test_intrinsic_geometry_honors_variable_budget():
    result = integrate_result(
        1,
        region=semialg.Point((1,)),
        variables=(x,),
        measure_dimension=0,
        policy=IntegrationPolicy(semialg=StrategyBudget(max_variables=0)),
    )
    assert result.status is IntegrationStatus.UNEVALUATED
    assert isinstance(result.value, UnevaluatedRegionIntegral)
    assert result.value.measure_dimension == "0"


def test_intrinsic_geometry_honors_piece_budget(monkeypatch):
    import multiple_integrate.core as core

    reduced = _regular_reduction(x)
    monkeypatch.setattr(core, "reduce_intrinsic_domain", lambda *args, **kwargs: reduced)
    result = integrate_result(
        1,
        region=sp.Eq(y, x),
        variables=(x, y),
        measure_dimension=1,
        policy=IntegrationPolicy(semialg=StrategyBudget(max_pieces=0)),
    )
    assert result.status is IntegrationStatus.UNEVALUATED
    assert "piece count" in result.limitations[0]


def test_regularized_unknown_is_not_complete(monkeypatch):
    import multiple_integrate.api.integration as api

    monkeypatch.setattr(
        api, "integrate", lambda *args, **kwargs: sp.Integral(sp.sin(x), (x, 0, sp.oo))
    )
    result = regularized_result(sp.sin(x), (x, 0, sp.oo), regularization=HadamardFinitePart())
    assert result.complete is False and result.exact is False


def test_fallback_preserves_fractional_power_branches(monkeypatch):
    import multiple_integrate.core as core
    from multiple_integrate import clear_cache

    clear_cache()
    # Retain an unevaluated backend result to inspect the mathematical integrand
    # without relying on the symbolic integrator's ability to finish this case.
    with monkeypatch.context() as context:
        context.setattr(core, "_cached_decomposition", lambda *args: None)
        context.setattr(
            core, "_iterated_integrate", lambda expr, limits, opts: sp.Integral(expr, *limits)
        )
        f = sp.sqrt(x) * sp.sqrt(y) / (3 - x - y)
        value = integrate(f, (x, -2, -1), (y, -2, -1))
        assert isinstance(value, sp.Integral)
        assert value.function.subs({x: -1, y: -1}) == f.subs({x: -1, y: -1}) == -sp.Rational(1, 5)
    clear_cache()
