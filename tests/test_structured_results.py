import sympy as sp

from multiple_integrate import (
    IntegrationResult,
    IntegrationStatus,
    integrate,
    integrate_result,
)
from multiple_integrate.planner import STRATEGY_OWNERS


def _case(result, status):
    return next(case for case in result.cases if case.status is status)


def test_flat_and_structured_api_agree_on_value():
    x = sp.symbols("x", real=True)
    flat = integrate(x**2, (x, 0, 1))
    detailed = integrate_result(x**2, (x, 0, 1))
    assert isinstance(detailed, IntegrationResult)
    assert detailed.value == flat == sp.Rational(1, 3)
    assert detailed.status is IntegrationStatus.EXACT
    assert detailed.exact is True


def test_strategy_provenance_has_declared_owner():
    x = sp.symbols("x", real=True)
    result = integrate_result(x**2, (x, 0, 1))
    assert result.method in STRATEGY_OWNERS
    assert result.diagnostics["strategy_owner"] == STRATEGY_OWNERS[result.method]
    assert result.attempts[-1].method == result.method


def test_exponential_tail_returns_complete_parameter_strata():
    x, a = sp.symbols("x a", real=True)
    result = integrate_result(sp.exp(-a * x), (x, 0, sp.oo))
    assert result.status is IntegrationStatus.CONDITIONAL
    assert result.complete is True
    good = _case(result, IntegrationStatus.EXACT)
    bad = _case(result, IntegrationStatus.DIVERGENT)
    assert sp.simplify(good.value - 1 / a) == 0
    assert sp.simplify_logic(sp.Equivalent(good.condition, a > 0)) is sp.true
    assert sp.simplify_logic(sp.Equivalent(bad.condition, a <= 0)) is sp.true


def test_power_tail_returns_complete_parameter_strata():
    x, a = sp.symbols("x a", real=True)
    result = integrate_result(x ** (-a), (x, 1, sp.oo))
    good = _case(result, IntegrationStatus.EXACT)
    bad = _case(result, IntegrationStatus.DIVERGENT)
    assert sp.simplify(good.value - 1 / (a - 1)) == 0
    assert sp.simplify_logic(sp.Equivalent(good.condition, a > 1)) is sp.true
    assert sp.simplify_logic(sp.Equivalent(bad.condition, a <= 1)) is sp.true


def test_unit_interval_power_returns_parameter_strata():
    x, a = sp.symbols("x a", real=True)
    result = integrate_result(x ** (a - 1), (x, 0, 1))
    good = _case(result, IntegrationStatus.EXACT)
    assert sp.simplify(good.value - 1 / a) == 0
    assert sp.simplify_logic(sp.Equivalent(good.condition, a > 0)) is sp.true


def test_gaussian_returns_parameter_strata():
    x, a = sp.symbols("x a", real=True)
    result = integrate_result(sp.exp(-a * x**2), (x, -sp.oo, sp.oo))
    good = _case(result, IntegrationStatus.EXACT)
    assert sp.refine(good.value, sp.Q.positive(a)) == sp.sqrt(sp.pi) / sp.sqrt(a)
    assert sp.simplify_logic(sp.Equivalent(good.condition, a > 0)) is sp.true


def test_conditional_specialization_matches_direct_queries():
    x, a = sp.symbols("x a", real=True)
    stratified = integrate_result(x ** (-a), (x, 1, sp.oo))
    good = _case(stratified, IntegrationStatus.EXACT)
    direct = integrate(x ** (-a), (x, 1, sp.oo), assumptions={a > 1})
    assert sp.simplify(good.value - direct) == 0


def test_assumptions_specialize_registered_convergence_before_backends(monkeypatch):
    import multiple_integrate.singularities as singularities

    x, a = sp.symbols("x a", real=True)

    def fail_query(*args, **kwargs):
        raise AssertionError("registered convergence theorem should short-circuit funcprops")

    monkeypatch.setattr(singularities, "function_singularities", fail_query)
    value = integrate(x ** (-a), (x, 1, sp.oo), assumptions={a > 1})
    assert sp.simplify(value - 1 / (a - 1)) == 0
