"""Deterministic work bounds, unsuccessful recognizers and cache scope."""

import pytest
import sympy as sp

from multiple_integrate import IntegrationPolicy, StrategyBudget, clear_cache, integrate_result
from multiple_integrate.fast_rules import _polynomial
from multiple_integrate.policy import _CURRENT_POLICY
from multiple_integrate.selected_families import selected_interval_rule

x, a, b = sp.symbols("x a b", real=True)


@pytest.mark.parametrize(
    "expression", [(a + b) ** 64 * x, (x + 1) ** 1000000, (a + b + 1) ** 100 * x**2]
)
def test_expression_growth_declines_before_polynomial_expansion(expression, monkeypatch):
    import multiple_integrate.fast_rules as rules

    token = _CURRENT_POLICY.set(IntegrationPolicy(semialg=StrategyBudget(max_pieces=4)))
    try:

        def forbidden(*args, **kwargs):
            pytest.fail("constructed Poly despite pre-expansion budget")

        monkeypatch.setattr(rules.sp, "Poly", forbidden)
        assert _polynomial(expression, (x,)) is None
    finally:
        _CURRENT_POLICY.reset(token)


@pytest.mark.parametrize("phase", [-(x**4) - x, -(x**6) + x**2])
def test_unsuccessful_phase_recognition_does_not_expand_amplitude(phase, monkeypatch):
    import multiple_integrate.selected_families as families

    def forbidden(*args, **kwargs):
        pytest.fail("expanded amplitude after rejecting phase")

    monkeypatch.setattr(families, "_powers", forbidden)
    assert selected_interval_rule((a + b) ** 64 * sp.exp(phase), x, 0, sp.oo) is None


@pytest.mark.parametrize(
    "density,bounds",
    [(sp.exp(-((x - 2) ** 4)), (2, sp.oo)), ((1 + x) * x ** -sp.Rational(1, 2), (0, 1))],
)
def test_selected_families_do_not_call_general_integration(density, bounds, monkeypatch):
    import multiple_integrate.assumptions as assumptions
    import multiple_integrate.strategy_common as common

    def forbidden(*args, **kwargs):
        pytest.fail("general fallback used for proved specialized family")

    monkeypatch.setattr(assumptions, "_integrate", forbidden)
    monkeypatch.setattr(common, "_integrate", forbidden)
    clear_cache()
    assert integrate_result(density, (x, *bounds)).absolute_convergent is True


def test_cache_scope_distinguishes_assumptions_bounds_and_density():
    clear_cache()
    positive = integrate_result(sp.exp(-a * x), (x, 0, sp.oo), assumptions=a > 0)
    boundary = integrate_result(sp.exp(-a * x), (x, 0, sp.oo), assumptions=sp.Eq(a, 0))
    assert positive.finite and not boundary.finite
    assert integrate_result(x**2, (x, 0, 1)).value == sp.Rational(1, 3)
    assert integrate_result(x**2, (x, 0, 2)).value == sp.Rational(8, 3)
    assert integrate_result(2 * x**2, (x, 0, 1)).value == sp.Rational(2, 3)
    again = integrate_result(sp.exp(-a * x), (x, 0, sp.oo), assumptions=a > 0)
    assert again.value == positive.value and again.absolute_convergent is True


def test_cache_eviction_and_clear_preserve_certificates(monkeypatch):
    import multiple_integrate.core as core
    from multiple_integrate.region_recognition import _recognize_ranges

    monkeypatch.setattr(core, "_CACHE_MAXSIZE", 2)
    clear_cache()
    for width in (1, 2, 3):
        assert integrate_result(x**2, (x, 0, width)).absolute_convergent is True
    assert len(core.integrate._cache) <= 2
    clear_cache()
    assert not core.integrate._cache and _recognize_ranges.cache_info().currsize == 0
    result = integrate_result(x**2, (x, 0, 1))
    assert result.value == sp.Rational(1, 3) and result.absolute_convergent is True


@pytest.mark.parametrize("ready", [False, True])
def test_backend_deadline_is_forwarded_and_worker_reaped(ready, monkeypatch):
    import multiple_integrate.isolated_queries as queries

    events = []

    class Connection:
        def close(self):
            events.append("close")

        def poll(self, seconds):
            assert seconds == 0.25
            return ready

        def recv(self):
            return True, sp.Integer(7)

    class Process:
        def start(self):
            events.append("start")

        def is_alive(self):
            return True

        def terminate(self):
            events.append("terminate")

        def join(self, *args, **kwargs):
            events.append("join")

        def kill(self):
            events.append("kill")

        def close(self):
            events.append("process_close")

    class Context:
        def Pipe(self, **kwargs):
            return Connection(), Connection()

        def Process(self, **kwargs):
            return Process()

    monkeypatch.setattr(queries.mp, "get_context", lambda mode: Context())
    result = queries.isolated_query("sympy", "integrate", (), {}, 0.25)
    assert result == (sp.Integer(7) if ready else None)
    assert "terminate" in events and "join" in events and events.count("close") >= 2
