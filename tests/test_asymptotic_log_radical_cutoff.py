"""Regression tests for exact logarithmic-tangency cutoffs through order zero.

These tests exercise the actual optional expansion engine, and explicitly
reject reintroducing the expensive term-count contract. A separate adapter
contract test covers a complete-result mock, an incomplete result, and the
independent SymPy fallback.
"""

import pytest
import sympy as sp

from multiple_integrate.adapters import asymptotic as adapter
from multiple_integrate.regularization import finite_part_from_cutoff


def _require_order_zero_backend(monkeypatch):
    engine = pytest.importorskip("asymptotic")
    multiseries = engine.multiseries
    observed = []

    def checked_multiseries(expr, variable, **kwargs):
        assert kwargs == {"point": 0, "order": 0, "return_result": True}
        observed.append(True)
        return multiseries(expr, variable, **kwargs)

    monkeypatch.setattr(
        adapter,
        "load_asymptotic_capabilities",
        lambda: adapter.AsymptoticCapabilities(checked_multiseries, "asymptotic.multiseries"),
    )
    return observed


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("scale", [1, 2, -3])
def test_tangent_log_radical_expansion_uses_order_zero(monkeypatch, order, scale):
    observed = _require_order_zero_backend(monkeypatch)
    epsilon = sp.Symbol("epsilon", positive=True)
    modulus = abs(scale)
    lower = (1 - sp.sqrt(1 - 4 * epsilon / modulus)) / 2
    higher = 1 - lower
    cutoff = sp.Rational(2, order) * (
        -(sp.log(lower) ** 2) / 4
        - sp.log(lower) * sp.log(higher) / 2
        + sp.log(higher) ** 2 / 4
        - sp.polylog(2, lower)
        + sp.pi**2 / 12
    )

    value, provenance = finite_part_from_cutoff(cutoff, epsilon)
    expected = sp.pi**2 / (6 * order) - sp.log(modulus) ** 2 / (2 * order)
    assert sp.simplify(value - expected) == 0
    assert provenance.details["asymptotic_backend"] == "asymptotic.multiseries"
    assert provenance.divergent_terms
    assert provenance.cutoff_integral == cutoff
    assert observed == [True]


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("scale", [1, 2, -3])
def test_weighted_tangent_cutoff_expands_through_constant(monkeypatch, order, scale):
    observed = _require_order_zero_backend(monkeypatch)
    epsilon = sp.Symbol("epsilon", positive=True)
    modulus = abs(scale)
    delta = epsilon / modulus
    discriminant = sp.sqrt(1 - 4 * delta)
    lower = (1 - discriminant) / 2
    higher = 1 - lower
    cutoff = ((1 - 2 * delta) * sp.log(lower / higher) + discriminant) / order

    value, provenance = finite_part_from_cutoff(cutoff, epsilon)
    assert sp.simplify(value - (1 - sp.log(modulus)) / order) == 0
    assert provenance.details["asymptotic_backend"] == "asymptotic.multiseries"
    assert provenance.cutoff_integral == cutoff
    assert observed == [True]


def test_nonradical_rational_cutoff_keeps_native_backend(monkeypatch):
    epsilon = sp.Symbol("epsilon", positive=True)
    calls = []

    class Complete:
        def is_complete_through(self, order):
            assert order == 0
            return True

        def truncate(self):
            return 2 / epsilon - 3

    def native(expr, variable, **kwargs):
        assert expr == 2 / epsilon - 3
        assert variable == epsilon
        assert kwargs == {"point": 0, "order": 0, "return_result": True}
        calls.append(True)
        return Complete()

    monkeypatch.setattr(
        adapter,
        "load_asymptotic_capabilities",
        lambda: adapter.AsymptoticCapabilities(native, "asymptotic.multiseries"),
    )
    result, backend = adapter.cutoff_expansion(2 / epsilon - 3, epsilon)
    assert result == 2 / epsilon - 3
    assert backend == "asymptotic.multiseries"
    assert calls == [True]
