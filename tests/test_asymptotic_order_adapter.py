"""Contract tests for inclusive order-zero cutoff expansion.

The adapter is imported independently of the integration package, so these
unit tests can run without the optional semialgebraic geometry dependencies.
The integration-level tangency and contact tests exercise the whole package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import sympy as sp


@pytest.fixture
def adapter(monkeypatch):
    file = Path(__file__).resolve().parents[1] / "src/multiple_integrate/adapters/asymptotic.py"
    name = "_multiple_integrate_asymptotic_order_contract"
    spec = importlib.util.spec_from_file_location(name, file)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class CompletePrefix:
    def __init__(self, expression):
        self.expression = expression
        self.status = "complete"
        self.certified = False

    def is_complete_through(self, order):
        return self.status == "complete" and order <= 0

    def truncate(self):
        return self.expression


class UnknownPrefix(CompletePrefix):
    status = "unknown"

    def __init__(self):
        self.status = "unknown"

    def truncate(self):
        pytest.fail("An incomplete expansion must not be truncated")


def test_requests_order_zero_not_eight_terms(adapter, monkeypatch):
    e = sp.Symbol("epsilon", positive=True)
    expr = 1 / e**2 + sp.log(e) / e + 7 * sp.log(e) ** 2 + 5 + e
    expected = expr - e
    seen = []

    def provider(*args, **kwargs):
        seen.append((args, kwargs))
        return CompletePrefix(expected)

    monkeypatch.setattr(
        adapter,
        "load_asymptotic_capabilities",
        lambda: adapter.AsymptoticCapabilities(provider, "asymptotic.multiseries"),
    )
    actual, backend = adapter.cutoff_expansion(expr, e)
    assert actual == expected
    assert backend == "asymptotic.multiseries"
    assert seen == [((expr, e), {"point": 0, "order": 0, "return_result": True})]


def test_incomplete_provider_falls_back_on_exact_original(adapter, monkeypatch):
    e = sp.Symbol("epsilon", positive=True)
    expr = sp.log(e) ** 2 + sp.pi**2 / 12 + e * sp.log(e)
    monkeypatch.setattr(
        adapter,
        "load_asymptotic_capabilities",
        lambda: adapter.AsymptoticCapabilities(
            lambda *a, **k: UnknownPrefix(), "asymptotic.multiseries"
        ),
    )
    actual, backend = adapter.cutoff_expansion(expr, e)
    assert backend == "sympy.series_fallback"
    assert sp.limit(actual - (sp.log(e) ** 2 + sp.pi**2 / 12), e, 0, dir="+") == 0


def test_legacy_provider_without_order_falls_back(adapter, monkeypatch):
    e = sp.Symbol("epsilon", positive=True)
    expr = 2 / e + 3 + e

    def old_api(expression, variable, *, terms, point, allow_series_fallback, return_result):
        pytest.fail("An old term-count backend must not be invoked with eight terms")

    monkeypatch.setattr(
        adapter,
        "load_asymptotic_capabilities",
        lambda: adapter.AsymptoticCapabilities(old_api, "asymptotic.multiseries"),
    )
    actual, backend = adapter.cutoff_expansion(expr, e)
    assert backend == "sympy.series_fallback"
    assert actual == 2 / e + 3


def test_unverified_sympy_expr_provider_result_is_not_trusted(adapter, monkeypatch):
    e = sp.Symbol("epsilon", positive=True)
    expr = 2 / e + 3 + e
    monkeypatch.setattr(
        adapter,
        "load_asymptotic_capabilities",
        lambda: adapter.AsymptoticCapabilities(lambda *a, **k: expr, "asymptotic.multiseries"),
    )
    actual, backend = adapter.cutoff_expansion(expr, e)
    assert backend == "sympy.series_fallback"
    assert actual == 2 / e + 3


def test_force_sympy_replay_does_not_load_optional_provider(adapter, monkeypatch):
    e = sp.Symbol("epsilon", positive=True)
    expr = 1 / e + sp.log(e) ** 2 + 7 + e

    def unexpected():
        pytest.fail("force_sympy should not load the optional backend")

    monkeypatch.setattr(adapter, "load_asymptotic_capabilities", unexpected)
    actual, backend = adapter.cutoff_expansion(expr, e, force_sympy=True)
    assert backend == "sympy.series_fallback"
    assert sp.simplify(actual - (1 / e + sp.log(e) ** 2 + 7)) == 0


def test_optional_provider_order_zero_real_tangency(adapter):
    pytest.importorskip("asymptotic")
    e = sp.Symbol("epsilon", positive=True)
    m, k = 2, -3
    d = e / abs(k)
    smaller = (1 - sp.sqrt(1 - 4 * d)) / 2
    larger = 1 - smaller
    exact_cutoff = (
        sp.log(larger) ** 2
        - sp.log(smaller) ** 2
        - 2 * sp.log(smaller) * sp.log(larger)
        - 4 * sp.polylog(2, smaller)
        + sp.pi**2 / 3
    ) / (2 * m)
    expansion, backend = adapter.cutoff_expansion(exact_cutoff, e)
    expected = (
        -(sp.log(e) ** 2) / (2 * m)
        + sp.log(abs(k)) * sp.log(e) / m
        + sp.pi**2 / (6 * m)
        - sp.log(abs(k)) ** 2 / (2 * m)
    )
    assert sp.simplify(expansion - expected) == 0
    assert backend == "asymptotic.multiseries"
