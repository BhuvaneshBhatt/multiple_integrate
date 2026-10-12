import sys
import types

import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    HypersurfacePrincipalValue,
    IntegrationPolicy,
    IntegrationStatus,
    StrategyBudget,
    UnevaluatedRegionIntegral,
    integrate_result,
    regularized_result,
    verify_regularization_result,
)
from multiple_integrate.adapters.asymptotic import cutoff_expansion
from multiple_integrate.hypersurface_atlas import build_hypersurface_atlas
from multiple_integrate.singularities import stratify_singular_locus
from multiple_integrate.sphere_integrals import sphere_linear_forms_integral


def test_budget_returns_structured_unevaluated():
    x, y = sp.symbols("x y", real=True)
    policy = IntegrationPolicy(semialg=StrategyBudget(max_variables=1))
    result = integrate_result(
        x + y,
        region=sp.And(x >= 0, x <= 1, y >= 0, y <= 1),
        variables=(x, y),
        policy=policy,
    )
    assert result.status is IntegrationStatus.UNEVALUATED
    assert result.complete is False
    assert result.limitations
    assert "variable count" in result.limitations[0]
    assert any(item.method == "semialg_budget_guard" for item in result.attempts)
    assert isinstance(result.value, UnevaluatedRegionIntegral)
    assert result.value.variables == (x, y)
    assert result.value.measure_dimension == "ambient"
    assert (
        sp.simplify(sp.Xor(result.value.domain, sp.And(x >= 0, x <= 1, y >= 0, y <= 1))) is sp.false
    )


def test_singular_strata_include_intersection():
    x, y = sp.symbols("x y", real=True)
    strata = stratify_singular_locus(1 / (x * y), (x, y))
    component_sets = {frozenset(map(sp.expand, item.components)) for item in strata}
    assert frozenset({x}) in component_sets
    assert frozenset({y}) in component_sets
    assert frozenset({x, y}) in component_sets


def test_parameter_strata_are_structured():
    x, a = sp.symbols("x a", real=True)
    result = integrate_result(sp.exp(a * x), (x, 0, sp.oo))
    assert result.status is IntegrationStatus.CONDITIONAL
    assert len(result.parameter_strata) == 2
    assert all(stratum.parameters == (a,) for stratum in result.parameter_strata)
    assert {stratum.convergent for stratum in result.parameter_strata} == {True, False}


def test_asymptotic_adapter_prefers_sibling(monkeypatch):
    epsilon = sp.Symbol("epsilon", positive=True)
    fake = types.ModuleType("asymptotic")
    calls = []

    class Series:
        def is_complete_through(self, order):
            assert order == 0
            return True

        def truncate(self):
            return 2 / epsilon - 3

    def multiseries(expr, variable, *, point, order, return_result):
        calls.append((expr, variable, point, order, return_result))
        return Series()

    fake.multiseries = multiseries
    monkeypatch.setitem(sys.modules, "asymptotic", fake)
    expansion, backend = cutoff_expansion(2 / epsilon - 3, epsilon)
    assert expansion == 2 / epsilon - 3
    assert backend == "asymptotic.multiseries"
    assert calls == [(2 / epsilon - 3, epsilon, 0, 0, True)]


def test_tensor_sphere_contractions():
    e1 = sp.ImmutableMatrix([1, 0, 0])
    e2 = sp.ImmutableMatrix([0, 1, 0])
    assert sp.simplify(sphere_linear_forms_integral((e1, e1)) - 4 * sp.pi / 3) == 0
    assert sp.simplify(sphere_linear_forms_integral((e1, e1, e2, e2)) - 4 * sp.pi / 15) == 0
    assert sphere_linear_forms_integral((e1, e2, e1)) == 0


def test_regularization_certificate_replays():
    x = sp.Symbol("x", real=True)
    result = regularized_result(
        1 / x**2,
        (x, -1, 1),
        regularization=HadamardFinitePart(),
    )
    verification = verify_regularization_result(result)
    assert verification.valid
    assert dict(verification.checks)["constant_term"]


def test_implicit_chart_needs_no_inverse(monkeypatch):
    x, y = sp.symbols("x y", real=True)
    import multiple_integrate.hypersurface_atlas as atlas_module

    monkeypatch.setattr(atlas_module, "_explicit_inverse", lambda *args, **kwargs: None)
    atlas = build_hypersurface_atlas(
        x**2 + y**2 - 1,
        x**2 + y**2 <= 4,
        (x, y),
    )
    assert atlas.charts
    assert all(not chart.explicit for chart in atlas.charts)
    assert all(chart.implicit_relation.has(x, y) for chart in atlas.charts)


def test_quadric_atlas_uses_implicit_charts(monkeypatch):
    x, y = sp.symbols("x y", real=True)
    import multiple_integrate.hypersurface_atlas as atlas_module
    from multiple_integrate.region_types import DiskRegion

    monkeypatch.setattr(atlas_module, "_explicit_inverse", lambda *args, **kwargs: None)
    disk = DiskRegion(
        ((y, -sp.sqrt(4 - x**2), sp.sqrt(4 - x**2)), (x, -2, 2)),
        radius=2,
        center=(0, 0),
    )
    result = regularized_result(
        1 / (x**2 + y**2 - 1),
        region=disk,
        principal_value=HypersurfacePrincipalValue(x**2 + y**2 - 1),
    )
    assert sp.simplify(result.value - sp.pi * sp.log(3)) == 0
    assert result.provenance.details["atlas_chart_count"] >= 2
    assert all(not item["explicit"] for item in result.provenance.details["charts"])


def test_public_root_api_is_curated():
    import multiple_integrate as package

    expected = {
        "integrate",
        "NumericalVerification",
        "verify_finite_result_numerically",
        "integrate_cells",
        "ParameterCell",
        "integrate_result",
        "regularized_result",
        "IntegrationResult",
        "IntegrationOutcome",
        "CoverageEvidence",
        "ConvergenceEvidence",
        "IntegrationPolicy",
        "StrategyBudget",
        "StrategyBudgetExceeded",
        "IntegrationStatus",
        "UnevaluatedRegionIntegral",
        "clear_cache",
        "CauchyPrincipalValue",
        "IteratedPrincipalValue",
        "GeometricPrincipalValue",
        "HypersurfacePrincipalValue",
        "HadamardFinitePart",
        "RegularizationResult",
        "RegularizationVerification",
        "verify_regularization_result",
        "Boole",
        "region_from_ranges",
    }
    assert set(package.__all__) == expected


def test_asymptotic_bare_expression_result_is_unverified(monkeypatch):
    from multiple_integrate.adapters import asymptotic as adapter

    e = sp.Symbol("epsilon", positive=True)
    monkeypatch.setattr(
        adapter,
        "load_asymptotic_capabilities",
        lambda: adapter.AsymptoticCapabilities(
            lambda *args, **kwargs: 2 / e - 3, "asymptotic.multiseries"
        ),
    )
    expression, provider = adapter.cutoff_expansion(2 / e - 3, e)
    assert expression == 2 / e - 3
    assert provider == "sympy.series_fallback"
