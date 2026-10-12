"""Regression evidence for classification, bounded diagnostics, and symbolic hygiene."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    IntegrationOutcome,
    IntegrationPolicy,
    StrategyBudget,
    UnevaluatedRegionIntegral,
    integrate_result,
    regularized_result,
    verify_regularization_result,
)
from multiple_integrate.singularities import stratify_singular_locus

x, y, a = sp.symbols("x y a", real=True)


@pytest.mark.parametrize("f", [sp.sin(x), sp.cos(x)])
def test_oscillatory_improper_is_not_finite(f):
    result = integrate_result(f, (x, 0, sp.oo))
    assert result.value.has(sp.AccumBounds)
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert result.convergent is None and result.complete is False
    assert result.absolute_convergent is None


def test_empty_components_prune_all_supersets():
    f = 1 / sp.prod(x - i for i in range(12))
    with patch(
        "multiple_integrate.singularities.semialg_region_dimension", return_value=-1
    ) as dimension:
        result = stratify_singular_locus(f, (x,), return_details=True)
    assert dimension.call_count == 12
    assert result.complete and result.strata == ()


def test_unknown_diagnostics_are_bounded_and_marked_partial():
    with (
        patch("multiple_integrate.singularities.semialg_region_dimension", return_value=None),
        patch("multiple_integrate.singularities.semialg_is_satisfiable", return_value=None),
    ):
        result = stratify_singular_locus(
            1 / (x * y * (x + y)), (x, y), max_queries=5, return_details=True
        )
    assert result.queries == 5 and not result.complete
    assert result.strata and all(s.dimension is None for s in result.strata)


def test_auxiliary_budget_does_not_erase_value():
    result = integrate_result(1 / x, (x, 1, 2), policy=IntegrationPolicy(max_singular_queries=0))
    assert sp.simplify(result.value - sp.log(2)) == 0
    assert result.outcome is IntegrationOutcome.FINITE
    assert result.diagnostics["singular_diagnostics_complete"] is False
    assert any("diagnostics are partial" in s for s in result.limitations)


def test_actual_domain_excludes_poles_from_diagnostics():
    result = integrate_result(1 / x, (x, 1, 2))
    assert result.singular_strata == ()
    assert result.diagnostics["singular_diagnostics_complete"] is True


@pytest.mark.parametrize(
    "operation", [lambda u: u.subs(a, 1), lambda u: u.xreplace({a: sp.Integer(1)})]
)
def test_region_placeholder_specialization(operation):
    u = UnevaluatedRegionIntegral(a * x, x**2 <= a, (x,), 1)
    v = operation(u)
    assert v.integrand == x and v.domain == (x**2 <= 1)
    assert v.variables == (x,) and v.measure_dimension == "1"
    assert v.func(*v.args) == v


def test_placeholder_substitution_avoids_capture():
    u = UnevaluatedRegionIntegral(a * x, x >= a, (x,))
    v = u.subs(a, x)
    bound = v.variables[0]
    assert isinstance(bound, sp.Dummy) and bound != x
    assert v.integrand == x * bound and v.domain == (bound >= x)
    assert v.free_symbols == {x}
    assert u.subs(x, y) == u


def test_finite_iterated_answer_does_not_claim_absolute_convergence():
    f = (x**2 - y**2) / (x**2 + y**2) ** 2
    first = integrate_result(f, (x, -1, 1), (y, -1, 1))
    second = integrate_result(f, (y, -1, 1), (x, -1, 1))
    assert first.value == -sp.pi and second.value == sp.pi
    assert first.ordinary_convergent is None and first.absolute_convergent is None
    assert any(e.kind == "iterated" and e.established is True for e in first.convergence_evidence)


def test_compact_polynomial_gets_sufficient_convergence_proof():
    result = integrate_result(x**2 + a * y, (x, 0, 1), (y, -1, 1))
    assert result.ordinary_convergent is True and result.absolute_convergent is True


def test_verification_strength_is_explicit():
    fp = regularized_result(x**-2, (x, -1, 1), regularization=HadamardFinitePart())
    verification = verify_regularization_result(fp)
    assert verification.valid and verification.strength == "cutoff_replay"
    assert verification.limitations
    constant_only = replace(
        fp, provenance=replace(fp.provenance, cutoff_parameter=None, cutoff_integral=None)
    )
    weak = verify_regularization_result(constant_only)
    assert weak.valid and weak.strength == "constant_consistency"
    assert verify_regularization_result(replace(fp, value=fp.value + 1)).valid is False


@pytest.mark.parametrize(
    "factory",
    [
        lambda: StrategyBudget(max_variables=True),
        lambda: IntegrationPolicy(max_singular_queries=True),
    ],
)
def test_boolean_budget_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_nested_policy_type_is_checked():
    with pytest.raises(TypeError):
        IntegrationPolicy(semialg={})


def test_placeholder_simultaneous_parameter_substitution_avoids_capture():
    u = UnevaluatedRegionIntegral(a * x, x >= a, (x,))
    v = u.subs(a, x, simultaneous=True)
    bound = v.variables[0]
    assert bound != x and v.free_symbols == {x}
    assert v.integrand == x * bound and v.domain == (bound >= x)


def test_unused_parameter_substitution_is_a_noop():
    b = sp.Symbol("unused", real=True)
    u = UnevaluatedRegionIntegral(a * x, x >= a, (x,))
    assert u.subs(b, x) == u
