"""Reproduced serious protocol and exceptional-fiber mistakes."""

import pytest
import semialg
import sympy as sp

from multiple_integrate import (
    IntegrationOutcome,
    IntegrationPolicy,
    ParameterCell,
    integrate_cells,
    integrate_result,
)
from multiple_integrate.assumptions import _ask, _normalize_assumptions
from multiple_integrate.conditional import conditional_status
from multiple_integrate.quadratic_radial import positive_quadratic

x, y, a = sp.symbols("x y a", real=True)


def test_finite_radial_series_is_not_divergent():
    z = sp.Symbol("z", real=True)
    q = x * x + y * y + z * z
    result = integrate_result(
        sp.exp(-q) / (2 + q), region=semialg.Ball((0, 0, 0), 1), variables=(x, y, z)
    )
    assert result.value.has(sp.Sum)
    assert result.outcome is IntegrationOutcome.FINITE
    assert result.finite is True and result.absolute_convergent is True


@pytest.mark.parametrize("term", [lambda k: 1 / k, lambda k: 2**k, lambda k: 1 / sp.factorial(k)])
def test_unresolved_series_needs_its_own_proof(term):
    k = sp.Symbol("k", positive=True, integer=True)
    value = sp.Sum(term(k), (k, 1, sp.oo))
    status, _, convergent, complete = conditional_status(value)
    assert status.value == "unknown" and convergent is None and complete is False


@pytest.mark.parametrize("value", [sp.oo, -sp.oo, sp.zoo])
def test_explicit_infinite_values_still_diverge(value):
    status, _, convergent, complete = conditional_status(value)
    assert status.value == "divergent" and convergent is False and complete is True


@pytest.mark.parametrize("condition,expression", [(a <= 0, -a), (sp.And(a > 0, a <= 1), 1 - a)])
def test_strict_sign_never_erases_a_weak_boundary(condition, expression):
    assert _ask(sp.Q.positive(expression), _normalize_assumptions(condition)) is not True


def test_gaussian_boundary_rank_drop_cannot_claim_uniform_convergence():
    context = sp.And(a > 0, a <= 1)
    assert positive_quadratic(x * x + (1 - a) * y * y, (x, y), context) is None
    result = integrate_result(
        sp.exp(-x * x - (1 - a) * y * y),
        (x, -sp.oo, sp.oo),
        (y, -sp.oo, sp.oo),
        assumptions=context,
        policy=IntegrationPolicy(backend_timeout_seconds=1),
    )
    assert result.absolute_convergent is not True
    assert result.complete is not True
    assert positive_quadratic(x * x + (1 - a) * y * y, (x, y), sp.And(a > 0, a < 1)) is not None


@pytest.mark.parametrize("formula", [sp.And(x >= 0, x <= a), x * x <= a, sp.Eq(x * x, a)])
def test_intrinsic_dimension_drop_requires_separate_cells(formula):
    policy = IntegrationPolicy(backend_timeout_seconds=2)
    weak = integrate_cells(
        1,
        region=formula,
        variables=(x,),
        parameters=(a,),
        assumptions=a >= 0,
        cells=(ParameterCell(a >= 0),),
        measure_dimension="intrinsic",
        policy=policy,
    )
    child = weak.diagnostics["cell_results"][0]
    assert child.outcome is IntegrationOutcome.UNKNOWN
    assert child.complete is not True and child.absolute_convergent is not True
    split = integrate_cells(
        1,
        region=formula,
        variables=(x,),
        parameters=(a,),
        assumptions=a >= 0,
        cells=(ParameterCell(sp.Eq(a, 0), {a: 0}), ParameterCell(a > 0)),
        measure_dimension="intrinsic",
        policy=policy,
    )
    assert split.cases[0].value == 1
    assert split.complete


def test_parameter_pole_is_not_uniformly_finite_on_a_weak_cell():
    context = sp.And(a > 0, a <= 1)
    assert _ask(sp.Q.finite(1 / (1 - a)), _normalize_assumptions(context)) is not True
    result = integrate_result(1 / (1 - a), (x, 0, 1), assumptions=context)
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert result.absolute_convergent is not True and result.complete is not True
    valid = integrate_result(1 / (1 - a), (x, 0, 1), assumptions=sp.And(a > 0, a < 1))
    assert valid.outcome is IntegrationOutcome.FINITE and valid.absolute_convergent is True


@pytest.mark.parametrize("condition", [a > 0, a < 0, sp.Ne(a, 0), sp.Q.nonzero(a)])
def test_explicit_nonzero_parameter_denominators_remain_supported(condition):
    result = integrate_result(1 / a**2, (x, 0, 1), assumptions=condition)
    assert result.outcome is IntegrationOutcome.FINITE
