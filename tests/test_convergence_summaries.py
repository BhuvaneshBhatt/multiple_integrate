"""Consensus requires genuine proof coverage, rather than finite-looking values."""

import pytest
import sympy as sp

from multiple_integrate import (
    ConvergenceEvidence,
    IntegrationResult,
    IntegrationStatus,
    ParameterCell,
    integrate_cells,
)

x, a = sp.symbols("x a", real=True)


def result(claims, assumptions=sp.true):
    return IntegrationResult(
        0,
        IntegrationStatus.EXACT,
        "test",
        assumptions=assumptions,
        convergence_evidence=tuple(
            ConvergenceEvidence("absolute", fact, "test", condition) for fact, condition in claims
        ),
    )


@pytest.mark.parametrize(
    "claims,expected",
    [
        ([(True, sp.true), (True, sp.true)], True),
        ([(False, sp.true), (False, sp.true)], False),
        ([(None, sp.true), (True, sp.true)], True),
        ([(True, sp.true), (False, sp.true)], None),
        ([(True, a > 0), (True, a <= 0)], True),
        ([(True, a > 0)], None),
        ([(True, a > 0), (None, a <= 0)], None),
        ([(True, a > 0), (False, a <= 0)], None),
    ],
)
def test_convergence_consensus_preserves_missing_and_conflicting_facts(claims, expected):
    assert result(claims).absolute_convergent is expected


def test_summary_uses_parent_domain_and_preserves_equality_fibers():
    facts = [(True, a > 0), (True, sp.Eq(a, 0)), (False, a < 0)]
    assert result(facts, a >= 0).absolute_convergent is True
    assert result([(True, a > 0)], a >= 0).absolute_convergent is None
    assert result([(True, a > 0)], sp.Q.positive(a)).absolute_convergent is True


def test_scoped_summary_is_cached_after_coverage_proof(monkeypatch):
    r = result([(True, a > 0), (True, a <= 0)])
    assert r.absolute_convergent is True
    monkeypatch.setattr(
        "multiple_integrate.adapters.semialg.is_satisfiable",
        lambda *args: pytest.fail("repeated coverage query"),
    )
    assert r.absolute_convergent is True


def test_finite_value_is_not_itself_an_absolute_certificate():
    assert result([]).absolute_convergent is None


def test_piecewise_moving_point_has_branch_finiteness_and_parent_consensus():
    r = integrate_cells(
        sp.Piecewise((1 / a, a > 0), (0, True)),
        region=sp.Eq(x, a),
        variables=(x,),
        parameters=(a,),
        cells=[ParameterCell(a >= 0)],
        assumptions=a >= 0,
        measure_dimension="intrinsic",
    )
    assert r.complete is True
    assert r.ordinary_convergent is True and r.absolute_convergent is True
    assert r.value.subs(a, 0) == 0 and r.value.subs(a, 2) == sp.Rational(1, 2)
    assert any(
        e.condition != sp.true and e.established is True
        for e in r.diagnostics["cell_results"][0].convergence_evidence
    )


def test_uncovered_parameter_cell_cannot_acquire_consensus():
    r = integrate_cells(1, (x, 0, 1), parameters=(a,), cells=[ParameterCell(a > 0)])
    assert r.complete is False and r.absolute_convergent is None


def test_scoped_summary_retains_request_budget(monkeypatch):
    from dataclasses import replace

    from multiple_integrate import IntegrationPolicy, StrategyBudget

    r = replace(
        result([(True, a > 0)], a >= 0),
        _convergence_policy=IntegrationPolicy(semialg=StrategyBudget(max_variables=0)),
    )
    monkeypatch.setattr(
        "multiple_integrate.adapters.semialg.is_satisfiable",
        lambda *args: pytest.fail("ignored retained summary budget"),
    )
    assert r.absolute_convergent is None


def test_regularization_summaries_use_the_same_scoped_consensus():
    from multiple_integrate.regularization import RegularizationResult

    r = RegularizationResult(
        0,
        "finite_part",
        "test",
        None,
        None,
        assumptions=a >= 0,
        convergence_evidence=tuple(
            ConvergenceEvidence("absolute", True, "test", condition)
            for condition in [a > 0, sp.Eq(a, 0)]
        ),
    )
    assert r.absolute_convergent is True
    assert r.ordinary_convergent is None
