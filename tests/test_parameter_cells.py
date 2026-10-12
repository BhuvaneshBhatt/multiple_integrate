"""Generic and exceptional fibers retain exact per-cell result semantics."""

import pytest
import sympy as sp

from multiple_integrate import (
    IntegrationOutcome,
    IntegrationPolicy,
    IntegrationStatus,
    ParameterCell,
    StrategyBudget,
    StrategyBudgetExceeded,
    integrate_cells,
)

x, a, b = sp.symbols("x a b", real=True)


def test_generic_and_transition_cells_are_exhaustive():
    result = integrate_cells(
        x,
        (x, 0, a),
        parameters=(a,),
        cells=(ParameterCell(a > 0), ParameterCell(sp.Eq(a, 0), {a: 0}), ParameterCell(a < 0)),
    )
    assert result.complete and result.exact
    assert len(result.cases) == 3
    assert (
        result.value.subs(a, 2) == 2
        and result.value.subs(a, 0) == 0
        and result.value.subs(a, -2) == 2
    )
    assert result.coverage[0].scope == "parameter_space" and result.coverage[0].complete
    assert result.diagnostics["cell_results"][1].value == 0


def test_uncovered_parameters_remain_unknown():
    result = integrate_cells(x, (x, 0, 1), parameters=(a,), cells=(ParameterCell(a > 0),))
    assert not result.complete and result.cases[-1].outcome is IntegrationOutcome.UNKNOWN
    assert result.value.subs(a, -1) == sp.Integral(x, (x, 0, 1))


def test_parent_assumptions_define_coverage_scope():
    result = integrate_cells(
        x, (x, 0, 1), parameters=(a,), assumptions=a > 0, cells=(ParameterCell(a > 0),)
    )
    assert result.coverage[0].complete and result.complete


def test_conditional_child_outcomes_are_retained():
    result = integrate_cells(
        sp.exp(-a * x),
        (x, 0, sp.oo),
        parameters=(a, b),
        cells=(ParameterCell(b > 0), ParameterCell(b <= 0)),
    )
    assert {c.status for c in result.cases} == {
        IntegrationStatus.EXACT,
        IntegrationStatus.DIVERGENT,
    }
    assert all(c.condition.has(b) for c in result.cases)
    assert result.value.subs({a: 2, b: 1}) == sp.Rational(1, 2)
    assert result.value.subs({a: -1, b: 1}).has(sp.Integral)


def test_exceptional_geometry_is_specialized_without_sample_points():
    # A zero-dimensional fiber becomes empty on one cell and a point on another.
    region = sp.And(sp.Eq(x, 0), a >= 0)
    result = integrate_cells(
        1,
        region=region,
        variables=(x,),
        measure_dimension=0,
        parameters=(a,),
        cells=(ParameterCell(a < 0), ParameterCell(sp.Eq(a, 0), {a: 0})),
    )
    assert result.cases[0].value == 0 and result.cases[1].value == 1
    assert result.cases[2].outcome is IntegrationOutcome.UNKNOWN


@pytest.mark.parametrize(
    "cells",
    [
        (ParameterCell(a >= 0), ParameterCell(a <= 0)),
        (ParameterCell(a > 0, {a: 1}),),
        (ParameterCell(x > 0),),
        (ParameterCell(sp.Eq(a, 0), {a: x}),),
    ],
)
def test_invalid_partition_or_substitution_is_rejected_before_evaluation(cells, monkeypatch):
    monkeypatch.setattr(
        "multiple_integrate.api.integration.integrate_result",
        lambda *args, **kwargs: pytest.fail("invalid partition evaluated"),
    )
    with pytest.raises(ValueError):
        integrate_cells(x, (x, 0, 1), parameters=(a,), cells=cells)


def test_cell_budget_precedes_geometry_work():
    with pytest.raises(StrategyBudgetExceeded):
        integrate_cells(
            x,
            (x, 0, 1),
            parameters=(a,),
            cells=(ParameterCell(a > 0), ParameterCell(a <= 0)),
            policy=IntegrationPolicy(semialg=StrategyBudget(max_pieces=1)),
        )


def test_empty_cells_are_pruned_and_parameter_order_is_irrelevant():
    cells = (ParameterCell(sp.false), ParameterCell(a + b > 0), ParameterCell(a + b <= 0))
    first = integrate_cells(x, (x, 0, 1), parameters=(a, b), cells=cells)
    second = integrate_cells(x, (x, 0, 1), parameters=(b, a), cells=cells)
    assert first.value == second.value and len(first.cases) == 2 and first.complete


def test_simultaneous_fiber_substitutions_and_bounds():
    result = integrate_cells(
        a + b,
        region=sp.And(x >= 0, x <= 2),
        variables=(x,),
        bounds={x: (0, a)},
        parameters=(a, b),
        cells=(ParameterCell(sp.And(sp.Eq(a, 1), sp.Eq(b, a)), {a: 1, b: 1}),),
    )
    assert result.cases[0].value == 2


def test_uncertified_disjointness_is_rejected(monkeypatch):
    monkeypatch.setattr("multiple_integrate.parameter_cells.is_satisfiable", lambda *args: None)
    with pytest.raises(ValueError, match="disjointness"):
        integrate_cells(
            x,
            (x, 0, 1),
            parameters=(a, b),
            cells=(
                ParameterCell(a * a + b * b < 1),
                ParameterCell(a * a + b * b > sp.Rational(1, 2)),
            ),
        )


def test_unreduced_generic_geometry_remains_unknown():
    result = integrate_cells(
        1,
        region=x**4 <= a,
        variables=(x,),
        parameters=(a,),
        cells=(ParameterCell(a > 0), ParameterCell(sp.Eq(a, 0), {a: 0})),
    )
    assert result.cases[0].outcome is IntegrationOutcome.UNKNOWN
    assert result.cases[1].value == 0  # ambient length of the singleton fiber
    assert not result.complete
    assert result.coverage[0].complete is False  # a<0 was not supplied


def test_exceptional_regulator_is_specialized_together_with_density():
    from multiple_integrate import HypersurfacePrincipalValue

    result = integrate_cells(
        a * x,
        (x, -1, 1),
        parameters=(a,),
        cells=(ParameterCell(sp.Eq(a, 2), {a: 2}),),
        principal_value=HypersurfacePrincipalValue(a * x),
    )
    child = result.diagnostics["cell_results"][0]
    assert child.value == 0 and child.status is IntegrationStatus.PRINCIPAL_VALUE


def test_empty_family_cell_is_proved_without_a_sample_fiber():
    result = integrate_cells(
        1,
        region=x * x <= a,
        variables=(x,),
        parameters=(a,),
        cells=(ParameterCell(a < 0), ParameterCell(sp.Eq(a, 0), {a: 0}), ParameterCell(a > 0)),
    )
    assert result.cases[0].value == 0 and result.cases[0].status is IntegrationStatus.EXACT
    assert result.cases[1].value == 0
    assert result.cases[2].value == 2 * sp.sqrt(a)
    assert result.coverage[0].complete and result.complete
