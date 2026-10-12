"""Cells solve simple geometry symbolically while retaining transition fibers."""

import pytest
import sympy as sp

from multiple_integrate import (
    IntegrationOutcome,
    IntegrationPolicy,
    ParameterCell,
    StrategyBudget,
    StrategyBudgetExceeded,
    integrate_cells,
)
from multiple_integrate.symbolic_geometry import symbolic_boxes

x, y, a, b = sp.symbols("x y a b", real=True)


@pytest.mark.parametrize(
    "measure,expected_zero", [("ambient", 0), ("intrinsic", 1), (0, 1), (1, 0)]
)
def test_quadratic_empty_point_segment_transition(measure, expected_zero):
    r = integrate_cells(
        1,
        region=x * x <= a,
        variables=(x,),
        parameters=(a,),
        measure_dimension=measure,
        cells=(ParameterCell(a < 0), ParameterCell(sp.Eq(a, 0), {a: 0}), ParameterCell(a > 0)),
    )
    assert r.cases[0].value == 0
    assert r.cases[1].value == expected_zero
    if measure == 0:
        assert r.cases[2].outcome is IntegrationOutcome.UNKNOWN
    else:
        assert r.cases[2].value == 2 * sp.sqrt(a)
        assert r.complete


def test_disconnected_to_connected_cells_and_boundary_coincidence():
    r = integrate_cells(
        1,
        region=sp.And(x * x >= 1 - a, x * x <= 1),
        variables=(x,),
        parameters=(a,),
        cells=(
            ParameterCell(a < 0),
            ParameterCell(sp.Eq(a, 0), {a: 0}),
            ParameterCell(sp.And(a > 0, a < 1)),
            ParameterCell(a >= 1),
        ),
    )
    assert [c.value for c in r.cases] == [0, 0, 2 - 2 * sp.sqrt(1 - a), 2]
    assert r.complete


def test_symbolic_box_volume_and_mixed_moments():
    formula = sp.And(x >= 0, x <= a, y >= -b, y <= b)
    r = integrate_cells(
        x * x * y * y,
        region=formula,
        variables=(x, y),
        parameters=(a, b),
        assumptions=b > 0,
        cells=(ParameterCell(a > 0), ParameterCell(a <= 0)),
    )
    assert r.cases[0].value == 2 * a**3 * b**3 / 9
    assert r.cases[1].value == 0 and r.complete


def test_symbolic_fixed_coordinate_intrinsic_curve():
    formula = sp.And(sp.Eq(y, b), x >= 0, x <= a)
    r = integrate_cells(
        x + y,
        region=formula,
        variables=(x, y),
        parameters=(a, b),
        assumptions=a > 0,
        cells=(ParameterCell(a > 0),),
        measure_dimension=1,
    )
    assert sp.simplify(r.cases[0].value - a * a / 2 - a * b) == 0 and r.complete
    ambient = integrate_cells(
        1,
        region=formula,
        variables=(x, y),
        parameters=(a, b),
        assumptions=a > 0,
        cells=(ParameterCell(a > 0),),
    )
    assert ambient.cases[0].value == 0


def test_quadratic_zero_dimensional_counting_keeps_both_roots():
    r = integrate_cells(
        x * x,
        region=sp.Eq(x * x, a),
        variables=(x,),
        parameters=(a,),
        assumptions=a > 0,
        cells=(ParameterCell(a > 0),),
        measure_dimension=0,
    )
    assert r.cases[0].value == 2 * a and r.complete


def test_strict_quadratic_inequality_removes_transition_point():
    r = integrate_cells(
        1,
        region=x * x < a,
        variables=(x,),
        parameters=(a,),
        cells=(ParameterCell(sp.Eq(a, 0), {a: 0}), ParameterCell(a > 0)),
        measure_dimension="intrinsic",
    )
    assert r.cases[0].value == 0 and r.cases[1].value == 2 * sp.sqrt(a)


def test_bounds_are_intersected_with_cell_geometry():
    r = integrate_cells(
        1,
        region=sp.And(x >= 0, x <= a),
        variables=(x,),
        bounds={x: (0, 1)},
        parameters=(a,),
        cells=(ParameterCell(sp.And(a > 0, a < 1)), ParameterCell(a >= 1)),
    )
    assert r.cases[0].value == a and r.cases[1].value == 1


def test_parameter_renaming_and_equivalent_formula():
    first = integrate_cells(
        1,
        region=x * x <= a,
        variables=(x,),
        parameters=(a,),
        assumptions=a > 0,
        cells=(ParameterCell(a > 0),),
    )
    second = integrate_cells(
        1,
        region=sp.Ge(b - y * y, 0),
        variables=(y,),
        parameters=(b,),
        assumptions=b > 0,
        cells=(ParameterCell(b > 0),),
    )
    assert first.cases[0].value.xreplace({a: b}) == second.cases[0].value
    assert first.cases[0].condition.xreplace({a: b}) == second.cases[0].condition


@pytest.mark.parametrize("formula", [x**3 <= a, sp.sin(x) <= a, x + y <= a])
def test_non_axiswise_or_higher_degree_geometry_stays_unresolved(formula):
    variables = (x, y) if formula.has(y) else (x,)
    assert symbolic_boxes(formula, variables, assumptions=a > 0) is None


def test_uncertain_root_order_requires_a_finer_cell():
    assert symbolic_boxes(sp.And(x >= a, x <= b), (x,)) is None


def test_symbolic_range_budget_is_checked_before_lowering():
    with pytest.raises(StrategyBudgetExceeded):
        symbolic_boxes(
            sp.And(x >= 0, x <= a),
            (x,),
            assumptions=a > 0,
            policy=IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=0)),
        )


def test_open_endpoint_union_does_not_double_count_a_point():
    formula = sp.Or(sp.And(x >= 0, x <= a), sp.Eq(x, 0))
    r = integrate_cells(
        1,
        region=formula,
        variables=(x,),
        parameters=(a,),
        cells=(ParameterCell(sp.Eq(a, 0), {a: 0}),),
        measure_dimension=0,
    )
    assert r.cases[0].value == 1
