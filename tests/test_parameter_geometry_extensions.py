"""Certified multiple-parameter and coupled affine geometry, including pole fibers."""

import pytest
import sympy as sp

from multiple_integrate import (
    IntegrationPolicy,
    IntegrationStatus,
    ParameterCell,
    StrategyBudget,
    StrategyBudgetExceeded,
    integrate_cells,
)
from multiple_integrate.transition_cells import linear_quadratic_cells

x, y, a, b = sp.symbols("x y a b", real=True)
policy = IntegrationPolicy(max_singular_queries=0)


def test_two_parameter_interval_partition_tracks_order_and_equality_only():
    formula = sp.And(x >= a, x <= b)
    cells = linear_quadratic_cells(parameters=(a, b), region=formula, variables=(x,))
    assert len(cells) == 3
    r = integrate_cells(x * x, parameters=(a, b), region=formula, variables=(x,), policy=policy)
    assert r.complete and r.absolute_convergent is True
    for lo, hi in [(-2, 3), (1, 2), (2, 2), (3, 1)]:
        expected = sp.Rational(hi**3 - lo**3, 3) if lo < hi else 0
        assert r.value.subs({a: lo, b: hi}, simultaneous=True) == expected


def test_coupled_parameter_affine_boxes_track_rank_drop():
    formula = sp.And(a * x + y >= 0, a * x + y <= 1, x - y >= 0, x - y <= 1)
    r = integrate_cells(1, parameters=(a,), region=formula, variables=(x, y), policy=policy)
    assert r.complete and r.absolute_convergent is True
    for coefficient in [-3, -1, 0, 2]:
        assert r.value.subs(a, coefficient) == (
            0 if coefficient == -1 else sp.Rational(1, abs(coefficient + 1))
        )
    assert any(c.condition == sp.Eq(a, -1) for c in r.cases)


def test_coupled_two_parameter_shear_and_size_in_supplied_cells():
    formula = sp.And(x >= 0, x <= 1, y - a * x >= 0, y - a * x <= b)
    cells = [ParameterCell(b > 0), ParameterCell(sp.Eq(b, 0)), ParameterCell(b < 0)]
    r = integrate_cells(
        x + y, parameters=(a, b), region=formula, variables=(x, y), cells=cells, policy=policy
    )
    assert r.complete
    assert sp.simplify(r.value.subs({a: 2, b: 3}) - 9) == 0
    assert r.value.subs({a: 2, b: -1}) == 0 and r.value.subs({a: 2, b: 0}) == 0


def test_automatic_multi_parameter_shear_cells_use_boundary_coincidence():
    formula = sp.And(x >= 0, x <= 1, y - a * x >= 0, y - a * x <= b)
    r = integrate_cells(1, parameters=(a, b), region=formula, variables=(x, y), policy=policy)
    assert r.complete
    assert r.value.subs({a: 2, b: 3}) == 3
    assert r.value.subs({a: -2, b: -1}) == 0


@pytest.mark.parametrize("as_ranges", [False, True])
def test_parameter_coefficient_poles_remain_unknown_not_empty_or_infinite(as_ranges):
    options = {"parameters": (a,), "policy": policy}
    if as_ranges:
        r = integrate_cells(1, (x, 0, 1 / a), **options)
        assert r.value.subs(a, -2) == -sp.Rational(1, 2)
    else:
        r = integrate_cells(1, region=sp.And(x >= 0, x <= 1 / a), variables=(x,), **options)
        assert r.value.subs(a, -2) == 0
    assert r.complete is False and r.absolute_convergent is None
    assert r.value.subs(a, 2) == sp.Rational(1, 2)
    fiber = next(c for c in r.cases if c.condition == sp.Eq(a, 0))
    assert fiber.status is IntegrationStatus.UNKNOWN
    assert any(
        "pole fiber" in reason
        for child in r.diagnostics["cell_results"]
        for reason in child.limitations
    )


def test_hyperplane_cell_budget_declines_before_evaluation():
    limited = IntegrationPolicy(semialg=StrategyBudget(max_pieces=2))
    with pytest.raises(StrategyBudgetExceeded, match="max_pieces"):
        linear_quadratic_cells(
            parameters=(a, b), region=sp.And(x >= a, x <= b), variables=(x,), policy=limited
        )


@pytest.mark.parametrize("formula", [x * y * (x + y) <= a, x <= sp.sin(a + b)])
def test_nonlinear_multi_parameter_transitions_remain_explicitly_unsupported(formula):
    with pytest.raises(NotImplementedError):
        linear_quadratic_cells(
            parameters=(a, b), region=formula, variables=(x, y) if formula.has(y) else (x,)
        )


def test_multiple_parameter_degree_drop_keeps_constant_sign_fibers():
    r = integrate_cells(
        1,
        parameters=(a, b),
        region=a * x + b >= 0,
        variables=(x,),
        bounds={x: (-1, 1)},
        policy=policy,
    )
    assert r.complete
    for aa, bb in [(0, -1), (0, 0), (0, 1), (1, 0), (-1, 0), (1, 2)]:
        expected = 0 if aa == 0 and bb < 0 else 2 if aa == 0 or bb >= abs(aa) else 1
        assert r.value.subs({a: aa, b: bb}, simultaneous=True) == expected


def test_coupled_rational_map_keeps_rank_and_pole_fibers_separate():
    formula = sp.And(x / a + y >= 0, x / a + y <= 1, x - y >= 0, x - y <= 1)
    r = integrate_cells(1, parameters=(a,), region=formula, variables=(x, y), policy=policy)
    assert r.complete is False
    assert r.value.subs(a, -2) == 2 and r.value.subs(a, 2) == sp.Rational(2, 3)
    assert r.value.subs(a, -1) == 0
    assert (
        next(c for c in r.cases if c.condition == sp.Eq(a, 0)).status is IntegrationStatus.UNKNOWN
    )


@pytest.mark.parametrize(
    "assumptions,expression,expected",
    [
        (a > 0, (a - b) / a, True),
        (a < 0, (a - b) / a, False),
        (a >= 0, 1 / a, False),
        (a < 0, -1 / a, True),
    ],
)
def test_rational_cell_signs_require_a_strict_denominator_sign(assumptions, expression, expected):
    from multiple_integrate.assumptions import _normalize_assumptions
    from multiple_integrate.cell_proofs import nonnegative

    context = _normalize_assumptions(sp.And(assumptions, a > b))
    assert nonnegative(expression, context, strict=True) is expected


def test_null_piece_evidence_cannot_certify_a_different_source_region(monkeypatch):
    from multiple_integrate import integrate_result
    from multiple_integrate.planner import record_coverage

    def fake_integrate(*args, **kwargs):
        record_coverage(
            "ambient_null_set",
            True,
            "semialgebraic_domain_reduction",
            source_formula=sp.Eq(x, 0),
            source_variables=(x,),
            source_density=sp.S.One,
        )
        return sp.S.Zero

    monkeypatch.setattr("multiple_integrate.api.integration.integrate", fake_integrate)
    r = integrate_result(1, region=sp.And(x >= 0, x <= 1), variables=(x,), policy=policy)
    assert r.absolute_convergent is None


def test_certified_whole_ambient_null_set_has_absolute_evidence():
    from multiple_integrate import integrate_result

    r = integrate_result(1, region=sp.Eq(x, y), variables=(x, y), policy=policy)
    assert r.value == 0 and r.absolute_convergent is True and r.ordinary_convergent is True
