import types

import pytest
import sympy as sp

import multiple_integrate.adapters.semialg as sa
import multiple_integrate.core as core
from multiple_integrate import IntegrationStatus, integrate, integrate_result
from multiple_integrate.adapters.semialg import DomainPiece, DomainReduction


def test_raw_semialgebraic_condition_requires_explicit_variables():
    x = sp.Symbol("x", real=True)
    with pytest.raises(ValueError, match="variables="):
        sa.semialgebraic_domain(x >= 0)


def test_semialgebraic_region_object_supplies_formula_and_variables():
    x = sp.Symbol("x", real=True)

    class FakeRegion:
        formula = sp.And(x >= 0, x <= 1)
        variables = (x,)

        def quantifier_free_formula(self):
            return self.formula

    formula, variables = sa.semialgebraic_domain(FakeRegion())
    assert formula == sp.And(x >= 0, x <= 1)
    assert variables == (x,)


def test_reduce_adapter_uses_public_structural_semialg_api(monkeypatch):
    x, y = sp.symbols("x y", real=True)
    calls = []

    def reduce_region_integral(integrand, condition, variables, *, bounds=None):
        calls.append((integrand, condition, tuple(variables), bounds))
        piece = types.SimpleNamespace(
            integrand=integrand,
            limits=((y, 0, 1), (x, y**2, y)),
            method="cad_variable_order_cell_integral",
            diagnostics={"certified": True},
        )
        return types.SimpleNamespace(
            condition=condition,
            variables=(y, x),
            pieces=(piece,),
            method="cad_variable_order_cell_integration",
            diagnostics={"piece_count": 1},
        )

    monkeypatch.setattr(
        sa,
        "_module",
        lambda: types.SimpleNamespace(reduce_region_integral=reduce_region_integral),
    )
    condition = sp.And(y >= 0, y <= 1, x >= y**2, x <= y)
    reduced = sa.reduce_domain(1, condition, variables=(x, y))
    assert calls == [(1, condition, (x, y), None)]
    assert reduced.method == "cad_variable_order_cell_integration"
    assert reduced.pieces[0].limits == ((y, 0, 1), (x, y**2, y))
    assert reduced.pieces[0].diagnostics["certified"] is True


def test_integrate_feeds_cad_piece_back_to_native_planner(monkeypatch):
    x, y = sp.symbols("x y", real=True)
    condition = sp.And(y >= 0, y <= 1, x >= y**2, x <= y)
    reduced = DomainReduction(
        condition=condition,
        variables=(y, x),
        pieces=(
            DomainPiece(
                integrand=sp.Integer(1),
                limits=((x, y**2, y), (y, 0, 1)),
                method="cad_variable_order_cell_integral",
                diagnostics={"certified": True},
            ),
        ),
        method="cad_variable_order_cell_integration",
        diagnostics={"piece_count": 1},
    )
    monkeypatch.setattr(core, "reduce_domain", lambda *args, **kwargs: reduced)
    assert integrate(1, region=condition, variables=(x, y)) == sp.Rational(1, 6)


def test_semialgebraic_boolean_pieces_are_summed_once(monkeypatch):
    x = sp.Symbol("x", real=True)
    condition = sp.Or(sp.And(x >= 0, x <= 1), sp.And(x >= 2, x <= 3))
    reduced = DomainReduction(
        condition=condition,
        variables=(x,),
        pieces=(
            DomainPiece(x, ((x, 0, 1),), "cell", {}),
            DomainPiece(x, ((x, 2, 3),), "cell", {}),
        ),
        method="complete_cad_boolean_cell_integration",
        diagnostics={"boolean_cell_decomposition": True, "piece_count": 2},
    )
    monkeypatch.setattr(core, "reduce_domain", lambda *args, **kwargs: reduced)
    assert integrate(x, region=condition, variables=(x,)) == 3


def test_semialgebraic_result_reports_semialg_cad_provenance(monkeypatch):
    x = sp.Symbol("x", real=True)
    condition = sp.And(x >= 0, x <= 1)
    reduced = DomainReduction(
        condition=condition,
        variables=(x,),
        pieces=(DomainPiece(x, ((x, 0, 1),), "cell", {}),),
        method="complete_cad_boolean_cell_integration",
        diagnostics={"piece_count": 1},
    )
    monkeypatch.setattr(core, "reduce_domain", lambda *args, **kwargs: reduced)
    result = integrate_result(x, region=condition, variables=(x,))
    assert result.value == sp.Rational(1, 2)
    assert result.status is IntegrationStatus.EXACT
    assert result.method == "semialg_cad_reduction"
    assert result.diagnostics["strategy_owner"] == "semialg"


def test_semialgebraic_region_can_be_positional(monkeypatch):
    x = sp.Symbol("x", real=True)

    class FakeRegion:
        formula = sp.And(x >= 0, x <= 1)
        variables = (x,)

    reduced = DomainReduction(
        condition=FakeRegion.formula,
        variables=(x,),
        pieces=(DomainPiece(1, ((x, 0, 1),), "cell", {}),),
        method="one_dimensional_cell_integration",
        diagnostics={},
    )
    monkeypatch.setattr(core, "reduce_domain", lambda *args, **kwargs: reduced)
    assert integrate(1, FakeRegion()) == 1


def test_explicit_ranges_and_region_are_mutually_exclusive():
    x = sp.Symbol("x", real=True)
    with pytest.raises(ValueError, match="either explicit ranges or region"):
        integrate(1, (x, 0, 1), region=x >= 0, variables=(x,))


def test_raw_semialgebraic_condition_can_be_positional_with_variables(monkeypatch):
    x = sp.Symbol("x", real=True)
    condition = sp.And(x >= 0, x <= 1)
    reduced = DomainReduction(
        condition=condition,
        variables=(x,),
        pieces=(DomainPiece(x, ((x, 0, 1),), "cell", {}),),
        method="one_dimensional_cell_integration",
        diagnostics={},
    )
    monkeypatch.setattr(core, "reduce_domain", lambda *args, **kwargs: reduced)
    assert integrate(x, condition, variables=(x,)) == sp.Rational(1, 2)
