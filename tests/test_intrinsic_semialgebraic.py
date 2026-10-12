import pytest
import sympy as sp

import multiple_integrate.adapters.semialg as sa
import multiple_integrate.core as core
from multiple_integrate import integrate, integrate_result
from multiple_integrate.adapters.semialg import (
    IntrinsicPiece,
    IntrinsicReduction,
)


def _regular_reduction(x, value=None):
    if value is None:
        value = sp.sqrt(2)
    piece = IntrinsicPiece(
        integrand=value,
        limits=((x, sp.Integer(0), sp.Integer(1)),),
        dimension=1,
        ambient_dimension=2,
        metric_factor=value,
        cell_index=(1, 2),
        method="semialg_intrinsic_cad_cell",
        diagnostics={"certified_bounds": True, "regular": True},
    )
    return IntrinsicReduction(
        condition=sp.true,
        variables=(x, sp.Symbol("y", real=True)),
        dimension=1,
        pieces=(piece,),
        singular_strata=(),
        method="semialg_intrinsic_cad_stratification",
        diagnostics={"regular_stratum_count": 1, "singular_stratum_count": 0},
    )


def test_intrinsic_adapter_preserves_metric_weighted_chart():
    x, y = sp.symbols("x y", real=True)
    reduced = sa.reduce_intrinsic_domain(1, sp.And(sp.Eq(y, x), x >= 0, x <= 1), variables=(x, y))
    assert reduced is not None
    assert reduced.certified
    assert reduced.dimension == 1
    assert reduced.pieces
    piece = reduced.pieces[0]
    assert sp.simplify(piece.metric_factor - sp.sqrt(2)) == 0
    assert sp.simplify(piece.integrand - sp.sqrt(2)) == 0
    assert piece.jacobian == sp.ImmutableMatrix([[1], [1]])
    assert piece.gram_matrix == sp.ImmutableMatrix([[2]])
    assert piece.diagnostics["chart_source"] == "cad_intrinsic"


def test_intrinsic_adapter_uses_public_semialg_cover(monkeypatch):
    x, y = sp.symbols("x y", real=True)
    calls = []
    original = sa._intrinsic_parametric_cover

    def wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(sa, "_intrinsic_parametric_cover", wrapped)
    reduced = sa.reduce_intrinsic_domain(1, sp.And(sp.Eq(y, x), x >= 0, x <= 1), variables=(x, y))
    assert reduced is not None
    assert calls


def test_intrinsic_curve_piece_is_evaluated_by_integrate(monkeypatch):
    x, y = sp.symbols("x y", real=True)
    reduced = _regular_reduction(x)
    monkeypatch.setattr(core, "reduce_intrinsic_domain", lambda *args, **kwargs: reduced)
    value = integrate(1, region=sp.Eq(y, x), variables=(x, y), measure_dimension="intrinsic")
    assert value == sp.sqrt(2)


def test_intrinsic_surface_chart_uses_multiple_integrate_planner(monkeypatch):
    u, v, z = sp.symbols("u v z", real=True)
    piece = IntrinsicPiece(
        integrand=sp.Integer(3),
        limits=((v, 0, 2), (u, 0, 1)),
        dimension=2,
        ambient_dimension=3,
        metric_factor=sp.Integer(3),
        cell_index=(2, 2, 1),
        method="semialg_intrinsic_cad_cell",
        diagnostics={"certified_bounds": True, "regular": True},
    )
    reduced = IntrinsicReduction(
        condition=sp.true,
        variables=(u, v, z),
        dimension=2,
        pieces=(piece,),
        singular_strata=(),
        method="semialg_intrinsic_cad_stratification",
        diagnostics={},
    )
    monkeypatch.setattr(core, "reduce_intrinsic_domain", lambda *args, **kwargs: reduced)
    assert integrate(1, region=sp.Eq(z, u + v), variables=(u, v, z), measure_dimension=2) == 6


def test_intrinsic_zero_dimensional_piece_uses_direct_value(monkeypatch):
    x = sp.Symbol("x", real=True)
    piece = IntrinsicPiece(
        integrand=sp.Integer(1),
        limits=(),
        dimension=0,
        ambient_dimension=1,
        metric_factor=sp.Integer(1),
        cell_index=(1,),
        method="semialg_intrinsic_cad_cell",
        diagnostics={"certified_bounds": True, "regular": True},
        direct_value=sp.Integer(7),
    )
    reduced = IntrinsicReduction(
        condition=sp.Eq(x, 2),
        variables=(x,),
        dimension=0,
        pieces=(piece,),
        singular_strata=(),
        method="semialg_intrinsic_cad_stratification",
        diagnostics={},
    )
    monkeypatch.setattr(core, "reduce_intrinsic_domain", lambda *args, **kwargs: reduced)
    assert integrate(x + 5, region=sp.Eq(x, 2), variables=(x,), measure_dimension=0) == 7


def test_intrinsic_singular_target_stratum_fails_conservatively(monkeypatch):
    x, y = sp.symbols("x y", real=True)
    reduced = IntrinsicReduction(
        condition=sp.Eq(y**2, x**3),
        variables=(x, y),
        dimension=1,
        pieces=(),
        singular_strata=(
            {"cell_index": (1, 1), "dimension": 1, "reasons": ("uncertified regularity",)},
        ),
        method="semialg_intrinsic_cad_stratification",
        diagnostics={},
    )
    monkeypatch.setattr(core, "reduce_intrinsic_domain", lambda *args, **kwargs: reduced)
    with pytest.raises(NotImplementedError, match="singular target-dimensional strata"):
        integrate(1, region=sp.Eq(y**2, x**3), variables=(x, y), measure_dimension="intrinsic")


def test_intrinsic_result_reports_semialg_ownership(monkeypatch):
    x, y = sp.symbols("x y", real=True)
    monkeypatch.setattr(
        core, "reduce_intrinsic_domain", lambda *args, **kwargs: _regular_reduction(x)
    )
    result = integrate_result(
        1, region=sp.Eq(y, x), variables=(x, y), measure_dimension="intrinsic"
    )
    assert result.value == sp.sqrt(2)
    assert result.method == "semialg_intrinsic_reduction"
    assert result.diagnostics["strategy_owner"] == "semialg"
    assert result.diagnostics["measure_dimension"] == "intrinsic"


def test_nonambient_measure_requires_region():
    x = sp.Symbol("x", real=True)
    with pytest.raises(ValueError, match="non-ambient measure_dimension"):
        integrate(1, (x, 0, 1), measure_dimension="intrinsic")
