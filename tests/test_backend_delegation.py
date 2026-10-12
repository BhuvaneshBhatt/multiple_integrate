import types

import sympy as sp

import multiple_integrate.adapters.funcprops as fp_adapter
import multiple_integrate.adapters.semialg as sa_adapter
import multiple_integrate.recognition as recognition
import multiple_integrate.region_strategies as region_strategies
import multiple_integrate.strategy_common as strategy_common
from multiple_integrate.region_types import BoxRegion, Region


def test_semialg_measure_adapter_uses_public_signature(monkeypatch):
    x = sp.Symbol("x", real=True)
    seen = {}

    def measure(condition, variables):
        seen["condition"] = condition
        seen["variables"] = tuple(variables)
        return sp.Integer(7)

    monkeypatch.setattr(
        sa_adapter, "_module", lambda: types.SimpleNamespace(semialgebraic_measure=measure)
    )
    region = Region(((x, 0, 7),))
    assert sa_adapter.semialgebraic_measure(region) == 7
    assert seen["variables"] == (x,)
    assert sp.simplify(seen["condition"] ^ ((x >= 0) & (x <= 7))) is sp.false


def test_semialgebraic_domain_moment_adapter_uses_public_signature(monkeypatch):
    x = sp.Symbol("x", real=True)
    seen = {}

    def moment(condition, variables, *, integrand):
        seen["condition"] = condition
        seen["variables"] = tuple(variables)
        seen["integrand"] = integrand
        return sp.Rational(1, 3)

    monkeypatch.setattr(sa_adapter, "_module", lambda: types.SimpleNamespace(region_moment=moment))
    region = Region(((x, 0, 1),))
    assert sa_adapter.region_moment(region, x**2) == sp.Rational(1, 3)
    assert seen["variables"] == (x,)
    assert seen["integrand"] == x**2


def test_native_volume_precedes_semialg(monkeypatch):
    x = sp.Symbol("x", real=True)
    region = BoxRegion(((x, 0, 2),))
    monkeypatch.setattr(strategy_common, "semialg_measure", lambda region: sp.Integer(99))
    assert strategy_common._const_result(sp.Integer(3), region, [(x, 0, 2)]) == 6


def test_semialg_measure_fills_missing_native_volume(monkeypatch):
    x = sp.Symbol("x", real=True)
    region = Region(((x, 0, 2),))
    monkeypatch.setattr(strategy_common, "semialg_measure", lambda region: sp.Integer(2))
    assert strategy_common._const_result(sp.Integer(3), region, [(x, 0, 2)]) == 6


def test_semialgebraic_domain_moment_fills_missing_native_shortcut(monkeypatch):
    x = sp.Symbol("x", real=True)
    region = Region(((x, 0, 1),))
    monkeypatch.setattr(
        region_strategies,
        "semialgebraic_domain_moment",
        lambda region, expr, assumptions=None: sp.Rational(1, 3),
    )
    assert region_strategies._region_shortcut(region, x**2) == sp.Rational(1, 3)


def test_funcprops_symmetry_is_used_after_local_fast_path(monkeypatch):
    x = sp.Symbol("x", real=True)
    f = sp.Function("f")
    monkeypatch.setattr(recognition, "funcprops_symmetry", lambda expr, var: "even")
    assert recognition._is_even_function(f(x), x)


def test_funcprops_range_is_used_before_local_critical_point_search(monkeypatch):
    x = sp.Symbol("x", real=True)
    monkeypatch.setattr(
        recognition, "funcprops_range", lambda expr, var, domain=None: sp.Interval(-2, 5)
    )
    monkeypatch.setattr(
        recognition,
        "_find_critical_points",
        lambda *args: (_ for _ in ()).throw(AssertionError("local fallback should not run")),
    )
    assert recognition._inner_expr_range(x**3, x, -1, 2) == (-2, 5)


def test_funcprops_sign_can_discharge_positivity(monkeypatch):
    a = sp.Symbol("a", real=True)
    monkeypatch.setattr(
        strategy_common, "funcprops_sign", lambda expr, assumptions=None: "positive"
    )
    assert strategy_common._symbolically_positive(a)


def test_funcprops_adapter_normalizes_enum_and_property_result(monkeypatch):
    class FakeEnum(str, __import__("enum").Enum):
        EVEN = "even"

    fake_result = types.SimpleNamespace(primary=FakeEnum.EVEN)
    monkeypatch.setattr(
        fp_adapter,
        "_module",
        lambda: types.SimpleNamespace(function_symmetry=lambda *args, **kwargs: fake_result),
    )
    x = sp.Symbol("x", real=True)
    assert fp_adapter.function_symmetry(sp.cos(x), x) == "even"


def test_optional_backend_absence_is_conservative(monkeypatch):
    monkeypatch.setattr(fp_adapter, "_module", lambda: None)
    monkeypatch.setattr(sa_adapter, "_module", lambda: None)
    x = sp.Symbol("x", real=True)
    assert fp_adapter.function_range(x**2, x, domain=sp.Interval(-1, 1)) is None
    assert sa_adapter.semialgebraic_measure(Region(((x, 0, 1),))) is None


def test_funcprops_validation_errors_are_not_hidden(monkeypatch):
    def invalid_query(*args, **kwargs):
        raise ValueError("invalid property domain")

    backend = types.SimpleNamespace(function_range=invalid_query)
    monkeypatch.setattr(fp_adapter, "_module", lambda: backend)
    x = sp.Symbol("x", real=True)
    with __import__("pytest").raises(ValueError, match="invalid property domain"):
        fp_adapter.function_range(x, x)


def test_semialg_api_errors_are_not_hidden(monkeypatch):
    def broken_measure(*args, **kwargs):
        raise TypeError("backend API mismatch")

    backend = types.SimpleNamespace(semialgebraic_measure=broken_measure)
    monkeypatch.setattr(sa_adapter, "_module", lambda: backend)
    x = sp.Symbol("x", real=True)
    with __import__("pytest").raises(TypeError, match="backend API mismatch"):
        sa_adapter.semialgebraic_measure(Region(((x, 0, 1),)))


def test_semialg_reduction_contract_rejects_missing_piece_fields(monkeypatch):
    x = sp.Symbol("x", real=True)

    class BadReduction:
        pieces = (types.SimpleNamespace(limits=((x, 0, 1),)),)
        condition = sp.And(x >= 0, x <= 1)
        variables = (x,)
        method = "bad"
        diagnostics = {}

    backend = types.SimpleNamespace(reduce_region_integral=lambda *args, **kwargs: BadReduction())
    monkeypatch.setattr(sa_adapter, "_module", lambda: backend)
    with __import__("pytest").raises(
        sa_adapter.SemialgContractError, match="missing required attribute 'integrand'"
    ):
        sa_adapter.reduce_domain(x, sp.And(x >= 0, x <= 1), variables=(x,))


def test_semialg_measure_contract_rejects_unknown_result_shape(monkeypatch):
    x = sp.Symbol("x", real=True)
    backend = types.SimpleNamespace(
        semialgebraic_measure=lambda *args, **kwargs: types.SimpleNamespace(value=1)
    )
    monkeypatch.setattr(sa_adapter, "_module", lambda: backend)
    with __import__("pytest").raises(
        sa_adapter.SemialgContractError, match="missing required attribute 'measure'"
    ):
        sa_adapter.semialgebraic_measure(Region(((x, 0, 1),)))
