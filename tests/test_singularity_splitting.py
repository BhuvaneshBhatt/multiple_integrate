from dataclasses import dataclass

import sympy as sp

import multiple_integrate.singularities as singularities
from multiple_integrate import integrate


@dataclass(frozen=True)
class _Singularity:
    condition: sp.Expr
    guard: sp.Expr = sp.true


def test_funcprops_points_split_improper_interval(monkeypatch):
    x = sp.Symbol("x", real=True)
    monkeypatch.setattr(
        singularities,
        "function_singularities",
        lambda *args, **kwargs: {"poles": (_Singularity(sp.Eq(x, 0)),)},
    )
    value = integrate(1 / sp.sqrt(sp.Abs(x)), (x, -1, 1))
    assert value == 4


def test_removable_records_do_not_force_split(monkeypatch):
    x = sp.Symbol("x", real=True)
    monkeypatch.setattr(
        singularities,
        "function_singularities",
        lambda *args, **kwargs: {"removable": (_Singularity(sp.Eq(x, 0)),)},
    )
    assert singularities.singular_points(x, x, -1, 1) == ()


def test_polynomial_integrand_skips_funcprops_singularity_query(monkeypatch):
    x = sp.Symbol("x", real=True)

    def fail_query(*args, **kwargs):
        raise AssertionError("polynomials should not query funcprops for singularities")

    monkeypatch.setattr(singularities, "function_singularities", fail_query)
    assert singularities.singular_points(x**4 - x + 1, x, -1, 1) == ()
    assert integrate(x**4 - x + 1, (x, -1, 1)) == sp.Rational(12, 5)
