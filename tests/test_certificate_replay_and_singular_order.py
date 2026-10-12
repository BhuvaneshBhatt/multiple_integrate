"""Regression tests for certificate replay, assumptions, and singular interval order."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import sympy as sp

import multiple_integrate.singularities as singularities
from multiple_integrate import (
    HadamardFinitePart,
    integrate,
    regularized_result,
    verify_regularization_result,
)
from multiple_integrate.regularization import RegularizationProvenance, RegularizationResult

x, a, b = sp.symbols("x a b", real=True)


@pytest.mark.parametrize("assumptions", [False, sp.false, [False]])
def test_false_assumptions_are_consistently_rejected(assumptions):
    with pytest.raises(ValueError, match="inconsistent assumptions"):
        integrate(x, (x, 0, 1), assumptions=assumptions)


def test_algebraic_singularities_are_numerically_ordered(monkeypatch):
    monkeypatch.setattr(
        singularities,
        "function_singularities",
        lambda *args, **kwargs: {
            "poles": (sp.Eq(x, 2), sp.Eq(x, sp.sqrt(2))),
        },
    )
    assert singularities.singular_points(1 / x, x, 0, 3) == (sp.sqrt(2), 2)
    limits = []

    def record(expr, limit, **opts):
        limits.append(limit)
        return sp.Integer(0)

    monkeypatch.setattr(singularities, "_integrate", record)
    singularities.split_integral(1 / x, x, 0, 3, {})
    assert limits == [(x, 0, sp.sqrt(2)), (x, sp.sqrt(2), 2), (x, 2, 3)]


def test_guarded_singularity_uses_caller_assumptions(monkeypatch):
    monkeypatch.setattr(
        singularities,
        "function_singularities",
        lambda *args, **kwargs: {
            "poles": (SimpleNamespace(condition=sp.Eq(x, a), guard=a > 0),),
        },
    )
    assert singularities.singular_points(
        1 / (x - a), x, 0, 1, assumptions=sp.And(a > 0, a < 1)
    ) == (a,)


def test_uncertified_singular_order_declines_split(monkeypatch):
    monkeypatch.setattr(
        singularities,
        "function_singularities",
        lambda *args, **kwargs: {
            "poles": (sp.Eq(x, a), sp.Eq(x, b)),
        },
    )
    assumptions = sp.And(a > 0, a < 1, b > 0, b < 1)
    assert (
        singularities.singular_points(1 / ((x - a) * (x - b)), x, 0, 1, assumptions=assumptions)
        == ()
    )
    assert (
        singularities.split_integral(1 / ((x - a) * (x - b)), x, 0, 1, {}, assumptions=assumptions)
        is None
    )


def test_zero_stored_expansion_does_not_bypass_replay():
    result = regularized_result(x**-2, (x, -1, 1), regularization=HadamardFinitePart())
    tampered = replace(
        result, provenance=replace(result.provenance, asymptotic_expansion=sp.Integer(0))
    )
    verification = verify_regularization_result(tampered)
    assert not verification.valid
    assert dict(verification.checks)["asymptotic_expansion"] is False


def test_correct_zero_expansion_is_valid():
    epsilon = sp.Symbol("epsilon", positive=True)
    provenance = RegularizationProvenance(
        "test",
        "finite_part",
        cutoff_parameter=epsilon,
        cutoff_integral=sp.Integer(0),
        asymptotic_expansion=sp.Integer(0),
        constant_term=sp.Integer(0),
    )
    result = RegularizationResult(
        sp.Integer(0), "finite_part", "test", HadamardFinitePart(), provenance
    )
    assert verify_regularization_result(result).valid


def test_stored_constant_is_verified_with_cutoff_evidence():
    result = regularized_result(x**-2, (x, -1, 1), regularization=HadamardFinitePart())
    tampered = replace(result, provenance=replace(result.provenance, constant_term=sp.Integer(0)))
    verification = verify_regularization_result(tampered)
    assert not verification.valid
    assert dict(verification.checks)["provenance_constant_term"] is False


@pytest.mark.parametrize("details", [{}, {"shape_matrix": sp.eye(2)}])
def test_geometry_alone_cannot_certify_a_regularized_value(details):
    provenance = RegularizationProvenance("test", "finite_part", details=details)
    result = RegularizationResult(
        sp.Integer(123), "finite_part", "test", HadamardFinitePart(), provenance
    )
    verification = verify_regularization_result(result)
    assert not verification.valid
    assert "value evidence" in verification.failures[-1]
