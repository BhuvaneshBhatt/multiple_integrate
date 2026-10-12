"""Comparison, affine normalization and complete domain decomposition proofs."""

import pytest
import sympy as sp

from multiple_integrate import IntegrationOutcome, integrate_result
from multiple_integrate.adapters.semialg import DomainPiece, DomainReduction
from multiple_integrate.convergence_evidence import absolute_rule
from multiple_integrate.region_types import BoxRegion, UnionRegion

x, y, a = sp.symbols("x y a", real=True)
FULL = ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))


@pytest.mark.parametrize(
    "f, ranges",
    [
        (sp.sin(x * y) / (1 + x * x + y * y) ** 2, FULL),
        (sp.cos(x + y) * sp.exp(-x * x - y * y), FULL),
        (sp.exp(sp.I * x * y) / (1 + x * x + y * y) ** 2, FULL),
        (a / (1 + x * x + y * y) ** 2, FULL),
        (sp.exp(-x * x - y * y) + (1 + x * x + y * y) ** -2, FULL),
        ((1 + x * x + y * y + x**4) ** -2, FULL),
        ((4 * (x - 2) ** 2 + 9 * (y + 1) ** 2) ** (-sp.Rational(1, 2)), ((x, 1, 3), (y, -2, 0))),
        (x * x / (x * x + y * y) ** sp.Rational(3, 2), ((x, -1, 1), (y, -1, 1))),
        (sp.sin(x * y) / (x * x + y * y) ** sp.Rational(3, 2), ((x, -1, 1), (y, -1, 1))),
        ((3 * (x - 2)) ** (-sp.Rational(1, 2)), ((x, 2, 5),)),
        (x**-2, ((x, 3, sp.oo),)),
        (1 / (x * sp.log(x) ** 2), ((x, 2, sp.oo),)),
        (1 / (x * sp.log(x) ** 2), ((x, 0, sp.Rational(1, 2)),)),
    ],
)
def test_comparison_prerequisites_prove_absolute_convergence(f, ranges):
    assert absolute_rule(f, ranges)[0] is True


@pytest.mark.parametrize(
    "f,ranges",
    [
        (sp.sin(x * y) / (1 + x * x + y * y), FULL),
        (sp.exp(x * y) / (1 + x * x + y * y) ** 2, FULL),
        (sp.exp(sp.I * (x + sp.I * y)) / (1 + x * x + y * y) ** 2, FULL),
        (x * x / (x * x + y * y) ** 2, ((x, -1, 1), (y, -1, 1))),
        (1 / (x * sp.log(x)), ((x, 2, sp.oo),)),
        (1 / (x * sp.log(x) ** 2), ((x, 0, 1),)),
    ],
)
def test_comparison_refuses_an_unproved_or_broken_majorant(f, ranges):
    assert absolute_rule(f, ranges)[0] is not True


def test_comparison_proves_existence_without_manufacturing_an_answer(monkeypatch):
    monkeypatch.setattr(
        "multiple_integrate.api.integration.integrate", lambda f, *rs, **kw: sp.Integral(f, *rs)
    )
    r = integrate_result((1 + x * x + y * y + x**4) ** -2, *FULL)
    assert r.outcome is IntegrationOutcome.UNKNOWN
    assert r.ordinary_convergent is True and r.absolute_convergent is True


def test_scaled_radial_divergence_requires_nonzero_coefficient():
    q = 4 * (x - 2) ** 2 + 9 * (y + 1) ** 2
    ranges = ((x, 1, 3), (y, -2, 0))
    assert absolute_rule(a / q, ranges)[0] is None
    assert absolute_rule(a / q, ranges, assumptions=sp.Ne(a, 0))[0] is False


def test_native_union_propagates_absolute_existence():
    region = UnionRegion(pieces=(BoxRegion(((x, 0, 1),)), BoxRegion(((x, 2, 3),))))
    r = integrate_result(x ** (-sp.Rational(1, 2)), region=region)
    assert r.absolute_convergent is True and r.ordinary_convergent is True


def test_complete_backend_decomposition_propagates_piece_proofs(monkeypatch):
    formula = sp.Or(sp.And(x >= 0, x <= 1), sp.And(x >= 2, x <= 3))
    pieces = (
        DomainPiece(sp.exp(x), ((x, 0, 1),), "test", {}),
        DomainPiece(sp.exp(x), ((x, 2, 3),), "test", {}),
    )
    reduction = DomainReduction(formula, (x,), pieces, "test", {})
    monkeypatch.setattr("multiple_integrate.core.reduce_domain", lambda *args, **kw: reduction)
    r = integrate_result(sp.exp(x), region=formula, variables=(x,))
    assert r.absolute_convergent is True
    assert any(c.scope == "absolute_convergence" for c in r.coverage)


def test_finite_sum_does_not_infer_convergence_from_cancellation():
    f = 1 / x - 1 / (x + 1)
    assert absolute_rule(f, ((x, -2, 2),))[0] is not True


@pytest.mark.parametrize("limits", [(0, sp.Rational(1, 2)), (2, sp.oo)])
@pytest.mark.parametrize(
    "first,second,converges",
    [
        (-sp.Rational(3, 2), sp.Rational(1, 2), False),
        (-sp.Rational(5, 2), sp.Rational(3, 2), False),
        (-sp.Rational(5, 2), sp.Rational(1, 2), True),
        (sp.Rational(1, 2), -sp.Rational(5, 2), True),
    ],
)
def test_opposite_log_branches_use_total_magnitude_exponent(limits, first, second, converges):
    density = sp.log(x) ** first * (-sp.log(x)) ** second / x
    claim, _ = absolute_rule(density, ((x, *limits),))
    # t = |log(x)| converts the absolute integral to integral t**(first+second) dt.
    assert (first + second < -1) == converges
    assert claim is True if converges else claim is not True


@pytest.mark.parametrize("limits", [(0, sp.Rational(1, 2)), (2, sp.oo)])
def test_divergent_mixed_log_branches_cannot_certify_an_unevaluated_result(monkeypatch, limits):
    monkeypatch.setattr(
        "multiple_integrate.api.integration.integrate", lambda f, *rs, **kw: sp.Integral(f, *rs)
    )
    density = sp.sqrt(-sp.log(x)) / (x * sp.log(x) ** sp.Rational(3, 2))
    result = integrate_result(density, (x, *limits))
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert result.absolute_convergent is not True
    assert result.ordinary_convergent is not True
