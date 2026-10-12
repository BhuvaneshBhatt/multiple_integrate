"""Fixed large-radius rational finite parts with a shared finite-pole cutoff."""

import pytest
import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    IntegrationOutcome,
    integrate,
    regularized_result,
    verify_regularization_result,
)
from multiple_integrate.regularization import _divergent_terms

x = sp.Symbol("x", real=True)


@pytest.mark.parametrize(
    "f,lo,hi,expected",
    [
        (x, 0, sp.oo, 0),
        (x * x, 2, sp.oo, -sp.Rational(8, 3)),
        (1 / x, 1, sp.oo, 0),
        (1 / x, 2, sp.oo, -sp.log(2)),
        (1 / x**2, 1, sp.oo, 1),
        (1 / x**2, -sp.oo, -1, 1),
        (1 / (1 + x * x), -sp.oo, sp.oo, sp.pi),
        (1 / x**2, -sp.oo, sp.oo, 0),
        (1 / (x - 1) ** 2, 0, sp.oo, -1),
        (1 / (x - sp.sqrt(2)), -sp.oo, sp.oo, 0),
        (1 / (x * x - 2), -sp.oo, sp.oo, 0),
    ],
)
def test_real_rational_finite_parts_at_infinity(f, lo, hi, expected):
    result = regularized_result(f, (x, lo, hi), regularization=HadamardFinitePart())
    assert result.outcome is IntegrationOutcome.FINITE
    assert sp.simplify(result.value - expected) == 0
    assert result.provenance.details["finite_poles_share_epsilon"]
    assert "R=1/epsilon" in result.provenance.details["infinity_cutoff"]
    assert verify_regularization_result(result).valid
    assert (
        sp.simplify(integrate(f, (x, lo, hi), regularization=HadamardFinitePart()) - expected) == 0
    )


def test_nonrational_infinity_finite_part_remains_unsupported():
    result = regularized_result(sp.sin(x), (x, 0, sp.oo), regularization=HadamardFinitePart())
    assert result.outcome is IntegrationOutcome.UNSUPPORTED


def test_decaying_logarithmic_terms_are_not_subtractions():
    e = sp.Symbol("epsilon", positive=True)
    assert _divergent_terms(e * sp.log(e) + sp.log(e) + e**-1, e) == (e**-1, sp.log(e))


def test_infinite_cutoff_carries_real_logarithms():
    result = regularized_result(
        1 / (x - sp.sqrt(2)), (x, 0, sp.oo), regularization=HadamardFinitePart()
    )
    assert result.value == -sp.log(2) / 2
    assert not result.provenance.cutoff_integral.has(sp.I)


def test_explicit_cutoff_cannot_omit_another_real_pole():
    result = regularized_result(
        1 / (x * (x - 1)),
        (x, -sp.oo, sp.oo),
        regularization=HadamardFinitePart(singularities=(sp.Integer(0),)),
    )
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert "every real pole" in result.limitations[0]
