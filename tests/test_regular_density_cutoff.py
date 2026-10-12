"""A null zero set is sufficient only for a regular bounded density."""

from dataclasses import replace

import pytest
import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    HypersurfacePrincipalValue,
    regularized_result,
    verify_regularization_result,
)
from multiple_integrate.regular_density import regular_density_prerequisites

x, y, a = sp.symbols("x y a", real=True)
ranges = ((x, -1, 1), (y, -1, 1))


@pytest.mark.parametrize("g", [y * y - x**3, x * x + y * y, x * y, x * x, sp.Integer(2)])
@pytest.mark.parametrize("kind", ["pv", "fp"])
def test_regular_density_handles_singular_polynomial_geometry(g, kind):
    kwargs = (
        {"principal_value": HypersurfacePrincipalValue(g)}
        if kind == "pv"
        else {"regularization": HadamardFinitePart(hypersurface=g)}
    )
    result = regularized_result(1 + x * x, *ranges, **kwargs)
    assert result.value == sp.Rational(16, 3)
    assert result.provenance.backend == "dominated_convergence"
    assert result.provenance.cutoff_integral is None
    verification = verify_regularization_result(result)
    assert verification.valid and verification.strength == "dominated_convergence"
    assert not verify_regularization_result(replace(result, value=result.value + 1)).valid


@pytest.mark.parametrize(
    "density,g,box",
    [
        (1 / x, x, ranges),
        (1 / (x * x + y * y), x * x + y * y, ranges),
        (1, x - x, ranges),
        (1, a * (x * x + y * y), ranges),
        (1, x, ((x, -sp.oo, sp.oo),)),
        (1, x, ((x, 1, -1),)),
    ],
)
def test_shortcut_refuses_broken_prerequisites(density, g, box):
    assert not regular_density_prerequisites(density, box, g)


def test_polynomial_nonzero_parameter_requires_proof():
    assert regular_density_prerequisites(1, ranges, a * (x * x + y * y), assumptions=sp.Ne(a, 0))
    assert not regular_density_prerequisites(1, ranges, a * (x * x + y * y))


def test_geometry_tampering_fails_verification():
    result = regularized_result(
        1, *ranges, principal_value=HypersurfacePrincipalValue(y * y - x**3)
    )
    bad = replace(result, provenance=replace(result.provenance, regulator=x))
    assert not verify_regularization_result(bad).valid
