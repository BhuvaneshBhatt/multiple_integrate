from dataclasses import replace

import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    HypersurfacePrincipalValue,
    regularized_result,
    verify_regularization_result,
)
from multiple_integrate.region_types import BoxRegion


def test_hypersurface_scaling_preserves_principal_value():
    x, y = sp.symbols("x y", real=True)
    box = BoxRegion(((x, -1, 1), (y, -2, 2)))
    base = regularized_result(
        1 / x,
        region=box,
        principal_value=HypersurfacePrincipalValue(x),
    )
    scaled = regularized_result(
        1 / x,
        region=box,
        principal_value=HypersurfacePrincipalValue(7 * x),
    )
    assert base.value == scaled.value == 0


def test_certificate_detects_changed_value():
    x = sp.Symbol("x", real=True)
    result = regularized_result(
        x**-2,
        (x, -1, 1),
        regularization=HadamardFinitePart(),
    )
    tampered = replace(result, value=result.value + 1)
    verification = verify_regularization_result(tampered)
    assert not verification.valid
    assert "constant_term" in dict(verification.checks)
    assert dict(verification.checks)["constant_term"] is False


def test_certificate_detects_changed_divergent_terms():
    x = sp.Symbol("x", real=True)
    result = regularized_result(
        x**-2,
        (x, -1, 1),
        regularization=HadamardFinitePart(),
    )
    provenance = replace(result.provenance, divergent_terms=())
    tampered = replace(result, provenance=provenance)
    verification = verify_regularization_result(tampered)
    assert not verification.valid
    assert dict(verification.checks)["divergent_terms"] is False


def test_certificate_detects_changed_expansion():
    x = sp.Symbol("x", real=True)
    result = regularized_result(
        x**-2,
        (x, -1, 1),
        regularization=HadamardFinitePart(),
    )
    provenance = replace(
        result.provenance,
        asymptotic_expansion=result.provenance.asymptotic_expansion + 1,
    )
    tampered = replace(result, provenance=provenance)
    verification = verify_regularization_result(tampered)
    assert not verification.valid
    assert dict(verification.checks)["asymptotic_expansion"] is False
