from __future__ import annotations

import pytest
import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    HypersurfacePrincipalValue,
    IntegrationStatus,
    RegularizationResult,
    integrate,
    integrate_result,
    regularized_result,
)
from multiple_integrate.region_types import BoxRegion


def test_polynomial_hypersurface_pv_uses_shared_level_set_cutoff():
    x, y = sp.symbols("x y", real=True)
    box = BoxRegion(((x, -1, 1), (y, -1, 1)))
    result = regularized_result(
        1 / x,
        region=box,
        principal_value=HypersurfacePrincipalValue(x),
    )
    assert isinstance(result, RegularizationResult)
    assert result.value == 0
    assert result.status == "principal_value"
    assert result.provenance.shared_cutoff is True
    assert result.provenance.backend == "semialg_cylindrical_reduction"
    eps = result.provenance.cutoff_parameter
    assert isinstance(eps, sp.Symbol)
    assert eps.is_positive is True
    assert result.provenance.cutoff_domain.has(x**2 > eps**2)


def test_polynomial_hypersurface_pv_supports_curved_graph_locus():
    x, y = sp.symbols("x y", real=True)
    box = BoxRegion(((x, -1, 1), (y, -1, 1)))
    value = integrate(
        1 / (y - x**2),
        region=box,
        principal_value=HypersurfacePrincipalValue(y - x**2),
    )
    assert sp.simplify(value - (sp.log(4) - sp.pi)) == 0


def test_implicit_circle_hypersurface_uses_multiple_certified_charts():
    x, y = sp.symbols("x y", real=True)
    from multiple_integrate.region_types import DiskRegion

    disk = DiskRegion(
        ((y, -sp.sqrt(4 - x**2), sp.sqrt(4 - x**2)), (x, -2, 2)),
        radius=2,
        center=(0, 0),
    )
    g = x**2 + y**2 - 1
    result = regularized_result(
        1 / g,
        region=disk,
        principal_value=HypersurfacePrincipalValue(g),
    )
    assert sp.simplify(result.value - sp.pi * sp.log(3)) == 0
    assert result.provenance.backend == "semialg_certified_implicit_atlas_affine_radial"
    assert result.provenance.details["atlas_chart_count"] == 4
    assert result.provenance.details["regularity_certified"] is True
    transverse = {item["transverse"] for item in result.provenance.details["charts"]}
    assert transverse == {x, y}


def test_implicit_hypersurface_rejects_nonregular_level_set():
    x, y = sp.symbols("x y", real=True)
    from multiple_integrate.region_types import DiskRegion

    disk = DiskRegion(
        ((y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)),
        radius=1,
        center=(0, 0),
    )
    with pytest.raises(NotImplementedError, match="regular"):
        integrate(
            1 / (x**2 + y**2),
            region=disk,
            principal_value=HypersurfacePrincipalValue(x**2 + y**2),
        )


def test_hadamard_finite_part_reuses_implicit_hypersurface_atlas():
    x, y = sp.symbols("x y", real=True)
    from multiple_integrate.region_types import DiskRegion

    disk = DiskRegion(
        ((y, -sp.sqrt(4 - x**2), sp.sqrt(4 - x**2)), (x, -2, 2)),
        radius=2,
        center=(0, 0),
    )
    g = x**2 + y**2 - 1
    result = regularized_result(
        g**-2,
        region=disk,
        regularization=HadamardFinitePart(hypersurface=g),
    )
    assert sp.simplify(result.value + 4 * sp.pi / 3) == 0
    assert result.provenance.backend == "semialg_certified_implicit_atlas_affine_radial"
    assert result.provenance.details["atlas_chart_count"] == 4
    assert result.provenance.divergent_terms


def test_hadamard_finite_part_of_symmetric_double_pole():
    x = sp.symbols("x", real=True)
    result = regularized_result(
        x**-2,
        (x, -1, 1),
        regularization=HadamardFinitePart(),
    )
    assert result.value == -2
    assert result.status == "finite_part"
    eps = result.provenance.cutoff_parameter
    assert isinstance(eps, sp.Symbol)
    assert eps.is_positive is True
    assert result.provenance.asymptotic_expansion == -2 + 2 / eps
    assert result.provenance.divergent_terms == (2 / eps,)
    assert result.provenance.constant_term == -2
    assert result.provenance.backend in {"asymptotic.multiseries", "sympy.series_fallback"}


def test_hadamard_finite_part_supports_singular_endpoint():
    x = sp.symbols("x", real=True)
    value = integrate(
        x**-2,
        (x, 0, 1),
        regularization=HadamardFinitePart(),
    )
    assert value == -1


def test_hadamard_hypersurface_reuses_semialg_cutoff_geometry():
    x, y = sp.symbols("x y", real=True)
    box = BoxRegion(((x, -1, 1), (y, -1, 1)))
    result = regularized_result(
        x**-2,
        region=box,
        regularization=HadamardFinitePart(hypersurface=x),
    )
    assert result.value == -4
    assert result.provenance.backend == "semialg_cylindrical_reduction"
    assert result.provenance.singular_locus == sp.Eq(x, 0)
    assert result.provenance.divergent_terms


def test_integrate_result_marks_finite_part_status():
    x = sp.symbols("x", real=True)
    result = integrate_result(
        x**-2,
        (x, -1, 1),
        regularization=HadamardFinitePart(),
    )
    assert result.status is IntegrationStatus.FINITE_PART
    assert result.value == -2
    assert result.diagnostics["regularization"] == "HadamardFinitePart"


def test_regularization_and_principal_value_are_mutually_exclusive():
    x = sp.symbols("x", real=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        integrate(
            1 / x,
            (x, -1, 1),
            principal_value=True,
            regularization=HadamardFinitePart(),
        )


def test_hadamard_requires_detectable_singularity():
    x = sp.symbols("x", real=True)
    with pytest.raises(ValueError, match="no finite singularities"):
        integrate(x, (x, -1, 1), regularization=HadamardFinitePart())
