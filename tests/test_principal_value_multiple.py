from __future__ import annotations

import pytest
import sympy as sp

from multiple_integrate import (
    CauchyPrincipalValue,
    GeometricPrincipalValue,
    IteratedPrincipalValue,
    integrate,
    integrate_result,
)
from multiple_integrate.region_types import BoxRegion, DiskRegion, EllipsoidRegion
from multiple_integrate.results import IntegrationStatus


def test_one_dimensional_pv_supports_multiple_interior_poles():
    x = sp.symbols("x", real=True)
    value = integrate(
        1 / (x * (x - 1)),
        (x, -2, 2),
        principal_value=CauchyPrincipalValue(),
    )
    assert sp.simplify(value + sp.log(3)) == 0


def test_parameter_conditioned_real_pole_uses_real_symmetric_cutoff():
    x, a = sp.symbols("x a", real=True)
    value = integrate(
        1 / (x - a),
        (x, -1, 1),
        assumptions=sp.And(a > -1, a < 1),
        principal_value=CauchyPrincipalValue(),
    )
    expected = sp.log(1 - a) - sp.log(1 + a)
    assert sp.simplify(value - expected) == 0
    assert not value.has(sp.I)


def test_iterated_pv_applies_to_selected_inner_variable():
    x, y = sp.symbols("x y", real=True)
    value = integrate(
        1 / (x - y),
        (x, -1, 1),
        (y, -sp.Rational(1, 2), sp.Rational(1, 2)),
        principal_value=IteratedPrincipalValue((x,)),
    )
    assert value == 0


def test_iterated_pv_rejects_unknown_variable():
    x, y = sp.symbols("x y", real=True)
    try:
        integrate(1 / x, (x, -1, 1), principal_value=IteratedPrincipalValue((y,)))
    except ValueError as exc:
        assert "not integration variables" in str(exc)
    else:
        raise AssertionError("missing iterated-PV variable should be rejected")


def test_geometric_disk_pv_uses_one_shared_spherical_cutoff():
    x, y = sp.symbols("x y", real=True)
    disk = DiskRegion(
        ((y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)),
        radius=1,
        center=(0, 0),
    )
    value = integrate(
        x / (x**2 + y**2),
        disk,
        principal_value=GeometricPrincipalValue((0, 0)),
    )
    assert value == 0


def test_geometric_ellipsoid_pv_supports_translated_isolated_point():
    x, y = sp.symbols("x y", real=True)
    region = EllipsoidRegion((), variables_nd=(x, y), axes=(2, 3), center=(1, -2))
    expr = (x - 1) / (((x - 1) / 2) ** 2 + ((y + 2) / 3) ** 2)
    value = integrate(
        expr,
        region=region,
        principal_value=GeometricPrincipalValue((1, -2), exclusion="ellipsoid"),
    )
    assert value == 0


def test_structured_result_reports_geometric_pv_provenance():
    x, y = sp.symbols("x y", real=True)
    disk = DiskRegion(
        ((y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)),
        radius=1,
        center=(0, 0),
    )
    result = integrate_result(
        x / (x**2 + y**2),
        disk,
        principal_value=GeometricPrincipalValue((0, 0)),
    )
    assert result.status is IntegrationStatus.PRINCIPAL_VALUE
    assert result.method == "geometric_principal_value"
    assert "owner:multiple_integrate.principal_value" in result.provenance


def test_geometric_pv_supports_spherical_puncture_in_arbitrary_box():
    x, y = sp.symbols("x y", real=True)
    box = BoxRegion(((x, -1, 1), (y, -1, 1)))
    value = integrate(
        x / (x**2 + y**2),
        region=box,
        principal_value=GeometricPrincipalValue((0, 0)),
    )
    assert value == 0


def test_geometric_pv_box_puncture_preserves_regular_volume_limit():
    x, y = sp.symbols("x y", real=True)
    box = BoxRegion(((x, -1, 1), (y, -1, 1)))
    value = integrate(
        1,
        region=box,
        principal_value=GeometricPrincipalValue((0, 0)),
    )
    assert value == 4


@pytest.mark.slow
def test_geometric_pv_handles_semialgebraic_ellipsoid():
    x, y = sp.symbols("x y", real=True)
    condition = sp.And(x >= -2, x <= 2, y >= -2, y <= 2, x + y <= 3)
    expr = x / (x**2 + y**2)
    spherical = integrate(
        expr,
        region=condition,
        variables=(x, y),
        principal_value=GeometricPrincipalValue((0, 0)),
    )
    ellipsoidal = integrate(
        expr,
        region=condition,
        variables=(x, y),
        principal_value=GeometricPrincipalValue((0, 0), exclusion="ellipsoid", axes=(2, 1)),
    )
    assert spherical == ellipsoidal
    assert spherical != 0
    assert not spherical.has(sp.Integral)


def test_general_ellipsoidal_pv_requires_explicit_axes():
    x, y = sp.symbols("x y", real=True)
    condition = sp.And(x >= -1, x <= 1, y >= -1, y <= 1)
    with pytest.raises(ValueError, match="requires axes"):
        integrate(
            1,
            region=condition,
            variables=(x, y),
            principal_value=GeometricPrincipalValue((0, 0), exclusion="ellipsoid"),
        )


def test_general_geometric_pv_rejects_uncertified_boundary_singularity():
    x, y = sp.symbols("x y", real=True)
    condition = sp.And(x >= 0, x <= 1, y >= -1, y <= 1)
    with pytest.raises(NotImplementedError, match="symmetric neighborhood"):
        integrate(
            1 / (x**2 + y**2),
            region=condition,
            variables=(x, y),
            principal_value=GeometricPrincipalValue((0, 0)),
        )


def test_geometric_pv_supports_four_dimensional_structured_ellipsoid():
    variables = sp.symbols("x0:4", real=True)
    region = EllipsoidRegion(
        (),
        variables_nd=variables,
        axes=(1, 1, 1, 1),
        center=(0, 0, 0, 0),
    )
    radius_sq = sum(var**2 for var in variables)
    value = integrate(
        variables[0] / radius_sq**3,
        region=region,
        principal_value=GeometricPrincipalValue((0, 0, 0, 0), exclusion="ellipsoid"),
    )
    assert value == 0


def test_geometric_pv_sphere_moments_scale_to_five_dimensions():
    variables = sp.symbols("x0:5", real=True)
    region = EllipsoidRegion(
        (),
        variables_nd=variables,
        axes=(1, 1, 1, 1, 1),
        center=(0, 0, 0, 0, 0),
    )
    radius_sq = sum(var**2 for var in variables)
    value = integrate(
        variables[0] / radius_sq**3,
        region=region,
        principal_value=GeometricPrincipalValue((0, 0, 0, 0, 0), exclusion="ellipsoid"),
    )
    assert value == 0


def test_four_dimensional_local_regular_volume_uses_exact_sphere_measure():
    variables = sp.symbols("x0:4", real=True)
    region = EllipsoidRegion(
        (),
        variables_nd=variables,
        axes=(1, 1, 1, 1),
        center=(0, 0, 0, 0),
    )
    value = integrate(
        1,
        region=region,
        principal_value=GeometricPrincipalValue((0, 0, 0, 0), exclusion="ellipsoid"),
    )
    assert sp.simplify(value - sp.pi**2 / 2) == 0


def test_general_four_dimensional_box_accepts_spherical_puncture():
    variables = sp.symbols("x0:4", real=True)
    box = BoxRegion(tuple((var, -1, 1) for var in variables))
    value = integrate(
        1,
        region=box,
        principal_value=GeometricPrincipalValue((0, 0, 0, 0)),
    )
    assert value == 16


def test_general_four_dimensional_box_uses_exact_reflection_pv_cancellation():
    variables = sp.symbols("q0:4", real=True)
    box = BoxRegion(tuple((var, -1, 1) for var in variables))
    radius_sq = sum(var**2 for var in variables)
    value = integrate(
        variables[0] / radius_sq**3,
        region=box,
        principal_value=GeometricPrincipalValue((0, 0, 0, 0)),
    )
    assert value == 0


def test_geometric_pv_accepts_positive_definite_shape_matrix():
    prescription = GeometricPrincipalValue(
        (0, 0),
        exclusion="ellipsoid",
        shape_matrix=sp.ImmutableMatrix([[2, 1], [1, 2]]),
    )
    assert prescription.shape_matrix == sp.ImmutableMatrix([[2, 1], [1, 2]])


def test_geometric_pv_rejects_invalid_shape_matrices():
    with pytest.raises(ValueError, match="symmetric"):
        GeometricPrincipalValue(
            (0, 0), exclusion="ellipsoid", shape_matrix=sp.ImmutableMatrix([[1, 1], [0, 1]])
        )
    with pytest.raises(ValueError, match="positive definite"):
        GeometricPrincipalValue(
            (0, 0), exclusion="ellipsoid", shape_matrix=sp.ImmutableMatrix([[1, 2], [2, 1]])
        )
    with pytest.raises(ValueError, match="either axes or shape_matrix"):
        GeometricPrincipalValue(
            (0, 0),
            exclusion="ellipsoid",
            axes=(1, 1),
            shape_matrix=sp.ImmutableMatrix([[1, 0], [0, 1]]),
        )


def test_shape_matrix_matches_axis_semantics_for_diagonal_ellipsoid():
    x, y = sp.symbols("x y", real=True)
    box = BoxRegion(((x, -3, 3), (y, -3, 3)))
    axes_pv = GeometricPrincipalValue((0, 0), exclusion="ellipsoid", axes=(2, 1))
    matrix_pv = GeometricPrincipalValue(
        (0, 0), exclusion="ellipsoid", shape_matrix=sp.ImmutableMatrix([[4, 0], [0, 1]])
    )
    assert integrate(1, region=box, principal_value=axes_pv) == 36
    assert integrate(1, region=box, principal_value=matrix_pv) == 36


def test_rotated_ellipsoid_preserves_central_odd_cancellation():
    x, y = sp.symbols("x y", real=True)
    shape = sp.ImmutableMatrix([[2, 1], [1, 2]])
    delta = sp.ImmutableMatrix([x, y])
    quadratic = sp.simplify((delta.T * shape.inv() * delta)[0])
    box = BoxRegion(((x, -2, 2), (y, -2, 2)))
    prescription = GeometricPrincipalValue((0, 0), exclusion="ellipsoid", shape_matrix=shape)
    assert integrate(x / quadratic, region=box, principal_value=prescription) == 0


def test_rotated_shape_matrix_box_cutoff_is_supported():
    x, y = sp.symbols("x y", real=True)
    box = BoxRegion(((x, -2, 2), (y, -2, 2)))
    shape = sp.ImmutableMatrix([[2, 1], [1, 2]])
    prescription = GeometricPrincipalValue((0, 0), exclusion="ellipsoid", shape_matrix=shape)
    assert integrate(1, region=box, principal_value=prescription) == 16


def test_rotated_shape_matrix_box_uses_central_inversion_cancellation():
    x, y = sp.symbols("x y", real=True)
    box = BoxRegion(((x, -2, 2), (y, -2, 2)))
    shape = sp.ImmutableMatrix([[2, 1], [1, 2]])
    delta = sp.ImmutableMatrix([x, y])
    quadratic = sp.simplify((delta.T * shape.inv() * delta)[0])
    value = integrate(
        x / quadratic,
        region=box,
        principal_value=GeometricPrincipalValue((0, 0), exclusion="ellipsoid", shape_matrix=shape),
    )
    assert value == 0
