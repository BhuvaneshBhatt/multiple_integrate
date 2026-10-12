"""Radial parametrizations used by principal-value integration strategies."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .semialg_geometry import ellipsoid_geometry, intrinsic_geometry_cover


@dataclass(frozen=True)
class RadialParametrization:
    """Hyperspherical parametrization and intrinsic metric data."""

    variables: tuple[sp.Symbol, ...]
    center: tuple[sp.Expr, ...]
    affine_matrix: sp.ImmutableMatrix
    radial_variable: sp.Symbol
    angular_variables: tuple[sp.Symbol, ...]
    mapping: tuple[tuple[sp.Symbol, sp.Expr], ...]
    angular_ranges: tuple[tuple[sp.Symbol, sp.Expr, sp.Expr], ...]
    jacobian_factor: sp.Expr

    @property
    def dimension(self) -> int:
        """Return the ambient dimension."""
        return len(self.variables)


def radial_parametrization(
    variables: tuple[sp.Symbol, ...],
    center: tuple[sp.Expr, ...],
    *,
    axes: tuple[sp.Expr, ...] | None = None,
    affine_matrix: sp.MatrixBase | None = None,
) -> RadialParametrization:
    """Build a hyperspherical parametrization for an affine radial domain."""
    variables = tuple(variables)
    center = tuple(map(sp.sympify, center))
    dimension = len(variables)
    if dimension < 2:
        raise ValueError("hyperspherical coordinates require dimension >= 2")
    if len(center) != dimension:
        raise ValueError("center dimension must match variables")
    if axes is not None and affine_matrix is not None:
        raise ValueError("provide either axes or affine_matrix, not both")
    if affine_matrix is None:
        axis_values = (
            tuple(sp.Integer(1) for _ in variables)
            if axes is None
            else tuple(map(sp.sympify, axes))
        )
        if len(axis_values) != dimension:
            raise ValueError("axes dimension must match variables")
        affine = sp.ImmutableMatrix(sp.diag(*axis_values))
    else:
        affine = sp.ImmutableMatrix(affine_matrix)
        if affine.shape != (dimension, dimension):
            raise ValueError("affine matrix dimension must match variables")
        if sp.simplify(affine.det()) == 0:
            raise ValueError("affine matrix must be nonsingular")

    # semialg owns parameter symbols and their certified ranges.  We retain the
    # supplied affine factor because principal-value cutoffs depend on that map,
    # not merely on the ellipsoid represented by A A^T.
    shape = sp.simplify(affine * affine.T)
    geometry = ellipsoid_geometry(center, shape)
    cover = intrinsic_geometry_cover(geometry, variables)
    if len(cover.charts) != 1:
        raise ValueError("expected one radial chart")
    chart = cover.charts[0]
    radial = chart.parameters[0]
    angles = tuple(chart.parameters[1:])

    direction = []
    sine_prefix = sp.Integer(1)
    for index in range(dimension):
        if index == dimension - 1:
            coordinate = sine_prefix
        else:
            coordinate = sine_prefix * sp.cos(angles[index])
            sine_prefix *= sp.sin(angles[index])
        direction.append(coordinate)

    mapped = sp.ImmutableMatrix(center) + radial * affine * sp.ImmutableMatrix(direction)
    mapping = tuple((var, sp.simplify(mapped[index])) for index, var in enumerate(variables))
    jacobian = sp.Matrix([expr for _, expr in mapping]).jacobian((radial, *angles))
    gram = sp.simplify(jacobian.T * jacobian)
    metric_factor = sp.simplify(sp.sqrt(sp.det(gram)))
    # All but the azimuthal angle lie in [0, pi], so these sine factors are
    # nonnegative on the certified chart domain.
    for angle in angles[:-1]:
        metric_factor = metric_factor.xreplace({sp.Abs(sp.sin(angle)): sp.sin(angle)})

    return RadialParametrization(
        variables=variables,
        center=center,
        affine_matrix=affine,
        radial_variable=radial,
        angular_variables=angles,
        mapping=mapping,
        angular_ranges=tuple(chart.bounds[1:]),
        jacobian_factor=sp.simplify(metric_factor),
    )


__all__ = ["RadialParametrization", "radial_parametrization"]
