"""Principal-value prescription objects and normalization."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


class CauchyPrincipalValue:
    """Ordinary one-dimensional Cauchy principal value prescription."""


@dataclass(frozen=True)
class IteratedPrincipalValue:
    """Apply one-dimensional Cauchy PVs sequentially in integration order.

    ``variables`` defaults to every integration variable.  When supplied, only
    those variables use PV semantics; the remaining integrations are ordinary.
    """

    variables: tuple[sp.Symbol, ...] = ()

    def __post_init__(self) -> None:
        variables = tuple(map(sp.sympify, self.variables))
        if any(not isinstance(var, sp.Symbol) for var in variables):
            raise ValueError("IteratedPrincipalValue variables must be SymPy Symbols")
        if len(set(variables)) != len(variables):
            raise ValueError("IteratedPrincipalValue variables must be distinct")
        object.__setattr__(self, "variables", variables)


@dataclass(frozen=True)
class HypersurfacePrincipalValue:
    """PV across a regular polynomial hypersurface ``g(x)=0``.

    A global graph chart is used when available. Otherwise semialg certifies
    regularity and MultipleIntegrate builds a local implicit atlas from
    transverse gradient components. Closed concentric quadrics use an exact
    affine-radial realization of the same shared ``|g| > epsilon`` regulator.
    """

    hypersurface: sp.Expr

    def __post_init__(self) -> None:
        g = sp.expand(sp.sympify(self.hypersurface))
        if g == 0:
            raise ValueError("hypersurface polynomial must be nonzero")
        object.__setattr__(self, "hypersurface", g)


@dataclass(frozen=True)
class GeometricPrincipalValue:
    """Shared-cutoff geometric PV around one isolated singular point.

    Spherical exclusions are supported on centered disks/balls and on arbitrary
    boxes or semialgebraic domains with a semialg-certified interior neighborhood.
    Ellipsoidal exclusions may be specified by axis lengths or by a symmetric
    positive-definite shape matrix ``Q``, representing
    ``(x-c).T * Q**-1 * (x-c) < epsilon**2``.
    """

    point: tuple[sp.Expr, ...]
    exclusion: str = "sphere"
    axes: tuple[sp.Expr, ...] | None = None
    shape_matrix: sp.ImmutableMatrix | None = None

    def __post_init__(self) -> None:
        point = tuple(map(sp.sympify, self.point))
        if not point:
            raise ValueError("GeometricPrincipalValue requires a singular point")
        if self.exclusion not in {"sphere", "ellipsoid"}:
            raise ValueError("exclusion must be 'sphere' or 'ellipsoid'")
        axes = None if self.axes is None else tuple(map(sp.sympify, self.axes))
        shape_matrix = None
        if self.shape_matrix is not None:
            shape_matrix = sp.ImmutableMatrix(self.shape_matrix)
            if shape_matrix.shape != (len(point), len(point)):
                raise ValueError("shape_matrix dimension must match the geometric-PV point")
            if shape_matrix != shape_matrix.T:
                raise ValueError("shape_matrix must be symmetric")
            if shape_matrix.is_positive_definite is not True:
                raise ValueError("shape_matrix must be certifiably positive definite")
        if axes is not None and shape_matrix is not None:
            raise ValueError("use either axes or shape_matrix for ellipsoidal exclusion, not both")
        if axes is not None:
            if len(axes) != len(point):
                raise ValueError("ellipsoidal exclusion requires one axis per coordinate")
            if any(axis.is_nonpositive is True for axis in axes):
                raise ValueError("ellipsoidal exclusion axes must be positive")
            if any(axis.has(sp.nan, sp.zoo, sp.oo, -sp.oo) for axis in axes):
                raise ValueError("ellipsoidal exclusion axes must be finite")
        if self.exclusion == "sphere" and (axes is not None or shape_matrix is not None):
            raise ValueError("axes and shape_matrix are only valid for ellipsoidal exclusion")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "shape_matrix", shape_matrix)


PrincipalValue = (
    bool
    | CauchyPrincipalValue
    | IteratedPrincipalValue
    | GeometricPrincipalValue
    | HypersurfacePrincipalValue
)


def _normalize_principal_value(value: PrincipalValue | None):
    if value in (False, None):
        return None
    if value is True:
        return CauchyPrincipalValue()
    if isinstance(
        value,
        (
            CauchyPrincipalValue,
            IteratedPrincipalValue,
            GeometricPrincipalValue,
            HypersurfacePrincipalValue,
        ),
    ):
        return value
    raise TypeError("principal_value must be bool or a supported principal-value prescription")


__all__ = [
    "CauchyPrincipalValue",
    "IteratedPrincipalValue",
    "GeometricPrincipalValue",
    "HypersurfacePrincipalValue",
    "PrincipalValue",
]
