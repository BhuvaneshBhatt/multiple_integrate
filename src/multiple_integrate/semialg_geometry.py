"""Translation from integration-domain models to canonical semialg geometry."""

from __future__ import annotations

import sympy as sp


def _semialg():
    import semialg

    return semialg


def ellipsoid_geometry(center, shape):
    """Return a canonical semialg ellipsoid for an affine radial domain."""
    return _semialg().Ellipsoid(tuple(map(sp.sympify, center)), sp.ImmutableMatrix(shape))


def domain_geometry(domain):
    """Return canonical semialg geometry for a structured integration domain.

    Integration-only domains such as graph and iterated descriptions return
    ``None`` because they do not correspond to one canonical geometric object.
    """
    from .region_types import (
        AffineSimplexRegion,
        AnnulusRegion,
        BallRegion,
        BoxRegion,
        DiskRegion,
        EllipsoidRegion,
        SimplexRegion,
        SphericalShellRegion,
    )

    sa = _semialg()
    if isinstance(domain, BoxRegion):
        return sa.Box(tuple((sp.sympify(lo), sp.sympify(hi)) for _, lo, hi in domain.ranges))
    if isinstance(domain, SimplexRegion):
        dimension = domain.dimension
        origin = tuple(sp.Integer(0) for _ in range(dimension))
        vertices = [origin]
        for index in range(dimension):
            vertex = [sp.Integer(0)] * dimension
            vertex[index] = sp.Integer(1)
            vertices.append(tuple(vertex))
        return sa.Simplex(vertices)
    if isinstance(domain, AffineSimplexRegion):
        origin = tuple(domain.shifts)
        vertices = [origin]
        for index, scale in enumerate(domain.scales):
            vertex = list(origin)
            vertex[index] = sp.simplify(vertex[index] + scale)
            vertices.append(tuple(vertex))
        return sa.Simplex(vertices)
    if isinstance(domain, (DiskRegion, BallRegion)):
        return sa.Ball(tuple(domain.center), domain.radius)
    if isinstance(domain, EllipsoidRegion):
        axes = tuple(map(sp.sympify, domain.axes))
        return sa.Ellipsoid(tuple(domain.center), sp.diag(*(axis**2 for axis in axes)))
    if isinstance(domain, AnnulusRegion):
        return sa.SphericalShell((0, 0), (domain.inner_radius, domain.outer_radius))
    if isinstance(domain, SphericalShellRegion):
        center = tuple(sp.Integer(0) for _ in domain.variables_nd)
        return sa.SphericalShell(center, (domain.inner_radius, domain.outer_radius))
    return None


def intrinsic_geometry_cover(geometry, variables, *, dimension=None):
    """Return semialg-owned intrinsic charts for canonical geometry."""
    from semialg.parametric_geometry import intrinsic_parametric_cover

    return intrinsic_parametric_cover(
        geometry, tuple(variables), dimension=dimension, require_verified=True
    )


def semialgebraic_domain(domain):
    """Return the semialgebraic representation of a structured integration domain."""
    geometry = domain_geometry(domain)
    if geometry is None:
        return None
    return geometry.as_semialgebraic_region(tuple(domain.variables))


def intrinsic_domain_cover(domain, *, dimension=None):
    """Return verified intrinsic charts for a structured integration domain."""
    geometry = domain_geometry(domain)
    if geometry is None:
        return None
    return geometry.intrinsic_parametric_cover(
        tuple(domain.variables), dimension=dimension, require_verified=True
    )


__all__ = [
    "domain_geometry",
    "ellipsoid_geometry",
    "intrinsic_geometry_cover",
    "semialgebraic_domain",
    "intrinsic_domain_cover",
]


def radial_geometry_signature(geometry, variables):
    """Describe canonical radial geometry for integration-only moment reduction."""
    sa = _semialg()
    n = len(variables)
    if isinstance(geometry, sa.Ball):
        return tuple(geometry.center), sp.eye(n), sp.S.Zero, sp.sympify(geometry.radius)
    if isinstance(geometry, sa.SphericalShell):
        return (
            tuple(geometry.center),
            sp.eye(n),
            sp.sympify(geometry.inner_radius),
            sp.sympify(geometry.outer_radius),
        )
    if isinstance(geometry, sa.Ellipsoid):
        shape = sp.ImmutableMatrix(geometry.shape_matrix)
        if shape.shape != (n, n) or shape != shape.T or shape.det() == 0:
            return None
        return tuple(geometry.center), shape.inv(), sp.S.Zero, sp.S.One
    return None
