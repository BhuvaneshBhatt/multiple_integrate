"""Coordinate transformations for structured integration regions."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import sympy as sp

from multiple_integrate.recognition import _fast_simplify
from multiple_integrate.region_types import (
    AnnulusRegion,
    BallRegion,
    DiskRegion,
    EllipsoidRegion,
    Region,
    SphericalShellRegion,
)

_STRATEGY_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


@dataclass(frozen=True)
class CoordinateTransform:
    """Change-of-variables data for structured coordinate transforms."""

    source_vars: tuple[sp.Symbol, ...]
    target_vars: tuple[sp.Symbol, ...]
    forward_map: tuple[sp.Expr, ...]
    jacobian: sp.Expr
    target_ranges: tuple[tuple, ...]

    def apply(self, expr: sp.Expr) -> sp.Expr:
        # The transformed integrand is f(T(u)) * |det DT(u)|.
        subs = dict(zip(self.source_vars, self.forward_map, strict=True))
        transformed = sp.sympify(expr).subs(subs, simultaneous=True) * self.jacobian
        with suppress(
            TypeError,
            ValueError,
            NotImplementedError,
            ZeroDivisionError,
            sp.PolynomialError,
        ):
            transformed = sp.trigsimp(sp.factor_terms(sp.cancel(transformed)))
        return _fast_simplify(transformed)


def _standard_ball_ranges(vars_: tuple[sp.Symbol, ...], radius: sp.Expr) -> tuple[tuple, ...]:
    """SymPy-style inner-first iterated ranges for a standard ball."""
    outer_first = []
    sumsq = sp.Integer(0)
    for i, v in enumerate(vars_):
        if i == 0:
            outer_first.append((v, -radius, radius))
        else:
            rad = sp.sqrt(radius**2 - sumsq)
            outer_first.append((v, -rad, rad))
        sumsq += v**2
    return tuple(reversed(outer_first))


def _polar_transform(variables, center, inner_radius, outer_radius) -> CoordinateTransform:
    r = sp.Dummy("r", nonnegative=True, real=True)
    theta = sp.Dummy("theta", real=True)
    forward = (r * sp.cos(theta), r * sp.sin(theta))
    if any(shift != 0 for shift in center):
        forward = tuple(
            shift + coordinate for shift, coordinate in zip(center, forward, strict=True)
        )
    return CoordinateTransform(
        source_vars=tuple(variables),
        target_vars=(theta, r),
        forward_map=forward,
        jacobian=r,
        target_ranges=((theta, 0, 2 * sp.pi), (r, inner_radius, outer_radius)),
    )


def _polar_disk_transform(region: DiskRegion) -> CoordinateTransform:
    return _polar_transform(region.variables, region.center, 0, region.radius)


def _spherical_transform(variables, center, inner_radius, outer_radius) -> CoordinateTransform:
    r = sp.Dummy("r", nonnegative=True, real=True)
    phi = sp.Dummy("phi", real=True)
    theta = sp.Dummy("theta", real=True)
    forward = (r * sp.sin(phi) * sp.cos(theta), r * sp.sin(phi) * sp.sin(theta), r * sp.cos(phi))
    if any(shift != 0 for shift in center):
        forward = tuple(
            shift + coordinate for shift, coordinate in zip(center, forward, strict=True)
        )
    return CoordinateTransform(
        source_vars=tuple(variables),
        target_vars=(theta, phi, r),
        forward_map=forward,
        jacobian=r**2 * sp.sin(phi),
        target_ranges=((theta, 0, 2 * sp.pi), (phi, 0, sp.pi), (r, inner_radius, outer_radius)),
    )


def _ball_spherical_map(region: BallRegion) -> CoordinateTransform | None:
    if region.dimension != 3 or len(region.variables) != 3:
        return None
    return _spherical_transform(region.variables, region.center, 0, region.radius)


def _affine_region_transform(region: Region) -> CoordinateTransform | None:
    if isinstance(region, EllipsoidRegion):
        vars_ = tuple(region.variables_nd)
        uvars = tuple(sp.Dummy(f"u{i}", real=True) for i in range(len(vars_)))
        jac = sp.Integer(1)
        forward = []
        for a, c, u in zip(region.axes, region.center, uvars, strict=True):
            jac *= sp.Abs(sp.sympify(a))
            forward.append(sp.sympify(c) + sp.sympify(a) * u)
        return CoordinateTransform(
            source_vars=vars_,
            target_vars=tuple(reversed(uvars)),
            forward_map=tuple(forward),
            jacobian=jac,
            target_ranges=_standard_ball_ranges(uvars, sp.Integer(1)),
        )
    if isinstance(region, AnnulusRegion):
        return _polar_transform(
            region.variables_xy, (0, 0), region.inner_radius, region.outer_radius
        )
    if isinstance(region, SphericalShellRegion) and len(region.variables_nd) == 3:
        return _spherical_transform(
            region.variables_nd, (0, 0, 0), region.inner_radius, region.outer_radius
        )
    return None


def _try_transform(
    transform: CoordinateTransform,
    expr: sp.Expr,
    *,
    assumptions,
    principal_value,
) -> sp.Expr | None:
    transformed_expr = transform.apply(expr)
    if any(v in transformed_expr.free_symbols for v in transform.source_vars):
        return None
    from multiple_integrate.core import integrate

    try:
        result = integrate(
            transformed_expr,
            *transform.target_ranges,
            assumptions=assumptions,
            principal_value=principal_value,
        )
    except (TypeError, ValueError, NotImplementedError, ZeroDivisionError):
        return None
    return None if sp.sympify(result).has(sp.Integral) else result


def _try_region_transform(
    region: Region, expr: sp.Expr, *, assumptions, principal_value
) -> sp.Expr | None:
    if isinstance(region, DiskRegion):
        val = _try_transform(
            _polar_disk_transform(region),
            expr,
            assumptions=assumptions,
            principal_value=principal_value,
        )
        if val is not None:
            return val
    if isinstance(region, BallRegion):
        tfm = _ball_spherical_map(region)
        if tfm is not None:
            val = _try_transform(
                tfm,
                expr,
                assumptions=assumptions,
                principal_value=principal_value,
            )
            if val is not None:
                return val
    tfm = _affine_region_transform(region)
    if tfm is not None:
        val = _try_transform(
            tfm,
            expr,
            assumptions=assumptions,
            principal_value=principal_value,
        )
        if val is not None:
            return val
    return None
