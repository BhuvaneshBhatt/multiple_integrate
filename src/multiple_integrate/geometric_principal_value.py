"""Geometric principal values around isolated singular points."""

from __future__ import annotations

import sympy as sp

from .adapters.semialg import implies as semialg_implies
from .adapters.semialg import semialgebraic_domain
from .assumptions import _integrate
from .planner import record_coverage, record_strategy
from .principal_value_prescriptions import GeometricPrincipalValue
from .radial_parametrization import radial_parametrization
from .recognition import _fast_simplify
from .region_types import BallRegion, BoxRegion, DiskRegion, EllipsoidRegion, Region
from .sphere_integrals import radial_sphere_reduction


def _shape_from_axes(axes: tuple[sp.Expr, ...]) -> sp.ImmutableMatrix:
    return sp.ImmutableMatrix(sp.diag(*(sp.sympify(axis) ** 2 for axis in axes)))


def _ellipsoid_shape(
    prescription: GeometricPrincipalValue,
    *,
    inferred_axes: tuple[sp.Expr, ...] | None = None,
) -> sp.ImmutableMatrix | None:
    if prescription.exclusion != "ellipsoid":
        return None
    if prescription.shape_matrix is not None:
        return prescription.shape_matrix
    if prescription.axes is not None:
        return _shape_from_axes(prescription.axes)
    if inferred_axes is not None:
        return _shape_from_axes(inferred_axes)
    return None


def _shape_factor(shape: sp.MatrixBase) -> sp.ImmutableMatrix:
    """Return exact ``A`` with ``A*A.T == shape`` for SPD ``shape``."""
    matrix = sp.ImmutableMatrix(shape)
    factor = matrix.cholesky()
    return sp.ImmutableMatrix(factor)


def _center_matches(region: Region, point: tuple[sp.Expr, ...]) -> bool:
    center = tuple(getattr(region, "center", ()))
    return len(center) == len(point) and all(
        sp.simplify(a - b) == 0 for a, b in zip(center, point, strict=True)
    )


def _structured_radial_data(
    region: Region,
    prescription: GeometricPrincipalValue,
):
    """Return variables, center, affine map, and outer radial bound for a local PV."""
    if isinstance(region, DiskRegion):
        if prescription.exclusion != "sphere" or not _center_matches(region, prescription.point):
            return None
        affine = sp.ImmutableMatrix(sp.eye(len(region.variables)))
        return tuple(region.variables), tuple(region.center), affine, sp.sympify(region.radius)
    if isinstance(region, BallRegion):
        if prescription.exclusion != "sphere" or not _center_matches(region, prescription.point):
            return None
        affine = sp.ImmutableMatrix(sp.eye(len(region.variables)))
        return tuple(region.variables), tuple(region.center), affine, sp.sympify(region.radius)
    if isinstance(region, EllipsoidRegion):
        if prescription.exclusion != "ellipsoid" or not _center_matches(region, prescription.point):
            return None
        axes = tuple(region.axes)
        region_shape = _shape_from_axes(axes)
        requested_shape = _ellipsoid_shape(prescription, inferred_axes=axes)
        if requested_shape is None or any(
            sp.simplify(value) != 0 for value in (requested_shape - region_shape)
        ):
            return None
        return (
            tuple(region.variables),
            tuple(region.center),
            _shape_factor(region_shape),
            sp.Integer(1),
        )
    return None


def _local_radial_pv(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    center: tuple[sp.Expr, ...],
    affine_matrix: sp.MatrixBase,
    outer_radius: sp.Expr,
    *,
    assumptions=None,
) -> sp.Expr:
    """Evaluate one n-dimensional local PV using sphere moments or a chart."""
    eps = sp.Dummy("pv_epsilon", positive=True, real=True)
    radial = sp.Dummy("pv_r", positive=True, real=True)
    reduced = radial_sphere_reduction(
        sp.sympify(expr),
        variables,
        center,
        radial,
        affine_matrix=affine_matrix,
    )
    if reduced is not None:
        record_coverage(
            "angular_domain",
            True,
            "radial_sphere_reduction",
            dimension=len(variables),
            basis="exact moments over the entire unit sphere",
        )
        cutoff = _integrate(reduced, (radial, eps, outer_radius), _mi_assumptions=assumptions)
        value = sp.limit(cutoff, eps, 0, dir="+")
        record_strategy(
            "geometric_principal_value",
            reason=f"n-D radial sphere-moment cutoff in dimension {len(variables)}",
        )
        return _fast_simplify(value)

    chart = radial_parametrization(variables, center, affine_matrix=affine_matrix)
    transformed = sp.sympify(expr).subs(dict(chart.mapping)) * chart.jacobian_factor
    ranges = chart.angular_ranges + ((chart.radial_variable, eps, outer_radius),)
    from .core import integrate

    cutoff = integrate(transformed, *ranges, assumptions=assumptions)
    value = sp.limit(cutoff, eps, 0, dir="+")
    record_strategy(
        "geometric_principal_value",
        reason=f"shared n-D hyperspherical cutoff in dimension {len(variables)}",
    )
    return _fast_simplify(value)


def _geometric_transform(expr: sp.Expr, region: Region, prescription: GeometricPrincipalValue):
    """Return the n-dimensional radial parametrization used by the cutoff."""
    data = _structured_radial_data(region, prescription)
    if data is None:
        return None
    variables, center, affine_matrix, outer_radius = data
    chart = radial_parametrization(variables, center, affine_matrix=affine_matrix)
    eps = sp.Dummy("pv_epsilon", positive=True, real=True)
    transformed = sp.sympify(expr).subs(dict(chart.mapping)) * chart.jacobian_factor
    ranges = chart.angular_ranges + ((chart.radial_variable, eps, outer_radius),)
    return transformed, ranges, eps


def _box_formula(region: BoxRegion) -> sp.Expr:
    """Return the exact Boolean formula for a Cartesian box."""
    constraints = []
    for var, lower, upper in region.ranges:
        constraints.extend((sp.Ge(var, lower), sp.Le(var, upper)))
    return sp.And(*constraints)


def _exclusion_radius_sq(
    variables: tuple[sp.Symbol, ...],
    prescription: GeometricPrincipalValue,
) -> sp.Expr:
    if prescription.exclusion == "sphere":
        return sp.Add(
            *(
                (var - center) ** 2
                for var, center in zip(variables, prescription.point, strict=True)
            )
        )
    shape = _ellipsoid_shape(prescription)
    if shape is None:
        raise ValueError(
            "ellipsoidal exclusion on a general domain requires axes=(...) or shape_matrix=... "
            "in GeometricPrincipalValue"
        )
    delta = sp.ImmutableMatrix(
        [var - center for var, center in zip(variables, prescription.point, strict=True)]
    )
    quadratic = (delta.T * shape.inv() * delta)[0]
    return sp.factor(sp.simplify(quadratic))


def _box_local_scale(
    region: BoxRegion,
    prescription: GeometricPrincipalValue,
) -> sp.Rational | None:
    """Certify one candidate ellipsoid scale directly from box support bounds."""
    if prescription.exclusion == "sphere":
        shape = sp.ImmutableMatrix(sp.eye(len(prescription.point)))
    else:
        shape = _ellipsoid_shape(prescription)
        if shape is None:
            return None
    ranges = {sp.sympify(var): (sp.sympify(lo), sp.sympify(hi)) for var, lo, hi in region.ranges}
    for scale in (sp.Integer(1), sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 8)):
        contained = True
        for index, (var, center) in enumerate(
            zip(region.variables, prescription.point, strict=True)
        ):
            lower, upper = ranges[var]
            reach = sp.simplify(scale * sp.sqrt(shape[index, index]))
            left = sp.simplify(center - reach - lower)
            right = sp.simplify(upper - center - reach)
            if left.is_nonnegative is not True or right.is_nonnegative is not True:
                contained = False
                break
        if contained:
            return scale
    return None


def _certified_local_scale(formula: sp.Expr, variables, radius_sq: sp.Expr) -> sp.Rational | None:
    """Find a simple positive scale whose closed exclusion ellipsoid lies in the domain."""
    for scale in (sp.Integer(1), sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 8)):
        local_region = sp.Le(radius_sq, scale**2)
        try:
            contained = semialg_implies(local_region, formula, tuple(variables))
        except (ArithmeticError, TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            contained = False
        if contained is True:
            return scale
    return None


def _local_structured_region(
    variables: tuple[sp.Symbol, ...],
    prescription: GeometricPrincipalValue,
    scale: sp.Expr,
) -> Region:
    if prescription.exclusion == "sphere":
        centered = tuple(
            var - point for var, point in zip(variables, prescription.point, strict=True)
        )
        outer_first = []
        sumsq = sp.Integer(0)
        for index, (var, point) in enumerate(zip(variables, prescription.point, strict=True)):
            radial_bound = scale if index == 0 else sp.sqrt(scale**2 - sumsq)
            outer_first.append((var, point - radial_bound, point + radial_bound))
            sumsq += centered[index] ** 2
        ranges = tuple(outer_first)
        if len(variables) == 2:
            return DiskRegion(ranges, center=prescription.point, radius=scale)
        return BallRegion(ranges, center=prescription.point, radius=scale, dimension=len(variables))
    if prescription.axes is None:
        raise ValueError(
            "rotated ellipsoidal neighborhoods do not have an axis-aligned Region form"
        )
    return EllipsoidRegion(
        (),
        variables_nd=variables,
        center=prescription.point,
        axes=tuple(sp.simplify(scale * axis) for axis in prescription.axes),
    )


def _is_regular_at_point(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    point: tuple[sp.Expr, ...],
) -> bool:
    value = sp.sympify(expr).subs(
        dict(zip(variables, point, strict=True)),
        simultaneous=True,
    )
    return value.is_finite is True


def _box_central_inversion_zero(
    expr: sp.Expr,
    region: BoxRegion,
    prescription: GeometricPrincipalValue,
) -> bool:
    """Recognize odd cancellation under simultaneous inversion about the center."""
    ranges = {sp.sympify(var): (sp.sympify(lo), sp.sympify(hi)) for var, lo, hi in region.ranges}
    reflection = {}
    for variable, center in zip(region.variables, prescription.point, strict=True):
        lower, upper = ranges[variable]
        if sp.simplify(lower + upper - 2 * center) != 0:
            return False
        reflection[variable] = 2 * center - variable
    reflected = sp.sympify(expr).subs(reflection, simultaneous=True)
    return sp.simplify(reflected + expr) == 0


def _box_reflection_zero(
    expr: sp.Expr,
    region: BoxRegion,
    prescription: GeometricPrincipalValue,
    radius_sq: sp.Expr,
) -> bool:
    """Recognize exact odd cancellation in a centered Cartesian box."""
    ranges = {sp.sympify(var): (sp.sympify(lo), sp.sympify(hi)) for var, lo, hi in region.ranges}
    for variable, center in zip(region.variables, prescription.point, strict=True):
        lower, upper = ranges[variable]
        if sp.simplify(lower + upper - 2 * center) != 0:
            continue
        reflection = {variable: 2 * center - variable}
        reflected_radius = sp.sympify(radius_sq).subs(reflection)
        if sp.simplify(reflected_radius - radius_sq) != 0:
            continue
        reflected = sp.sympify(expr).subs(reflection)
        if sp.simplify(reflected + expr) == 0:
            return True
    return False


def _general_geometric_pv(
    expr: sp.Expr,
    region: object,
    prescription: GeometricPrincipalValue,
    *,
    variables=None,
    bounds=None,
    assumptions=None,
):
    """Evaluate geometric PV on an arbitrary certified semialgebraic domain.

    Rather than asking CAD to carry a symbolic epsilon, certify a fixed local
    ball/ellipsoid around the singular point, evaluate its PV with the direct
    shared-cutoff transform, and integrate the nonsingular remainder over the
    original domain with that fixed neighborhood removed.  This is equivalent
    to the original epsilon definition whenever the local neighborhood is
    contained in the domain.
    """
    if isinstance(region, BoxRegion):
        formula = _box_formula(region)
        vars_ = tuple(region.variables)
    else:
        formula, vars_ = semialgebraic_domain(region, variables)
    if len(vars_) != len(prescription.point):
        raise ValueError("geometric PV point dimension must match the integration domain")

    from .core import _integrate_domain

    radius_sq = _exclusion_radius_sq(vars_, prescription)
    if _is_regular_at_point(expr, vars_, prescription.point):
        if isinstance(region, BoxRegion):
            from .core import integrate

            return integrate(
                expr,
                *region.ranges,
                assumptions=assumptions,
                principal_value=False,
            )
        return _integrate_domain(
            expr,
            formula,
            variables=vars_,
            bounds=bounds,
            assumptions=assumptions,
            principal_value=False,
        )
    if isinstance(region, BoxRegion) and (
        _box_central_inversion_zero(expr, region, prescription)
        or _box_reflection_zero(expr, region, prescription, radius_sq)
    ):
        record_strategy(
            "geometric_principal_value",
            reason=f"exact reflection cancellation in dimension {len(vars_)}",
        )
        return sp.Integer(0)

    scale = (
        _box_local_scale(region, prescription)
        if isinstance(region, BoxRegion)
        else _certified_local_scale(formula, vars_, radius_sq)
    )
    if scale is None:
        raise NotImplementedError(
            "geometric PV requires a certified symmetric neighborhood of the singular point "
            "inside the general domain; semialg could not certify one"
        )

    # A proved globally radial profile has no singularity away from its center.
    # If its local absolute moments converge, every shrinking positive-definite
    # exclusion has the ordinary limit. Avoid adding a redundant curved CAD hole.
    from .planner import record_coverage
    from .quadratic_radial import radial_region_data

    probe = _local_structured_region(vars_, GeometricPrincipalValue(prescription.point), sp.S.One)
    if radial_region_data(expr, probe, assumptions=assumptions) is not None:
        record_coverage(
            "local_cutoff_shape_independence",
            True,
            "geometric_principal_value",
            basis="proved radial local absolute moments and continuity away from the center",
        )
        return _integrate_domain(
            expr,
            formula,
            variables=vars_,
            bounds=bounds,
            assumptions=assumptions,
            principal_value=False,
        )

    if prescription.exclusion == "ellipsoid" and prescription.shape_matrix is not None:
        affine = sp.simplify(scale) * _shape_factor(prescription.shape_matrix)
        local_value = _local_radial_pv(
            expr,
            tuple(vars_),
            prescription.point,
            affine,
            sp.Integer(1),
            assumptions=assumptions,
        )
    else:
        local_region = _local_structured_region(vars_, prescription, scale)
        local_prescription = prescription
        if prescription.exclusion == "ellipsoid":
            local_prescription = GeometricPrincipalValue(
                prescription.point,
                exclusion="ellipsoid",
                axes=tuple(local_region.axes),
            )
        local_value = geometric_pv(
            expr,
            local_region,
            local_prescription,
            assumptions=assumptions,
        )

    exterior = sp.Ge(radius_sq, scale**2)
    remainder_formula = sp.And(formula, exterior)
    remainder = _integrate_domain(
        expr,
        remainder_formula,
        variables=vars_,
        bounds=bounds,
        assumptions=assumptions,
        principal_value=False,
    )
    record_strategy(
        "geometric_principal_value",
        reason=(
            f"certified local {prescription.exclusion} scale {scale} at {prescription.point}; "
            "semialgebraic remainder"
        ),
    )
    return _fast_simplify(local_value + remainder)


def geometric_pv(
    expr,
    region,
    prescription: GeometricPrincipalValue,
    *,
    variables=None,
    bounds=None,
    assumptions=None,
):
    """Evaluate a shared-cutoff geometric PV.

    Structured disks, balls, and homothetic ellipsoids use n-dimensional
    radial coordinates, preferring exact unit-sphere moments before explicit
    hyperspherical realization.  Cartesian boxes and general semialgebraic
    domains use a certified local neighborhood plus a semialgebraic remainder.
    """
    if isinstance(region, Region):
        data = _structured_radial_data(region, prescription)
        if data is not None:
            variables_, center, affine_matrix, outer_radius = data
            return _local_radial_pv(
                sp.sympify(expr),
                variables_,
                center,
                affine_matrix,
                outer_radius,
                assumptions=assumptions,
            )
        if not isinstance(region, BoxRegion):
            raise NotImplementedError(
                "geometric PV direct transforms support centered DiskRegion/BallRegion and "
                "homothetic EllipsoidRegion in arbitrary dimension; use a BoxRegion or "
                "semialgebraic region for general punctured-domain reduction"
            )
    return _general_geometric_pv(
        sp.sympify(expr),
        region,
        prescription,
        variables=variables,
        bounds=bounds,
        assumptions=assumptions,
    )


__all__ = ["geometric_pv"]
