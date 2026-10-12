"""Hadamard finite-part evaluation from explicit cutoff asymptotics."""

from __future__ import annotations

import sympy as sp

from .assumptions import _refine_with_assumptions
from .cauchy_principal_value import _inside_interval, _rational_poles
from .hypersurface_principal_value import hypersurface_cutoff_integral
from .planner import record_strategy
from .principal_value_prescriptions import HypersurfacePrincipalValue
from .rational_cutoff import certified_order, real_rational_cutoff, requires_real_rational_cutoff
from .recognition import _fast_simplify
from .regularization import HadamardFinitePart, _record_regularization, finite_part_from_cutoff
from .results import UnresolvedIntegration


def _one_dimensional_cutoff(
    expr: sp.Expr,
    range_: tuple[sp.Symbol, sp.Expr, sp.Expr],
    prescription: HadamardFinitePart,
    *,
    assumptions=None,
) -> tuple[sp.Expr, sp.Symbol, sp.Expr]:
    variable, lower, upper = range_
    infinite = lower == -sp.oo or upper == sp.oo
    if lower == sp.oo or upper == -sp.oo:
        raise NotImplementedError("finite parts require increasing integration endpoints")
    if infinite and not requires_real_rational_cutoff(expr, variable, assumptions=assumptions):
        raise NotImplementedError(
            "finite parts at infinity currently require real rational densities"
        )
    points = prescription.singularities
    if not points:
        candidates = _rational_poles(sp.sympify(expr), variable)
        points = tuple(
            point
            for point in candidates
            if sp.simplify(point - lower) == 0
            or sp.simplify(point - upper) == 0
            or _inside_interval(point, lower, upper, assumptions)
        )
    if not points and not infinite:
        raise ValueError(
            "HadamardFinitePart found no finite singularities in the integration interval"
        )

    epsilon = sp.Dummy("regularization_epsilon", positive=True, real=True)
    ordered = certified_order(points, assumptions=assumptions)
    for point in ordered:
        if (
            sp.simplify(point - lower) != 0
            and sp.simplify(point - upper) != 0
            and not _inside_interval(point, lower, upper, assumptions)
        ):
            raise ValueError(
                "explicit finite-part singularity is not certified inside the interval"
            )
    cutoff_lower = -1 / epsilon if lower == -sp.oo else lower
    cutoff_upper = 1 / epsilon if upper == sp.oo else upper
    # Finite endpoint poles still use the same epsilon as the large-radius cutoff.
    canonical = real_rational_cutoff(
        expr,
        variable,
        cutoff_lower,
        cutoff_upper,
        ordered,
        epsilon,
        assumptions=assumptions,
        require_all_poles=True,
    )
    if canonical is not None:
        locus = sp.Or(*(sp.Eq(variable, point) for point in ordered))
        return _refine_with_assumptions(canonical, assumptions), epsilon, locus
    if requires_real_rational_cutoff(expr, variable, assumptions=assumptions):
        raise UnresolvedIntegration(
            "a canonical real rational finite-part primitive could not be certified"
        )
    total = sp.Integer(0)
    start = lower
    for point in ordered:
        at_lower = sp.simplify(point - lower) == 0
        at_upper = sp.simplify(point - upper) == 0
        if at_lower:
            start = lower + epsilon
            continue
        stop = upper - epsilon if at_upper else point - epsilon
        total += sp.integrate(expr, (variable, start, stop))
        start = point + epsilon
    if sp.simplify(ordered[-1] - upper) != 0:
        total += sp.integrate(expr, (variable, start, upper))
    locus = sp.Or(*(sp.Eq(variable, point) for point in ordered))
    return _refine_with_assumptions(sp.simplify(total), assumptions), epsilon, locus


def hadamard_finite_part(
    expr: sp.Expr,
    *ranges,
    region=None,
    variables=None,
    bounds=None,
    prescription: HadamardFinitePart,
    assumptions=None,
) -> sp.Expr:
    """Evaluate a Hadamard finite part from the constant cutoff-asymptotic term."""
    expr = sp.sympify(expr)
    if prescription.hypersurface is not None:
        if region is None:
            if not ranges:
                raise ValueError("hypersurface finite part requires a region or explicit ranges")
            from .regions import region_from_ranges

            region = region_from_ranges(ranges, structural_order="inner-first")
        from .regular_density import regular_density_cutoff

        regular = regular_density_cutoff(
            expr,
            region,
            prescription.hypersurface,
            assumptions=assumptions,
            prescription="HadamardFinitePart",
        )
        if regular is not None:
            return regular
        hypersurface = HypersurfacePrincipalValue(prescription.hypersurface)
        cutoff, epsilon, cutoff_domain, _, details = hypersurface_cutoff_integral(
            expr,
            region,
            hypersurface,
            variables=variables,
            bounds=bounds,
            assumptions=assumptions,
            return_details=True,
        )
        locus = sp.Eq(prescription.hypersurface, 0)
        value, provenance = finite_part_from_cutoff(
            cutoff,
            epsilon,
            singular_locus=locus,
            cutoff_domain=cutoff_domain,
            backend=str(details.get("backend")),
            details=details,
        )
    else:
        if region is not None or len(ranges) != 1 or len(ranges[0]) != 3:
            raise NotImplementedError(
                "HadamardFinitePart without hypersurface= currently requires one explicit 1-D range"
            )
        range_ = tuple(map(sp.sympify, ranges[0]))
        cutoff, epsilon, locus = _one_dimensional_cutoff(
            expr, range_, prescription, assumptions=assumptions
        )
        value, provenance = finite_part_from_cutoff(
            cutoff,
            epsilon,
            singular_locus=locus,
            details={
                "infinity_cutoff": "R=1/epsilon; symmetric about zero for two infinite endpoints",
                "finite_poles_share_epsilon": True,
            }
            if any(b in (-sp.oo, sp.oo) for b in range_[1:])
            else None,
        )
    _record_regularization(provenance)
    record_strategy("hadamard_finite_part", reason="constant term of shared-cutoff asymptotics")
    return _fast_simplify(value)


__all__ = ["hadamard_finite_part"]
