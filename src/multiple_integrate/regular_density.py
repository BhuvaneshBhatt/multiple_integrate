"""Dominated-convergence shortcut for regular densities and polynomial cutoffs."""

from __future__ import annotations

import sympy as sp

from .assumptions import _ask, _normalize_assumptions
from .convergence_evidence import compact_continuity
from .planner import record_coverage, record_strategy
from .region_types import BoxRegion
from .regularization import RegularizationProvenance, _record_regularization


def regular_density_prerequisites(expr, ranges, polynomial, *, assumptions=None, policy=None):
    """A compact box, continuous density, and a nonzero real polynomial suffice.

    Polynomial zero sets have ambient Lebesgue measure zero. The density is
    bounded on this compact box, so dominated convergence applies to |g|>eps.
    """
    context = _normalize_assumptions(assumptions)
    coords = tuple(r[0] for r in ranges)
    if not coords or len(set(coords)) != len(coords):
        return False
    if not all(
        _ask(sp.Q.positive(sp.sympify(hi) - sp.sympify(lo)), context) is True
        for _, lo, hi in ranges
    ):
        return False
    if not compact_continuity(expr, ranges, assumptions=context, policy=policy):
        return False
    try:
        poly = sp.Poly(polynomial, *coords)
    except sp.PolynomialError:
        return False
    coefficients = poly.coeffs()
    return all(_ask(sp.Q.real(c) & sp.Q.finite(c), context) is True for c in coefficients) and any(
        _ask(sp.Q.nonzero(c), context) is True for c in coefficients
    )


def regular_density_cutoff(
    expr, region, polynomial, *, assumptions=None, prescription, policy=None
):
    """Return a proved cutoff limit; never fabricate a truncated integral."""
    if not isinstance(region, BoxRegion) or not regular_density_prerequisites(
        expr, region.ranges, polynomial, assumptions=assumptions, policy=policy
    ):
        return None
    from .core import integrate

    value = integrate(expr, *region.ranges, assumptions=assumptions, policy=policy)
    if value.is_finite is not True or value.has(sp.Integral, sp.Limit, sp.Piecewise):
        return None
    epsilon = sp.Dummy("regularization_epsilon", positive=True)
    domain = sp.And(*(atom for v, lo, hi in region.ranges for atom in (v >= lo, v <= hi)))
    _record_regularization(
        RegularizationProvenance(
            method="regular_density_cutoff",
            prescription=prescription,
            cutoff_parameter=epsilon,
            regulator=sp.Abs(polynomial),
            singular_locus=sp.Eq(polynomial, 0),
            cutoff_domain=sp.And(domain, polynomial**2 > epsilon**2),
            constant_term=value,
            backend="dominated_convergence",
            details={
                "basis": "compact continuous density; nonzero real polynomial has null zero set",
                "density": sp.sympify(expr),
                "ranges": region.ranges,
                "polynomial": sp.sympify(polynomial),
                "assumptions": _normalize_assumptions(assumptions).original,
            },
        )
    )
    record_coverage(
        "integration_domain",
        True,
        "regular_density_cutoff",
        basis="dominated convergence across an ambient null polynomial zero set",
    )
    record_strategy(
        "regular_density_cutoff", reason="proved compact continuity and null polynomial zero set"
    )
    return value
