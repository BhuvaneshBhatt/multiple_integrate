"""Certified real ordering and canonical cutoffs for split real rational poles."""

from __future__ import annotations

from functools import cmp_to_key

import sympy as sp

from .assumptions import _ask, _normalize_assumptions
from .results import UnresolvedIntegration


def certified_order(points, *, assumptions=None):
    context = _normalize_assumptions(assumptions)

    def compare(a, b):
        difference = sp.simplify(a - b)
        if difference == 0:
            return 0
        if difference.is_negative is True or _ask(sp.Q.negative(difference), context) is True:
            return -1
        if difference.is_positive is True or _ask(sp.Q.positive(difference), context) is True:
            return 1
        raise UnresolvedIntegration("real singular-point order could not be certified")

    return tuple(sorted(set(map(sp.sympify, points)), key=cmp_to_key(compare)))


def real_rational_cutoff(
    expr, variable, lower, upper, points, epsilon, *, assumptions=None, require_all_poles=False
):
    """Return an exact real cutoff when the real poles and smooth real remainder are resolved.

    None declines other families. Unknown real ordering/signs are refused, never
    guessed. No unnormalized complex logarithms enter this calculation.
    """
    if lower in (-sp.oo, sp.oo) or upper in (-sp.oo, sp.oo):
        return None
    context = _normalize_assumptions(assumptions)
    numerator, denominator = sp.cancel(expr).as_numer_denom()
    try:
        polynomial = sp.Poly(denominator, variable)
        roots = sp.roots(polynomial.as_expr(), variable)
        if sum(roots.values()) != polynomial.degree():
            return None
        coefficients = sp.Poly(numerator, variable).all_coeffs() + polynomial.all_coeffs()
        if not all(c.is_real is True or _ask(sp.Q.real(c), context) is True for c in coefficients):
            return None
        real_roots = {}
        for root, multiplicity in roots.items():
            if root.is_real is True or _ask(sp.Q.real(root), context) is True:
                real_roots[root] = multiplicity
            elif root.is_real is not False:
                return None
        quotient, _ = sp.div(numerator, denominator, variable)
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError):
        return None
    ordered = certified_order(points, assumptions=context)
    if require_all_poles:
        for root in real_roots:
            if root in ordered:
                continue
            # Infinite endpoints are represented by +/-1/epsilon here.
            left = sp.limit(root - lower, epsilon, 0, dir="+")
            right = sp.limit(upper - root, epsilon, 0, dir="+")
            outside = (
                left.is_negative is True
                or right.is_negative is True
                or _ask(sp.Q.negative(left), context) is True
                or _ask(sp.Q.negative(right), context) is True
            )
            if not outside:
                raise UnresolvedIntegration(
                    "finite-part cutoff must exclude every real pole in the interval"
                )
    terms = []
    rebuilt = quotient
    for root, multiplicity in real_roots.items():
        local = sp.cancel(expr * (variable - root) ** multiplicity, extension=True)
        for k in range(1, multiplicity + 1):
            derivative_order = multiplicity - k
            coefficient = sp.simplify(
                sp.diff(local, variable, derivative_order).subs(variable, root)
                / sp.factorial(derivative_order)
            )
            if coefficient.has(sp.nan, sp.oo, sp.zoo):
                return None
            if coefficient != 0:
                terms.append((root, k, coefficient))
                rebuilt += coefficient / (variable - root) ** k
    residual = sp.cancel(expr - rebuilt, extension=True) + quotient
    polynomial_primitive = sp.integrate(residual, variable)
    if polynomial_primitive.has(sp.Integral) or polynomial_primitive.is_real is not True:
        return None
    if sp.cancel(sp.diff(polynomial_primitive, variable) - residual, extension=True) != 0:
        return None

    def primitive_at(endpoint):
        result = sp.simplify(polynomial_primitive.subs(variable, endpoint))
        for root, k, coefficient in terms:
            distance = sp.simplify(endpoint - root)
            if k == 1:
                base = sp.limit(distance, epsilon, 0, dir="+")
                if base == 0:
                    real_distance = sp.Abs(distance)
                elif (
                    base == sp.oo
                    or base.is_positive is True
                    or _ask(sp.Q.positive(base), context) is True
                ):
                    real_distance = distance
                elif (
                    base == -sp.oo
                    or base.is_negative is True
                    or _ask(sp.Q.negative(base), context) is True
                ):
                    real_distance = -distance
                else:
                    raise UnresolvedIntegration(
                        "real logarithm branch at cutoff endpoint is uncertified"
                    )
                result += coefficient * sp.log(real_distance)
            else:
                result += coefficient * distance ** (1 - k) / (1 - k)
        return result

    total = sp.Integer(0)
    start = lower
    for point in ordered:
        if sp.simplify(point - lower) == 0:
            start = lower + epsilon
            continue
        stop = upper - epsilon if sp.simplify(point - upper) == 0 else point - epsilon
        total += primitive_at(stop) - primitive_at(start)
        start = point + epsilon
    if not ordered or sp.simplify(ordered[-1] - upper) != 0:
        total += primitive_at(upper) - primitive_at(start)
    return sp.expand(total)


def requires_real_rational_cutoff(expr, variable, *, assumptions=None):
    """Whether falling back to an unnormalized rational primitive is unsafe."""
    context = _normalize_assumptions(assumptions)
    try:
        numerator, denominator = sp.cancel(expr).as_numer_denom()
        coefficients = (
            sp.Poly(numerator, variable).all_coeffs() + sp.Poly(denominator, variable).all_coeffs()
        )
        return all(c.is_real is True or _ask(sp.Q.real(c), context) is True for c in coefficients)
    except (sp.PolynomialError, TypeError, ValueError):
        return False
