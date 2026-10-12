"""Budgeted exact Taylor moments with a rigorous finite-box remainder bound."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .fast_rules import _polynomial, box_polynomial_moment
from .policy import _normalize_policy, _with_policy

_DEFAULT_TOLERANCE = sp.Rational(1, 10**20)


@dataclass(frozen=True)
class SeriesApproximation:
    """An exact partial sum and its analytic error bound, before float rounding."""

    value: sp.Expr
    error_bound: sp.Rational
    terms: int
    converged: bool
    certified: bool = True
    limitations: tuple[str, ...] = (
        "floating evaluation of the exact partial sum has additional rounding error",
    )


@_with_policy
def evaluate_entire_box(
    f,
    *ranges,
    substitutions=None,
    tolerance=_DEFAULT_TOLERANCE,
    max_terms=128,
    max_monomials=4096,
    max_majorant_exponent=1024,
    policy=None,
):
    """Approximate P*exp/sin/cos(Q) on an exact finite Cartesian box.

    The returned value is an exact finite expression. Its integral error is
    bounded by a rational majorant. A budget exhaustion retains a partial sum
    and its bound, with converged=False; it never claims the requested accuracy.
    """
    if any(type(n) is not int or n < 1 for n in (max_terms, max_monomials, max_majorant_exponent)):
        raise ValueError("series budgets must be positive integers")
    tolerance = sp.sympify(tolerance)
    if tolerance.is_Rational is not True or tolerance.is_positive is not True:
        raise ValueError("tolerance must be an exact positive rational")
    if not ranges or any(len(r) != 3 or not isinstance(r[0], sp.Symbol) for r in ranges):
        raise ValueError("independent coordinate ranges are required")
    variables = tuple(r[0] for r in ranges)
    if len(set(variables)) != len(variables):
        raise ValueError("coordinate symbols must be distinct")
    mapping = (
        {}
        if substitutions is None
        else {sp.sympify(k): sp.sympify(v) for k, v in substitutions.items()}
    )
    if set(variables) & mapping.keys():
        raise ValueError("substitutions must not replace integration coordinates")
    expr = sp.sympify(f).subs(mapping, simultaneous=True)
    limits = tuple(
        (
            v,
            sp.sympify(lo).subs(mapping, simultaneous=True),
            sp.sympify(hi).subs(mapping, simultaneous=True),
        )
        for v, lo, hi in ranges
    )
    budget = _normalize_policy(policy)
    if (
        not budget.allow_expensive_exact
        and len(variables) > budget.semialg.max_variables
        or any(
            b.free_symbols or b.is_real is not True or b.is_finite is not True
            for r in limits
            for b in r[1:]
        )
        or expr.free_symbols - set(variables)
    ):
        raise NotImplementedError("resolve parameters on a finite real independent box")
    if any(b.has(sp.Float) for r in limits for b in r[1:]) or expr.has(sp.Float):
        raise NotImplementedError("certified bounds require exact input coefficients and endpoints")
    if any((hi - lo).is_positive is not True for _, lo, hi in limits):
        raise ValueError("series evaluation requires strictly ordered endpoints")
    functions = [a for a in sp.Mul.make_args(expr) if a.func in (sp.exp, sp.sin, sp.cos)]
    if not functions:
        polynomial_value = box_polynomial_moment(expr, limits)
        if polynomial_value is not None:
            return SeriesApproximation(polynomial_value, sp.S.Zero, 1, True)
    if len(functions) != 1:
        raise NotImplementedError("one exponential, sine or cosine factor is required")
    function = functions[0]
    amplitude = _polynomial(expr / function, variables)
    phase = _polynomial(function.args[0], variables)
    if (
        amplitude is None
        or phase is None
        or any(c.is_finite is not True for c in (*amplitude.coeffs(), *phase.coeffs()))
    ):
        raise NotImplementedError(
            "bounded polynomial amplitude and phase with finite coefficients are required"
        )
    zero = (0,) * len(variables)
    amplitude_terms = amplitude.terms()
    constant = phase.coeff_monomial(zero)
    kernel = {m: c for m, c in phase.terms() if any(m)}
    radii = tuple(sp.Max(sp.Abs(lo), sp.Abs(hi)) for _, lo, hi in limits)

    def size(terms):
        return sp.Add(
            *(sp.Abs(c) * sp.prod(r**p for r, p in zip(radii, m, strict=True)) for m, c in terms)
        )

    def ceiling(value):
        upper = sp.ceiling(value)
        if upper.is_Integer is not True or upper < 0:
            raise NotImplementedError("an exact magnitude bound was not established")
        return upper

    magnitude = ceiling(size(kernel.items()))
    volume = sp.prod(ceiling(hi - lo) for _, lo, hi in limits)
    exponent = ceiling(sp.Abs(constant)) + magnitude
    if exponent > max_majorant_exponent:
        raise NotImplementedError("exponential majorant exceeds its magnitude budget")
    prefactor = volume * ceiling(size(amplitude_terms)) * 3**exponent
    # e<3 supplies a rational derivative majorant, avoiding numerical comparisons.
    current = {zero: sp.S.One}
    value, bound, count = sp.S.Zero, prefactor, 0
    for k in range(max_terms):
        if len(current) * len(amplitude_terms) > max_monomials:
            break
        moment = sp.S.Zero
        for m, c in current.items():
            for p, weight in amplitude_terms:
                degrees = tuple(i + j for i, j in zip(m, p, strict=True))
                moment += (
                    c
                    * weight
                    * sp.prod(
                        (hi ** (d + 1) - lo ** (d + 1)) / sp.Integer(d + 1)
                        for (_, lo, hi), d in zip(limits, degrees, strict=True)
                    )
                )
        derivative = (
            sp.exp(constant) if function.func is sp.exp else function.func(constant + sp.pi * k / 2)
        )
        value += derivative * moment / sp.factorial(k)
        count = k + 1
        bound = prefactor * magnitude**count / sp.factorial(count)
        if bound <= tolerance:
            return SeriesApproximation(value, bound, count, True)
        if len(current) * len(kernel) > max_monomials:
            break
        following = {}
        for m, c in current.items():
            for p, coefficient in kernel.items():
                powers = tuple(i + j for i, j in zip(m, p, strict=True))
                following[powers] = following.get(powers, sp.S.Zero) + c * coefficient
        current = {m: c for m, c in following.items() if c != 0}
    return SeriesApproximation(value, bound, count, False)


__all__ = ["SeriesApproximation", "evaluate_entire_box"]
