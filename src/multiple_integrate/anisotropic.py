"""Weighted even-power geometry: degenerate point singularities and exact moments."""

from __future__ import annotations

import sympy as sp

from .assumptions import _ask, _normalize_assumptions
from .planner import record_coverage, record_strategy
from .policy import _normalize_policy


def _true(predicate, assumptions):
    return _ask(predicate, _normalize_assumptions(assumptions)) is True


def even_power_profile(expr, variables, assumptions=None, policy=None):
    """Recognize P/(b+sum(c_i*x_i**m_i))**s with positive even powers."""
    variables = tuple(variables)
    budget = _normalize_policy(policy)
    if not variables or (
        not budget.allow_expensive_exact and len(variables) > budget.semialg.max_variables
    ):
        return None
    for factor in sp.Mul.make_args(sp.sympify(expr)):
        if (
            not factor.is_Pow
            or factor.exp.has(*variables)
            or not _true(sp.Q.real(factor.exp) & sp.Q.finite(factor.exp), assumptions)
        ):
            continue
        try:
            base = sp.Poly(factor.base, *variables)
            numerator = sp.Poly(sp.cancel(expr / factor), *variables)
        except sp.PolynomialError:
            continue
        powers = [None] * len(variables)
        coefficients = [None] * len(variables)
        shift = sp.S.Zero
        valid = True
        for monomial, c in base.terms():
            active = [i for i, p in enumerate(monomial) if p]
            if not active:
                shift = c
            elif len(active) == 1:
                i = active[0]
                m = monomial[i]
                if (
                    powers[i] is not None
                    or m % 2
                    or not _true(sp.Q.positive(c) & sp.Q.finite(c), assumptions)
                ):
                    valid = False
                    break
                powers[i] = m
                coefficients[i] = c
            else:
                valid = False
                break
        if not valid or any(p is None for p in powers):
            continue
        if not _true(sp.Q.nonnegative(shift) & sp.Q.finite(shift), assumptions):
            continue
        if _true(sp.Q.zero(shift), assumptions):
            shift = sp.S.Zero
        if not all(_true(sp.Q.finite(c), assumptions) for c in numerator.coeffs()):
            continue
        if not budget.allow_expensive_exact and (
            base.total_degree() > budget.semialg.max_polynomial_degree
            or numerator.total_degree() > budget.semialg.max_polynomial_degree
            or len(numerator.terms()) > budget.semialg.max_pieces
        ):
            continue
        return numerator, shift, -factor.exp, tuple(powers), tuple(coefficients)
    return None


def anisotropic_absolute(expr, ranges, assumptions=None, policy=None):
    coords = tuple(r[0] for r in ranges)
    data = even_power_profile(expr, coords, assumptions, policy)
    if data is None:
        return None
    poly, b, s, powers, _ = data
    weights = [
        sp.Add(*(sp.Rational(p + 1, m) for p, m in zip(monomial, powers, strict=True)))
        for monomial, c in poly.terms()
        if c != 0
    ]
    full = all(lo == -sp.oo and hi == sp.oo for _, lo, hi in ranges)
    local = all(
        _true(
            sp.Q.finite(lo) & sp.Q.finite(hi) & sp.Q.negative(lo) & sp.Q.positive(hi), assumptions
        )
        for _, lo, hi in ranges
    )
    if (
        full
        and _true(sp.Q.positive(b), assumptions)
        and all(_true(sp.Q.positive(s - w), assumptions) for w in weights)
    ):
        return True
    if local and all(_true(sp.Q.positive(w - s), assumptions) for w in weights):
        return True
    if poly.total_degree() == 0 and _true(sp.Q.nonzero(poly.as_expr()), assumptions):
        w = weights[0] if weights else None
        if w is not None and (
            (local and b == 0 and _true(sp.Q.nonnegative(s - w), assumptions))
            or (full and ((b == 0) or _true(sp.Q.nonpositive(s - w), assumptions)))
        ):
            return False
    return None


def try_anisotropic_moments(expr, ranges, assumptions=None):
    if not ranges or not all(lo == -sp.oo and hi == sp.oo for _, lo, hi in ranges):
        return None
    data = even_power_profile(expr, tuple(r[0] for r in ranges), assumptions)
    if data is None or anisotropic_absolute(expr, ranges, assumptions) is not True:
        return None
    poly, b, s, powers, coefficients = data
    total = sp.S.Zero
    for monomial, c in poly.terms():
        if any(p % 2 for p in monomial):
            continue
        alpha = tuple(sp.Rational(p + 1, m) for p, m in zip(monomial, powers, strict=True))
        total += (
            c
            * sp.gamma(s - sp.Add(*alpha))
            * b ** (sp.Add(*alpha) - s)
            / sp.gamma(s)
            * sp.prod(
                2 * sp.gamma(a) / (m * d**a)
                for a, m, d in zip(alpha, powers, coefficients, strict=True)
            )
        )
    record_strategy("anisotropic_moments")
    record_coverage(
        "absolute_convergence",
        True,
        "anisotropic_moments",
        basis="all weighted even-power moments satisfy Dirichlet tail thresholds",
    )
    return sp.simplify(total)
