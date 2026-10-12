"""Exact analytic strategies used by the integration planner."""

from __future__ import annotations

import sympy as sp

from multiple_integrate.adapters import region_moment as semialgebraic_domain_moment
from multiple_integrate.planner import record_strategy, without_coverage
from multiple_integrate.recognition import (
    _fast_simplify,
)
from multiple_integrate.region_types import (
    AffineSimplexRegion,
    AnnulusRegion,
    BallRegion,
    DiskRegion,
    EllipsoidRegion,
    GraphRegion,
    Region,
    SimplexRegion,
    SphericalShellRegion,
    UnionRegion,
)
from multiple_integrate.strategy_common import _symbolically_positive

_STRATEGY_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


def _simplex_dirichlet_term(term: sp.Expr, vars_: tuple[sp.Symbol, ...]):
    """Parse coeff * prod(x_i**a_i) * (1-sum x_i)**b for a simplex term."""
    term = sp.factor_terms(sp.sympify(term))
    rem = sp.expand(1 - sum(vars_))
    coeff = sp.Integer(1)
    exponents = {v: sp.Integer(0) for v in vars_}
    rem_exp = sp.Integer(0)
    factors = term.args if term.is_Mul else (term,)
    for factor in factors:
        factor = sp.sympify(factor)
        if not (factor.free_symbols & set(vars_)):
            coeff *= factor
            continue
        base, exp = factor.as_base_exp()
        base = sp.expand(base)
        if base in exponents:
            exponents[base] += exp
            continue
        if sp.simplify(base - rem) == 0:
            rem_exp += exp
            continue
        return None
    return coeff, tuple(exponents[v] for v in vars_), rem_exp


def _simplex_dirichlet(region: Region, expr: sp.Expr, assumptions=None) -> sp.Expr | None:
    """Exact Dirichlet-type integration on standard and affine simplices."""
    if isinstance(region, AffineSimplexRegion):
        simplex, transformed = region._standard_simplex_pullback(expr)
        return _simplex_dirichlet(simplex, transformed, assumptions=assumptions)

    if not isinstance(region, SimplexRegion):
        return None

    vars_ = tuple(region.variables)
    expr = sp.sympify(expr)
    total = sp.Integer(0)
    terms = expr.args if expr.is_Add else (expr,)
    for term in terms:
        parsed = _simplex_dirichlet_term(term, vars_)
        if parsed is None:
            return None
        coeff, exponents, rem_exp = parsed
        alphas = [sp.simplify(e + 1) for e in exponents] + [sp.simplify(rem_exp + 1)]
        if any(a.is_nonpositive is True for a in alphas):
            return None
        if any(
            (a.is_positive is not True) and not _symbolically_positive(a, assumptions=assumptions)
            for a in alphas
        ):
            return None
        numer = coeff
        for a in alphas:
            numer *= sp.gamma(a)
        total += numer / sp.gamma(sp.Add(*alphas))
    return _fast_simplify(total)


def _region_shortcut(region: Region, expr: sp.Expr, assumptions=None) -> sp.Expr | None:
    """Exact moments and simple radial integrals on recognized regions."""
    from .fast_rules import _certificate, box_polynomial_moment
    from .region_types import BoxRegion

    if isinstance(region, BoxRegion):
        from .assumptions import _ask, _normalize_assumptions

        context = _normalize_assumptions(assumptions)
        normalized = []
        for v, lo, hi in region.ranges:
            if _ask(sp.Q.nonnegative(hi - lo), context) is True:
                normalized.append((v, lo, hi))
            elif _ask(sp.Q.nonnegative(lo - hi), context) is True:
                normalized.append((v, hi, lo))
            else:
                break
        moment = (
            box_polynomial_moment(expr, normalized, assumptions)
            if len(normalized) == len(region.ranges)
            else None
        )
        if moment is not None:
            _certificate("box_polynomial_moments")
            return moment
    if isinstance(region, UnionRegion):
        total = sp.Integer(0)
        with without_coverage():
            for piece in region.pieces:
                val = _region_shortcut(piece, expr, assumptions=assumptions)
                if val is None:
                    return None
                total += val
        return sp.simplify(total)

    from multiple_integrate.quadratic_radial import try_weighted_radial_region

    weighted = try_weighted_radial_region(expr, region, assumptions=assumptions)
    if weighted is not None:
        return weighted
    if isinstance(region, (SimplexRegion, AffineSimplexRegion)):
        poly_res = region.polynomial_moment(expr)
        if poly_res is not None:
            return poly_res
        dirichlet_res = _simplex_dirichlet(region, expr, assumptions=assumptions)
        if dirichlet_res is not None:
            return dirichlet_res

    if isinstance(
        region,
        (DiskRegion, BallRegion, EllipsoidRegion, AnnulusRegion, SphericalShellRegion),
    ):
        poly_res = region.polynomial_moment(expr)
        if poly_res is not None:
            return poly_res
        radial = region.radial_integral(expr)
        if radial is not None:
            return radial

    delegated = semialgebraic_domain_moment(region, expr, assumptions=assumptions)
    if delegated is not None:
        record_strategy("semialgebraic_domain_moment")
        return delegated
    return None


def _try_graph_reversal(
    expr: sp.Expr,
    region: Region,
    assumptions,
    principal_value: bool,
) -> sp.Expr | None:
    """Safely reverse simple 2-D graph regions when it reduces dependency.

    This order-reversal shortcut applies only when the integrand depends on the
    inner variable but not the outer one.  Reversing then converts the problem
    into one where the first integration produces a simple geometric factor.
    """
    if not isinstance(region, GraphRegion):
        return None
    if region.outer_var is None or region.inner_var is None:
        return None
    outer = region.outer_var
    inner = region.inner_var
    if outer in expr.free_symbols or inner not in expr.free_symbols:
        return None
    pieces = region.reversed_pieces()
    if not pieces:
        return None

    from multiple_integrate.core import integrate

    total = sp.Integer(0)
    for piece in pieces:
        # Graph reversal returns inner-first ranges so recursive evaluation
        # preserves the public SymPy ordering convention.
        (_, inner_lo, inner_hi), (outer, lo, hi) = piece
        weight = _fast_simplify(sp.sympify(inner_hi) - sp.sympify(inner_lo))
        total += integrate(
            _fast_simplify(weight * expr),
            (outer, lo, hi),
            assumptions=assumptions,
            principal_value=principal_value,
        )
    return _fast_simplify(total)
