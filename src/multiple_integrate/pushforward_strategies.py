"""Exact analytic strategies used by the integration planner."""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp
from sympy import (
    Abs,
    Dummy,
    diff,
    limit,
    oo,
    simplify,
    solve,
)

from multiple_integrate.assumptions import _ask, _integrate
from multiple_integrate.recognition import (
    _fast_simplify,
    _find_critical_points,
    _quadratic_coefficients,
    _run_with_signal_timeout,
)

_STRATEGY_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


def _try_linear_pushforward(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """
    ∫_{[0,∞)^n} f(b·x + c) dx  reduced to a 1-D integral via simplex measure.

    As x ranges over [0,∞)^n the linear form g = b·x + c ranges over
    [c, ∞) when all bᵢ > 0, or (-∞, c] when all bᵢ < 0.

    All-positive b:
        1/(∏bᵢ·(n-1)!) ∫_c^∞ (y-c)^{n-1} f(y) dy

    All-negative b:
        1/(∏|bᵢ|·(n-1)!) ∫_{-∞}^c (c-y)^{n-1} f(y) dy

    Mixed-sign b cannot be handled by this formula; returns None.
    """
    if not all(r[1] == 0 and r[2] == oo for r in ranges):
        return None
    try:
        c, b_vec, A = _quadratic_coefficients(g, vars_)
    except _STRATEGY_ERRORS:
        return None
    n = len(vars_)
    if sp.zeros(n, n) != A:
        return None
    b_list = list(b_vec)
    if any(bi == 0 for bi in b_list):
        return None

    all_pos = all(_ask(sp.Q.positive(bi)) for bi in b_list)
    all_neg = all(_ask(sp.Q.negative(bi)) for bi in b_list)
    if not all_pos and not all_neg:
        return None

    def _integrate_timed(expr, bounds, secs=3.0):
        try:
            return _run_with_signal_timeout(
                lambda: _integrate(expr, bounds, **opts),
                secs,
                sp.Integral(expr, bounds),
            )
        except (NotImplementedError, ValueError, TypeError):
            return None

    y = Dummy("y")
    abs_b_prod = sp.prod([sp.Abs(bi) for bi in b_list])
    prefactor = sp.Integer(1) / (abs_b_prod * sp.factorial(n - 1))
    if all_pos:
        integrand = prefactor * (y - c) ** (n - 1) * f_outer(y)
        result = _integrate_timed(integrand, (y, c, oo))
    else:  # all_neg: g decreases from c to -\infty
        integrand = prefactor * (c - y) ** (n - 1) * f_outer(y)
        result = _integrate_timed(integrand, (y, -oo, c))
    return result


def _try_monotone_change(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """
    For a single-variable g(x), if g is monotone on [lo, hi]:

        ∫_lo^hi f(g(x)) dx  =  ∫_{g(lo)}^{g(hi)} f(y) / |g'(g⁻¹(y))| dy

    Uses the co-area formula:  μ'(y) = |dx/dy| = 1/|g'(x)|.

    For multivariate f(g(x)) where g depends only on one variable,
    the other dimensions are integrated out as a volume factor first.
    """
    active = [v for v in vars_ if v in g.free_symbols]
    if len(active) != 1:
        return None
    xi = active[0]
    r_xi = next((r for r in ranges if r[0] == xi), None)
    if r_xi is None:
        return None
    lo, hi = r_xi[1], r_xi[2]
    # A one-dimensional substitution is valid only when its integration
    # interval is independent of every inactive integration variable.
    if (sp.sympify(lo).free_symbols | sp.sympify(hi).free_symbols) & (set(vars_) - {xi}):
        return None

    cpts = _find_critical_points(g, xi, lo, hi)
    if cpts:
        return None

    dg = diff(g, xi)
    try:
        mid = (lo + hi) / 2 if lo not in (-oo, oo) and hi not in (-oo, oo) else sp.Integer(0)
        dg_sign = _ask(sp.Q.positive(dg.subs(xi, mid)))
    except _STRATEGY_ERRORS:
        dg_sign = None

    g_lo = limit(g, xi, lo, "+") if lo == -oo else g.subs(xi, lo)
    g_hi = limit(g, xi, hi, "-") if hi == oo else g.subs(xi, hi)
    g_lo, g_hi = simplify(g_lo), simplify(g_hi)

    if dg_sign is False:  # decreasing → flip
        g_lo, g_hi = g_hi, g_lo

    y = Dummy("y_mono")
    try:
        inv_solutions = solve(g - y, xi)
    except _STRATEGY_ERRORS:
        return None
    inv_solutions = [s for s in inv_solutions if not s.has(sp.I)]
    if not inv_solutions:
        return None

    if len(inv_solutions) > 1:
        valid = []
        for s in inv_solutions:
            try:
                s_mid = s.subs(y, (g_lo + g_hi) / 2)
                ok = _ask(
                    sp.Q.positive(s_mid - lo + sp.Rational(1, 1000))
                    & sp.Q.positive(hi - s_mid + sp.Rational(1, 1000))
                )
                if ok is not False:
                    valid.append(s)
            except _STRATEGY_ERRORS:
                valid.append(s)
        if len(valid) != 1:
            return None
        inv_solutions = valid

    xi_of_y = simplify(inv_solutions[0])
    jacobian = Abs(diff(xi_of_y, y))  # |dx/dy| = 1/|g'(x)|

    other_ranges = [r for r in ranges if r[0] != xi]
    volume = sp.Integer(1)
    for r in other_ranges:
        v, vlo, vhi = r
        vlo_s, vhi_s = sp.sympify(vlo), sp.sympify(vhi)
        if vlo_s in (-oo, oo) or vhi_s in (-oo, oo):
            return None
        if (vlo_s.free_symbols | vhi_s.free_symbols) & {xi}:
            return None
        volume *= vhi_s - vlo_s

    integrand_1d = _fast_simplify(f_outer(y) * jacobian * volume)
    try:
        result = _integrate(integrand_1d, (y, g_lo, g_hi), **opts)
        return None if result.has(sp.Integral) else simplify(result)
    except _STRATEGY_ERRORS:
        return None


def _try_piecewise_change(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """
    Split the domain at critical points of g(x), apply the monotone
    substitution on each piece, and sum.

    The co-area density is:
        μ'(y) = Σ_{branches k : g(xₖ)=y}  1 / |g'(xₖ)|

    Only handles the single-active-variable case.
    """
    active = [v for v in vars_ if v in g.free_symbols]
    if len(active) != 1:
        return None
    xi = active[0]
    r_xi = next(r for r in ranges if r[0] == xi)
    lo, hi = r_xi[1], r_xi[2]

    cpts = _find_critical_points(g, xi, lo, hi)
    if not cpts:
        return None

    endpoints = [lo] + sorted(cpts, key=lambda e: float(e) if e.is_number else 0) + [hi]
    sub_intervals = list(zip(endpoints[:-1], endpoints[1:], strict=False))

    other_ranges = [r for r in ranges if r[0] != xi]
    volume = sp.Integer(1)
    for r in other_ranges:
        v, vlo, vhi = r
        if vlo in (-oo, oo) or vhi in (-oo, oo):
            return None
        volume *= vhi - vlo

    total = sp.Integer(0)
    for a, b in sub_intervals:
        sub_r = [(xi, a, b)] + other_ranges
        sub_vars = [xi] + [r[0] for r in other_ranges]
        piece = _try_monotone_change(f_outer, g, sub_vars, sub_r, opts)
        if piece is None:
            try:
                piece = _integrate(f_outer(g) * volume, (xi, a, b), **opts)
                if piece.has(sp.Integral):
                    return None
            except _STRATEGY_ERRORS:
                return None
        total = total + piece

    result = simplify(total)
    return None if result.has(sp.Integral) else result
