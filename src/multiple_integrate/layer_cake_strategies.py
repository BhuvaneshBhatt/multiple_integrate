"""Layer-cake type strategies used by the integration planner."""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp
from sympy import (
    Dummy,
    Heaviside,
    diff,
    gamma,
    limit,
    oo,
    simplify,
)

from multiple_integrate.assumptions import _integrate
from multiple_integrate.recognition import (
    _fast_simplify,
    _find_critical_points,
    _heaviside_to_piecewise,
    _parse_superellipse_core,
)
from multiple_integrate.strategy_common import _inactive_finite_volume, _should_try_layercake

_STRATEGY_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


def _try_poly_layercake(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """
    Layer-cake via symbolic Heaviside integral. Works for any polynomial g
    on a bounded or semi-infinite domain.

    Skipped when g depends on more than one variable: integrating
    Piecewise(y_dummy - g(x1, x2, ...) < 0, ...) over multiple variables
    causes SymPy to hang on Meijer-G reduction. Those cases fall through
    to _iterated_integrate which handles them correctly.
    """
    active_vars = [v for v in vars_ if v in g.free_symbols]
    if len(active_vars) > 1:
        return None
    if len(active_vars) == 1 and _inactive_finite_volume(active_vars, vars_, ranges) is None:
        return None
    y = Dummy("y")
    mu_y = _heaviside_to_piecewise(Heaviside(y - g))
    try:
        for r in ranges:
            mu_y = _integrate(mu_y, (r[0], r[1], r[2]), **opts)
            if mu_y.has(sp.Integral):
                return None
    except _STRATEGY_ERRORS:
        return None

    density = simplify(diff(mu_y, y))

    y_vals = []
    for r in ranges:
        for ep in [r[1], r[2]]:
            if ep not in (oo, -oo):
                y_vals.append(g.subs(r[0], ep))
    y_min = sp.Min(*y_vals) if y_vals else -oo
    y_max = sp.Max(*y_vals) if y_vals else oo

    try:
        result = _integrate(f_outer(y) * density, (y, y_min, y_max), **opts)
        return None if result.has(sp.Integral) else result
    except _STRATEGY_ERRORS:
        return None


def _try_superellipse(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """Fast homogeneous layer-cake for orthant superellipse-type integrals."""
    if len(vars_) < 2:
        return None

    for _, lo, hi in ranges:
        if sp.sympify(lo) != 0 or sp.sympify(hi) != oo:
            return None

    parsed = _parse_superellipse_core(g, vars_)
    if parsed is None:
        return None
    _, k, term_map = parsed

    alpha = sp.Integer(0)
    const = sp.Integer(1)
    for v in vars_:
        coeff, pwr = term_map[v]
        alpha += sp.Integer(1) / pwr
        const *= gamma(1 + sp.Integer(1) / pwr) / coeff ** (sp.Integer(1) / pwr)
    const /= gamma(1 + alpha)

    t = Dummy("y_super")
    density = _fast_simplify(const * alpha / k * t ** (alpha / k - 1))

    # Positive-orthant superellipse densities have a power-law tail;
    # If the outer function tends to +∞, or to a positive nonzero constant, the
    # weighted tail integral diverges because alpha/k > 0. This avoids SymPy
    # producing opaque lowergamma(..., -oo) style results on obvious cases.
    try:
        outer_t = f_outer(t)
        tail_lim = limit(outer_t, t, oo)
        if tail_lim is oo:
            return oo
        if getattr(tail_lim, "is_positive", False) and tail_lim != 0:
            return oo
    except _STRATEGY_ERRORS:
        outer_t = f_outer(t)

    try:
        result = _integrate(outer_t * density, (t, 0, oo), **opts)
    except _STRATEGY_ERRORS:
        return None
    return None if result.has(sp.Integral) else _fast_simplify(result)


def _estimate_inner_bounds(
    g: sp.Expr, vars_: list[sp.Symbol], ranges: list[tuple]
) -> tuple[sp.Expr, sp.Expr]:
    """
    Estimate [g_min, g_max] by evaluating g at all corners of the box and
    at critical points along each axes.
    """
    corners = [{}]
    for r in ranges:
        v, lo, hi = r
        new_corners = []
        for c in corners:
            for ep in [lo, hi]:
                if ep not in (oo, -oo):
                    new_corners.append({**c, v: ep})
        if new_corners:
            corners = new_corners

    vals = []
    for corner in corners:
        try:
            val = simplify(g.subs(list(corner.items())))
            if val.is_real or val.is_number:
                vals.append(val)
        except _STRATEGY_ERRORS:
            pass

    for r in ranges:
        v, lo, hi = r
        cpts = _find_critical_points(g, v, lo, hi)
        for cp in cpts:
            try:
                val = g.subs(v, cp)
                vals.append(simplify(val))
            except _STRATEGY_ERRORS:
                pass

    if not vals:
        return -oo, oo
    return simplify(sp.Min(*vals)), simplify(sp.Max(*vals))


def _try_general_layercake(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """
    Applies the layer-cake formula for arbitrary g:

        ∫_Ω f(g(x)) dx = ∫_{y_min}^{y_max} f(y) · μ'(y) dy

    where μ(y) = ∫_Ω Θ(y - g(x)) dx is computed symbolically by SymPy.

    Here ``g`` may be transcendental; SymPy must be able to integrate
    ``Heaviside(y - g(x))`` in closed form.
    """
    if not _should_try_layercake(f_outer, g, vars_, ranges):
        return None

    yy = Dummy("y_gen")

    mu_y = _heaviside_to_piecewise(Heaviside(yy - g))
    try:
        for r in ranges:
            mu_y = _integrate(mu_y, (r[0], r[1], r[2]), **opts)
            if isinstance(mu_y, sp.Integral) or mu_y.has(sp.Integral):
                return None
        mu_y = simplify(mu_y)
    except _STRATEGY_ERRORS:
        return None

    density = simplify(diff(mu_y, yy))

    y_lo, y_hi = _estimate_inner_bounds(g, vars_, ranges)

    try:
        result = _integrate(f_outer(yy) * density, (yy, y_lo, y_hi), **opts)
        if result.has(sp.Integral):
            return None
        return simplify(result)
    except _STRATEGY_ERRORS:
        return None
