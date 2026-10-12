"""Exact analytic strategies used by the integration planner."""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp
from sympy import (
    Dummy,
    Heaviside,
    Piecewise,
    diff,
    oo,
)

from multiple_integrate.assumptions import _integrate
from multiple_integrate.recognition import (
    _fast_simplify,
    _heaviside_to_piecewise,
    _inner_expr_range,
)

_STRATEGY_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


def _recognize_additive_separable(
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
) -> tuple[dict[sp.Symbol, sp.Expr], sp.Expr, list[sp.Symbol], list[tuple], sp.Expr] | None:
    """Recognize an additive single-variable decomposition on a product domain."""
    split: dict[sp.Symbol, sp.Expr] = {}
    residual = sp.Integer(0)
    for term in g.args if g.is_Add else (g,):
        active = [var for var in vars_ if var in term.free_symbols]
        if not active:
            residual += term
        elif len(active) == 1:
            var = active[0]
            split[var] = split.get(var, sp.Integer(0)) + term
        else:
            return None
    if len(split) < 2:
        return None

    variable_set = set(vars_)
    if any(
        (sp.sympify(lower).free_symbols | sp.sympify(upper).free_symbols) & variable_set
        for _, lower, upper in ranges
    ):
        return None

    active_ranges = [entry for entry in ranges if entry[0] in split]
    active_vars = [entry[0] for entry in active_ranges]
    inactive_volume = sp.Integer(1)
    for var, lower, upper in ranges:
        if var in split:
            continue
        if lower in (oo, -oo) or upper in (oo, -oo):
            return None
        inactive_volume *= upper - lower
    return split, residual, active_vars, active_ranges, inactive_volume


def _additive_pushforward_density(
    component: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    opts: dict,
) -> tuple[sp.Expr, sp.Symbol, sp.Expr, sp.Expr] | None:
    """Return the one-dimensional pushforward density for one additive component."""
    image_lower, image_upper = _inner_expr_range(component, variable, lower, upper)
    image_var = Dummy("y_sep")
    try:
        cumulative = _integrate(
            _heaviside_to_piecewise(Heaviside(image_var - component)),
            (variable, lower, upper),
            **opts,
        )
        if cumulative.has(sp.Integral):
            return None
        density = _fast_simplify(diff(cumulative, image_var))
    except _STRATEGY_ERRORS:
        return None
    if image_var in density.free_symbols:
        return None
    density = Piecewise(
        (density, (image_var > image_lower) & (image_var < image_upper)),
        (sp.Integer(0), True),
    )
    return density, image_var, image_lower, image_upper


def _convolve_pushforward_densities(
    densities: list[tuple[sp.Expr, sp.Symbol, sp.Expr, sp.Expr]],
    opts: dict,
) -> tuple[sp.Expr, sp.Symbol, sp.Expr, sp.Expr] | None:
    """Convolve independent one-dimensional pushforward densities."""
    conv_var = Dummy("z_conv")
    density, image_var, lower, upper = densities[0]
    convolution = density.subs(image_var, conv_var)
    conv_lower, conv_upper = lower, upper

    for density, image_var, lower, upper in densities[1:]:
        integration_var = Dummy("t_conv")
        next_var = Dummy("z_new")
        integrand = _heaviside_to_piecewise(
            convolution.subs(conv_var, integration_var)
            * density.subs(image_var, next_var - integration_var)
        )
        integration_lower = sp.Max(conv_lower, next_var - upper)
        integration_upper = sp.Min(conv_upper, next_var - lower)
        try:
            convolution = _integrate(
                integrand,
                (integration_var, integration_lower, integration_upper),
                **opts,
            )
            if convolution.has(sp.Integral):
                return None
            convolution = _fast_simplify(convolution).subs(next_var, conv_var)
        except _STRATEGY_ERRORS:
            return None
        conv_lower += lower
        conv_upper += upper
    return convolution, conv_var, conv_lower, conv_upper


def _try_additive_separable(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """Evaluate an additive-separable pushforward by symbolic convolution."""
    recognized = _recognize_additive_separable(g, vars_, ranges)
    if recognized is None:
        return None
    split, residual, active_vars, active_ranges, inactive_volume = recognized

    densities = []
    for variable, lower, upper in active_ranges:
        density = _additive_pushforward_density(split[variable], variable, lower, upper, opts)
        if density is None:
            return None
        densities.append(density)
    convolved = _convolve_pushforward_densities(densities, opts)
    if convolved is None:
        return None
    convolution, convolution_var, lower, upper = convolved

    final_var = Dummy("y_final")
    try:
        result = _integrate(
            f_outer(final_var + residual) * convolution.subs(convolution_var, final_var),
            (final_var, lower, upper),
            **opts,
        )
        if result.has(sp.Integral):
            return None
        return _fast_simplify(inactive_volume * result)
    except _STRATEGY_ERRORS:
        return None


def _try_product_separable(
    f_expr: sp.Expr, vars_: list[sp.Symbol], ranges: list[tuple], opts: dict
) -> sp.Expr | None:
    """
    Handle integrands that factorise as a product of single-variable functions:

        f(x₁, …, xₙ) = c · f₁(x₁) · f₂(x₂) · … · fₙ(xₙ)

    By Fubini, the integral factors into independent 1-D integrals:

        ∫_Ω f dxⁿ = c · ∏ᵢ ∫_{aᵢ}^{bᵢ} fᵢ(xᵢ) dxᵢ

    This handles sin(x)·exp(-y), x²·cos(y), exp(-x)·exp(-y), and similar
    product integrands that appear very frequently in practice.

    Only fires when every factor depends on at most one integration variable
    and limits are independent (non-variable).
    """
    if not f_expr.is_Mul:
        return None

    vars_set = set(vars_)

    for r in ranges:
        if sp.sympify(r[1]).free_symbols & vars_set:
            return None
        if sp.sympify(r[2]).free_symbols & vars_set:
            return None

    const_part = sp.Integer(1)
    var_factors: dict[sp.Symbol, sp.Expr] = {}

    for factor in f_expr.args:
        active = [v for v in vars_ if v in factor.free_symbols]
        if len(active) == 0:
            const_part = const_part * factor
        elif len(active) == 1:
            v = active[0]
            var_factors[v] = var_factors.get(v, sp.Integer(1)) * factor
        else:
            return None  # factor mixes variables

    if len(var_factors) < 2:
        return None

    result = const_part
    for r in ranges:
        v, lo, hi = r
        if v in var_factors:
            from .strategy_common import _try_standard_1d

            integral_1d = _try_standard_1d(var_factors[v], v, lo, hi, opts)
            if integral_1d is None:
                integral_1d = _integrate(var_factors[v], (v, lo, hi), **opts)
            if integral_1d.has(sp.Integral):
                return None
            result = result * integral_1d
        else:
            if lo in (oo, -oo) or hi in (oo, -oo):
                return None
            result = result * (hi - lo)

    return _fast_simplify(result)
