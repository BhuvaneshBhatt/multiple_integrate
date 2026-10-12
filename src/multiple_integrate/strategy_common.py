"""Exact analytic strategies used by the integration planner."""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp
from sympy import (
    oo,
)

from multiple_integrate.adapters import function_sign as funcprops_sign
from multiple_integrate.adapters import semialgebraic_measure as semialg_measure
from multiple_integrate.assumptions import _ask, _integrate, _normalize_assumptions
from multiple_integrate.planner import record_strategy
from multiple_integrate.recognition import (
    _fast_simplify,
    _heaviside_to_piecewise,
)
from multiple_integrate.region_types import (
    Region,
)

_STRATEGY_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


def _const_result(
    expr: sp.Expr,
    region: Region,
    parsed_ranges: list[tuple],
    *,
    use_region_volume: bool = True,
    assumptions=None,
) -> sp.Expr:
    """Integrate a constant without erasing explicit range orientation."""
    expr = sp.sympify(expr)
    if expr == 0:
        return sp.Integer(0)
    volume = None
    if use_region_volume:
        from .assumptions import _normalize_assumptions
        from .cell_proofs import nonnegative
        from .region_types import BoxRegion

        if isinstance(region, BoxRegion):
            context = _normalize_assumptions(assumptions)
            widths = []
            for _, lo, hi in region.ranges:
                width = sp.sympify(hi) - sp.sympify(lo)
                widths.append(
                    width
                    if nonnegative(width, context)
                    else -width
                    if nonnegative(-width, context)
                    else sp.Abs(width)
                )
            volume = sp.prod(widths)
        else:
            volume = region.constant_volume()
    if volume is None and use_region_volume:
        volume = semialg_measure(region)
        if volume is not None:
            record_strategy("semialg_measure")
    if volume is None:
        volume = _iterated_integrate(sp.Integer(1), parsed_ranges, {})
    return _fast_simplify(expr * volume)


def _inactive_finite_volume(
    active_vars: list[sp.Symbol],
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    *,
    simplify: bool = True,
) -> sp.Expr | None:
    """Return the product of inactive finite dimensions when it's safe.
    This succeeds only when every inactive bound is finite and independent of
    the active variables, and active bounds do not depend on inactive vars.
    """
    active_set = set(active_vars)
    all_set = set(vars_)
    inactive_set = all_set - active_set
    volume = sp.Integer(1)
    for v, lo, hi in ranges:
        lo_s = sp.sympify(lo)
        hi_s = sp.sympify(hi)
        if v in active_set:
            if (lo_s.free_symbols | hi_s.free_symbols) & inactive_set:
                return None
            continue
        if lo_s in (-oo, oo) or hi_s in (-oo, oo):
            return None
        if (lo_s.free_symbols | hi_s.free_symbols) & active_set:
            return None
        volume *= hi_s - lo_s
    return _fast_simplify(volume) if simplify else volume


def _should_try_layercake(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
) -> bool:
    """Entry point for the expensive generic layer-cake fallback."""
    active_vars = [v for v in vars_ if v in g.free_symbols]
    if len(active_vars) > 1:
        return False
    return not (
        len(active_vars) == 1 and _inactive_finite_volume(active_vars, vars_, ranges) is None
    )


def _try_standard_1d(expr: sp.Expr, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr, opts: dict):
    """Tiny recognizers for common exact 1-D definite integrals."""
    from .fast_rules import fast_interval_rule

    fast = fast_interval_rule(expr, var, lo, hi, opts.get("_mi_assumptions"), evidence=False)
    if fast is not None:
        return fast[1]
    lo_s, hi_s = sp.sympify(lo), sp.sympify(hi)
    try:
        # Elementary antiderivative formulas are safe only on finite endpoints.
        # Improper exponential/trigonometric integrals require convergence analysis
        # and are delegated to SymPy below.
        finite_endpoints = lo_s not in (-oo, oo) and hi_s not in (-oo, oo)
        if finite_endpoints and expr.func == sp.exp:
            arg = sp.expand(expr.args[0])
            poly = sp.Poly(arg, var)
            if poly.degree() <= 1:
                a = poly.nth(1)
                b = poly.nth(0)
                if a != 0:
                    value = _fast_simplify(sp.exp(b) * (sp.exp(a * hi_s) - sp.exp(a * lo_s)) / a)
                    if a.is_zero is False:
                        return value
                    return sp.Piecewise((sp.exp(b) * (hi_s - lo_s), sp.Eq(a, 0)), (value, True))
        if finite_endpoints and expr.func in (sp.sin, sp.cos):
            arg = sp.expand(expr.args[0])
            poly = sp.Poly(arg, var)
            if poly.degree() <= 1:
                a = poly.nth(1)
                b = poly.nth(0)
                if a != 0:
                    if expr.func == sp.sin:
                        value = _fast_simplify((-sp.cos(a * hi_s + b) + sp.cos(a * lo_s + b)) / a)
                    else:
                        value = _fast_simplify((sp.sin(a * hi_s + b) - sp.sin(a * lo_s + b)) / a)
                    if a.is_zero is False:
                        return value
                    return sp.Piecewise((expr.func(b) * (hi_s - lo_s), sp.Eq(a, 0)), (value, True))
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError, ZeroDivisionError):
        pass
    try:
        factors = sp.Mul.make_args(expr)
        exp_factor = next((f for f in factors if f.func == sp.exp), None)
        if exp_factor is not None:
            rest = _fast_simplify(expr / exp_factor)
            q = sp.expand(exp_factor.args[0])
            qpoly = sp.Poly(q, var)
            if qpoly.degree() <= 2:
                a = qpoly.nth(2)
                b = qpoly.nth(1)
                c = qpoly.nth(0)
                if a != 0 and not rest.has(sp.exp) and sp.Poly(rest, var) is not None:
                    res = _integrate(
                        sp.expand(rest) * sp.exp(a * var**2 + b * var + c),
                        (var, lo_s, hi_s),
                        **opts,
                    )
                    if not isinstance(res, sp.Integral):
                        return _fast_simplify(res)
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError, ZeroDivisionError):
        pass
    try:
        num, den = sp.fraction(sp.together(expr))
        if sp.simplify(num).free_symbols.isdisjoint({var}) and lo_s == -oo and hi_s == oo:
            dpoly = sp.Poly(sp.expand(den), var)
            if dpoly.degree() == 2 and dpoly.nth(1) == 0:
                a2 = _fast_simplify(dpoly.nth(0) / dpoly.nth(2))
                if _ask(sp.Q.positive(a2)):
                    return _fast_simplify(sp.pi * num / (dpoly.nth(2) * sp.sqrt(a2)))
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError, ZeroDivisionError):
        pass
    return None


def _iterated_integrate(expr: sp.Expr, ranges: list[tuple], opts: dict) -> sp.Expr:
    """
    Iterated SymPy integration in forward order (first range integrated first).

    Forward order is required so that variable limits are respected correctly.
    For example, with ranges [(y, 0, 1-x), (x, 0, 1)], y must be integrated
    first because its upper limit depends on x.  Reversing the order would
    integrate x first, leaving x free when y's limits are applied.

    Heaviside sub-expressions are rewritten as Piecewise before integration
    because SymPy's _integrate() falls back to Meijer G functions for
    Heaviside(linear(x, y)) with two free variables, producing incorrect results.
    """
    result = _heaviside_to_piecewise(expr)
    for r in ranges:
        v, lo, hi = r
        std = _try_standard_1d(result, v, lo, hi, opts)
        result = std if std is not None else _integrate(result, (v, lo, hi), **opts)
    return result


def _symbolically_positive(expr: sp.Expr, assumptions=None) -> bool:
    """Best-effort positivity test honoring normalized SymPy assumptions."""
    expr = sp.sympify(expr)
    if expr.is_positive is True:
        return True
    context = _normalize_assumptions(assumptions)
    try:
        if _ask(sp.Q.positive(expr), context) is True:
            return True
    except ValueError:
        raise
    classification = funcprops_sign(expr, assumptions=context.predicate_expr)
    if classification == "positive":
        return True
    try:
        val = sp.N(expr)
        if val.is_real and float(val) > 0:
            return True
    except (TypeError, ValueError, OverflowError):
        pass
    return False
