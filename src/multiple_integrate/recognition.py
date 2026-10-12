"""Structural recognition helpers for integrands and integration ranges."""

from __future__ import annotations

import functools
import gc
import math
import signal
import threading
import time
from collections.abc import Callable

import sympy as sp
from sympy import (
    Dummy,
    Heaviside,
    Piecewise,
    diff,
    oo,
    solve,
)
from sympy.matrices import Matrix

from multiple_integrate.adapters import function_range as funcprops_range
from multiple_integrate.adapters import function_symmetry as funcprops_symmetry
from multiple_integrate.assumptions import _ask

_STRATEGY_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


class _IntegrandDecomposition:
    """
    Result of decomposing an integrand F(x₁,…,xₙ) into f ∘ g.

    Attributes
    ----------
    f_outer : Callable   – univariate function, maps SymPy expr -> SymPy expr
    g_inner : sp.Expr    – the "inner" expression in the integration variables
    is_polynomial : bool – True if g_inner is a polynomial in the variables
    """

    __slots__ = ("f_outer", "g_inner", "is_polynomial")

    def __init__(self, f_outer: Callable, g_inner: sp.Expr, is_polynomial: bool):
        self.f_outer = f_outer
        self.g_inner = g_inner
        self.is_polynomial = is_polynomial


def _normalize_seq(obj):
    """Normalize a list/tuple-like input to a tuple."""
    if isinstance(obj, (list, tuple)):
        return tuple(obj)
    return (obj,)


def _vars_set(vars_: list[sp.Symbol] | tuple[sp.Symbol, ...]) -> set[sp.Symbol]:
    return set(vars_)


def _depends_on_vars(expr: sp.Expr, vars_set: set[sp.Symbol]) -> bool:
    return bool(sp.sympify(expr).free_symbols & vars_set)


def _is_constant_wrt(expr: sp.Expr, vars_set: set[sp.Symbol]) -> bool:
    return not _depends_on_vars(expr, vars_set)


def _clean_expr(expr: sp.Expr) -> sp.Expr:
    expr = sp.sympify(expr)
    try:
        return sp.simplify(expr)
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError):
        return expr


def _split_unary_wrapper(
    expr: sp.Expr, vars_: list[sp.Symbol], vars_set: set[sp.Symbol]
) -> _IntegrandDecomposition | None:
    n_var_args = sum(1 for a in expr.args if _depends_on_vars(a, vars_set))
    if n_var_args != 1:
        return None
    inner = next(a for a in expr.args if _depends_on_vars(a, vars_set))
    t = Dummy("t")
    try:
        outer = sp.Lambda(t, expr.subs(inner, t))
        return _IntegrandDecomposition(outer, inner, is_polynomial=_is_polynomial(inner, vars_))
    except (TypeError, ValueError, NotImplementedError):
        return None


def _split_product_constants(
    expr: sp.Expr, vars_: list[sp.Symbol], vars_set: set[sp.Symbol]
) -> _IntegrandDecomposition | None:
    if not expr.is_Mul:
        return None
    const_part = sp.Integer(1)
    var_part = sp.Integer(1)
    for factor in expr.args:
        if _depends_on_vars(factor, vars_set):
            var_part *= factor
        else:
            const_part *= factor
    if const_part == 1 or var_part == 1:
        return None
    sub = _decompose_integrand(var_part, vars_)
    if sub is None:
        return None
    inner_f = sub.f_outer
    t = Dummy("t")
    outer = sp.Lambda(t, const_part * inner_f(t))
    return _IntegrandDecomposition(outer, sub.g_inner, is_polynomial=sub.is_polynomial)


def _split_sum_constants(
    expr: sp.Expr, vars_: list[sp.Symbol], vars_set: set[sp.Symbol]
) -> _IntegrandDecomposition | None:
    if not expr.is_Add:
        return None
    const_part = sp.Integer(0)
    var_part = sp.Integer(0)
    for term in expr.args:
        if _depends_on_vars(term, vars_set):
            var_part += term
        else:
            const_part += term
    if const_part == 0 or var_part == 0:
        return None
    sub = _decompose_integrand(var_part, vars_)
    if sub is None:
        return None
    inner_f = sub.f_outer
    t = Dummy("t")
    outer = sp.Lambda(t, inner_f(t) + const_part)
    return _IntegrandDecomposition(outer, sub.g_inner, is_polynomial=sub.is_polynomial)


def _split_single_variable(
    expr: sp.Expr, vars_: list[sp.Symbol], vars_set: set[sp.Symbol]
) -> _IntegrandDecomposition | None:
    active = [v for v in vars_ if v in sp.sympify(expr).free_symbols]
    if len(active) != 1:
        return None
    t = Dummy("t")
    return _IntegrandDecomposition(sp.Lambda(t, t), expr, is_polynomial=_is_polynomial(expr, vars_))


def _split_composition_once(
    expr: sp.Expr, vars_: list[sp.Symbol]
) -> _IntegrandDecomposition | None:
    """Single-pass decomposition used as the base case for recursive peeling."""
    vars_set = _vars_set(vars_)

    try:
        sp.Poly(expr, *vars_)
        t = Dummy("t")
        return _IntegrandDecomposition(sp.Lambda(t, t), expr, is_polynomial=True)
    except sp.PolynomialError:
        pass

    if expr.func in (sp.log, sp.Abs, sp.sign, sp.floor, sp.ceiling):
        sub = _split_single_variable(expr, vars_, vars_set)
        if sub is not None:
            return sub

    sub = _split_unary_wrapper(expr, vars_, vars_set)
    if sub is not None:
        return sub

    if expr.is_Pow:
        base, exp_ = expr.args
        if _is_constant_wrt(exp_, vars_set) and _depends_on_vars(base, vars_set):
            t = Dummy("t")
            return _IntegrandDecomposition(
                sp.Lambda(t, t**exp_), base, is_polynomial=_is_polynomial(base, vars_)
            )

    sub = _split_product_constants(expr, vars_, vars_set)
    if sub is not None:
        return sub

    sub = _split_sum_constants(expr, vars_, vars_set)
    if sub is not None:
        return sub

    return _split_single_variable(expr, vars_, vars_set)


@functools.lru_cache(maxsize=256)
def _cached_decomposition(
    expr: sp.Expr, vars_tuple: tuple[sp.Symbol, ...]
) -> _IntegrandDecomposition | None:
    """Cache the inexpensive one-pass composition decomposition across calls."""
    return _split_composition_once(expr, list(vars_tuple))


def _decompose_integrand(expr: sp.Expr, vars_: list[sp.Symbol]) -> _IntegrandDecomposition | None:
    """Recursively peel wrappers to expose a deeper public decomposition."""
    vars_set = _vars_set(vars_)

    if expr.func in (sp.log, sp.Abs, sp.sign, sp.floor, sp.ceiling):
        sub = _split_single_variable(expr, vars_, vars_set)
        if sub is not None:
            return sub

    n_var_args = sum(1 for a in expr.args if _depends_on_vars(a, vars_set))
    if n_var_args == 1:
        inner = next(a for a in expr.args if _depends_on_vars(a, vars_set))
        t = Dummy("t")
        try:
            outer = sp.Lambda(t, expr.subs(inner, t))
            sub = _decompose_integrand(inner, vars_)
            if sub is not None:
                u = Dummy("u")
                try:
                    return _IntegrandDecomposition(
                        sp.Lambda(u, outer(sub.f_outer(u))),
                        sub.g_inner,
                        sub.is_polynomial,
                    )
                except _STRATEGY_ERRORS:
                    pass
            return _IntegrandDecomposition(outer, inner, is_polynomial=_is_polynomial(inner, vars_))
        except _STRATEGY_ERRORS:
            pass

    if expr.is_Pow:
        base, exp_ = expr.args
        if _is_constant_wrt(exp_, vars_set) and _depends_on_vars(base, vars_set):
            sub = _decompose_integrand(base, vars_)
            t = Dummy("t")
            if sub is not None:
                u = Dummy("u")
                try:
                    return _IntegrandDecomposition(
                        sp.Lambda(u, sub.f_outer(u) ** exp_),
                        sub.g_inner,
                        sub.is_polynomial,
                    )
                except _STRATEGY_ERRORS:
                    pass
            return _IntegrandDecomposition(
                sp.Lambda(t, t**exp_), base, is_polynomial=_is_polynomial(base, vars_)
            )

    if expr.is_Mul:
        const_part = sp.Integer(1)
        var_part = sp.Integer(1)
        for factor in expr.args:
            if _depends_on_vars(factor, vars_set):
                var_part *= factor
            else:
                const_part *= factor
        if const_part != 1 and var_part != 1:
            sub = _decompose_integrand(var_part, vars_)
            if sub is not None:
                t = Dummy("t")
                return _IntegrandDecomposition(
                    sp.Lambda(t, const_part * sub.f_outer(t)),
                    sub.g_inner,
                    sub.is_polynomial,
                )

    if expr.is_Add:
        const_part = sp.Integer(0)
        var_part = sp.Integer(0)
        for term in expr.args:
            if _depends_on_vars(term, vars_set):
                var_part += term
            else:
                const_part += term
        if const_part != 0 and var_part != 0:
            sub = _decompose_integrand(var_part, vars_)
            if sub is not None:
                t = Dummy("t")
                return _IntegrandDecomposition(
                    sp.Lambda(t, sub.f_outer(t) + const_part),
                    sub.g_inner,
                    sub.is_polynomial,
                )

    return _split_composition_once(expr, vars_)


def _is_polynomial(expr: sp.Expr, vars_: list[sp.Symbol]) -> bool:
    try:
        sp.Poly(expr, *vars_)
        return True
    except sp.PolynomialError:
        return False


def _quadratic_coefficients(poly: sp.Expr, vars_: list[sp.Symbol]):
    """
    Return (constant, linear_vector, quadratic_matrix) for a degree-≤2 polynomial.
    Raises ValueError for higher-degree polynomials.
    """
    poly_obj = sp.Poly(poly, *vars_)
    if sp.degree(poly_obj) > 2:
        raise ValueError("Polynomial degree > 2")

    n = len(vars_)
    c = poly_obj.nth(*([0] * n))
    b = Matrix([poly_obj.nth(*([1 if j == i else 0 for j in range(n)])) for i in range(n)])
    A = sp.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            idx = [0] * n
            if i == j:
                idx[i] = 2
                coeff = poly_obj.nth(*idx)
            else:
                idx[i] = 1
                idx[j] = 1
                coeff = poly_obj.nth(*idx) / 2
            A[i, j] = coeff
            A[j, i] = coeff
    return c, b, A


def _is_even_function(expr: sp.Expr, var: sp.Symbol) -> bool:
    if sp.simplify(expr.subs(var, -var) - expr) == 0:
        return True
    classification = funcprops_symmetry(expr, var)
    return classification in {"even", "both"}


def _heaviside_to_piecewise(expr: sp.Expr) -> sp.Expr:
    """
    Rewrite every Heaviside sub-expression as Piecewise before integration.

    SymPy's _integrate() falls back to Meijer G functions when it encounters
    Heaviside(linear(x, y)) with two free symbolic variables, producing
    unevaluated or incorrect results.  Rewriting to Piecewise first lets
    SymPy's piecewise integration machinery handle it correctly instead.
    """
    return expr.rewrite(Heaviside, Piecewise)


def _simplify_preserving_sums(expr):
    """Simplify outer algebra without rewriting certified infinite summands.

    Hypergeometric coefficients can be expanded to Lerch functions whose numeric
    evaluation performs a fresh quadrature at every summation index. Treating
    complete sums as opaque also prevents an unrelated simplifier from launching
    another symbolic summation. Substitution reconstructs the original exact Sum.
    """
    expr = sp.sympify(expr)
    sums = expr.atoms(sp.Sum)
    if not sums:
        return sp.simplify(expr)
    placeholders = {value: sp.Dummy("exact_sum") for value in sums}
    return sp.simplify(expr.xreplace(placeholders)).xreplace(
        {symbol: value for value, symbol in placeholders.items()}
    )


def _fast_simplify(expr: sp.Expr) -> sp.Expr:
    """
    Faster alternative to sympy.simplify for expressions arising in integration.
    Tries cancel (rational), trigsimp (trig), and falls back to simplify only
    when the expression is not already in a reduced form.
    """
    if expr.has(sp.Sum):
        return _simplify_preserving_sums(expr)
    if expr.is_number or expr.is_symbol:
        return expr
    try:
        c = sp.cancel(expr)
        if c != expr:
            return c
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError):
        pass
    try:
        t = sp.trigsimp(expr)
        if t != expr:
            return t
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError):
        pass
    if sp.count_ops(expr) < 40:
        try:
            return sp.simplify(expr)
        except (sp.PolynomialError, TypeError, ValueError, NotImplementedError):
            pass
    return expr


def _split_additive_terms(expr: sp.Expr) -> list[sp.Expr] | None:
    """Return additive terms for early sum splitting when worthwhile."""
    if expr.is_Add and len(expr.args) > 1:
        return list(expr.args)
    return None


def _signal_timeout_ready() -> bool:
    """Whether POSIX interval timers are usable in the current execution context."""
    return (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "ITIMER_REAL")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )


def _run_with_signal_timeout(func, seconds: float, timeout_value):
    """Bound an optional heuristic without taking ownership of a caller's alarm.

    Automatic GC is postponed during this short main-thread operation so a
    timeout cannot be lost inside a GC callback. Repeated delivery also protects
    against a callback explicitly swallowing the first interruption.
    """
    if not _signal_timeout_ready() or not math.isfinite(seconds) or seconds <= 0:
        return timeout_value
    old_delay, old_interval = signal.getitimer(signal.ITIMER_REAL)
    if old_delay > 0 or old_interval > 0:
        return timeout_value

    class _TimedOut(BaseException):
        pass

    def expire(*_):
        raise _TimedOut()

    old_handler = signal.getsignal(signal.SIGALRM)
    gc_enabled = gc.isenabled()
    if gc_enabled:
        gc.disable()
    started = time.monotonic()
    try:
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, seconds, min(seconds, 0.05))
        try:
            value = func()
        except _TimedOut:
            return timeout_value
        # A swallowed interruption must never turn into a successful heuristic.
        return timeout_value if time.monotonic() - started >= seconds else value
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if gc_enabled:
            gc.enable()


def _find_critical_points(g: sp.Expr, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr) -> list[sp.Expr]:
    """
    Return sorted list of real critical points of g(var) strictly inside (lo, hi).
    Includes pts where g is not differentiable (e.g. |x| at 0).
    Cached because monotone and piecewise-monotone analysis share these critical points.
    """
    lo = sp.sympify(lo)
    hi = sp.sympify(hi)
    pts = []

    # Solving g'(x)=0 can explode on transcendental inputs, so cap that step
    # and continue with any critical points that were found quickly.
    def _solve_timed(expr, var, secs=1.0):
        if not _signal_timeout_ready():
            return []
        try:
            return _run_with_signal_timeout(lambda: solve(expr, var), secs, [])
        except (NotImplementedError, ValueError, TypeError, sp.PolynomialError):
            return []

    try:
        dg = diff(g, var)
        solns = _solve_timed(dg, var)
        for s in solns:
            s = sp.simplify(s)
            if not s.is_real:
                continue
            try:
                if lo.is_number and hi.is_number and s.is_number:
                    inside = float(lo) < float(s) < float(hi)
                else:
                    inside = _ask(sp.Q.positive(s - lo) & sp.Q.positive(hi - s))
            except (TypeError, ValueError, OverflowError, sp.PolynomialError):
                inside = False
            if inside is True:
                pts.append(s)
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        pass
    # Absolute-value zeros create kinks even when the ordinary derivative
    # does not expose a stationary point.
    for sub in sp.preorder_traversal(g):
        if sub.is_Pow and sub.args[1] == sp.Rational(1, 2):
            base = sub.args[0]
            for s in _solve_timed(base, var):
                s = sp.simplify(s)
                if s.is_real:
                    pts.append(s)
        if isinstance(sub, sp.Abs):
            for s in _solve_timed(sub.args[0], var):
                s = sp.simplify(s)
                if s.is_real:
                    pts.append(s)
    seen, result = set(), []
    for p in sorted(pts, key=lambda e: float(e) if e.is_number else 0):
        key = str(sp.simplify(p))
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def _inner_expr_range(
    g: sp.Expr, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr
) -> tuple[sp.Expr, sp.Expr]:
    """
    Return (g_min, g_max) of g over [lo, hi] by evaluating at endpoints and
    critical points.
    """
    domain = sp.Interval(sp.sympify(lo), sp.sympify(hi))
    delegated = funcprops_range(g, var, domain=domain)
    delegated = getattr(delegated, "primary", delegated)
    if isinstance(delegated, sp.Interval):
        return delegated.inf, delegated.sup
    if isinstance(delegated, sp.FiniteSet) and delegated:
        values = tuple(delegated)
        return sp.Min(*values), sp.Max(*values)

    cpts = _find_critical_points(g, var, lo, hi)
    candidates = []
    for pt in [lo, hi] + cpts:
        try:
            val = g.subs(var, pt)
            val = sp.simplify(val)
            if val.is_real or val.is_number:
                candidates.append(val)
        except _STRATEGY_ERRORS:
            pass
    if not candidates:
        return -oo, oo
    return sp.Min(*candidates), sp.Max(*candidates)


def _parse_superellipse_core(
    g: sp.Expr, vars_: list[sp.Symbol]
) -> tuple[sp.Expr, sp.Expr, dict[sp.Symbol, tuple[sp.Expr, sp.Expr]]] | None:
    """Return ``(h, k, term_map)`` for g = h**k with h = Σ a_i x_i**p_i."""
    k = sp.Integer(1)
    h = g
    if g.is_Pow:
        base, exp_ = g.as_base_exp()
        if exp_.free_symbols or not (exp_.is_positive and exp_.is_real):
            return None
        k = sp.sympify(exp_)
        h = base
    if not h.is_Add:
        return None
    vars_set = set(vars_)
    term_map: dict[sp.Symbol, tuple[sp.Expr, sp.Expr]] = {}
    for term in h.args:
        coeff, rest = term.as_coeff_Mul()
        coeff = sp.sympify(coeff)
        active = list(rest.free_symbols & vars_set)
        if len(active) != 1:
            return None
        v = active[0]
        if v in term_map:
            return None
        if not rest.is_Pow or rest.base != v:
            return None
        pwr = sp.sympify(rest.exp)
        if pwr.free_symbols or not (pwr.is_positive and pwr.is_real):
            return None
        if coeff.free_symbols or not (coeff.is_positive and coeff.is_real):
            return None
        term_map[v] = (coeff, pwr)
    if set(term_map) != vars_set:
        return None
    return h, k, term_map


def _split_superellipse(expr: sp.Expr, vars_: list[sp.Symbol]) -> tuple[Callable, sp.Expr] | None:
    """Find a unique superellipse-type inner core inside ``expr``.

    We look for a subexpression of the form ``(Σ a_i x_i**p_i)**k`` and, if it
    occurs uniquely, replace it by a dummy variable to obtain the outer
    univariate function.
    """
    vars_set = set(vars_)
    matches = []
    for sub in sp.preorder_traversal(expr):
        if sub == expr or not (sub.free_symbols & vars_set):
            continue
        if _parse_superellipse_core(sub, vars_) is not None:
            matches.append(sub)
    uniq = []
    for m in matches:
        if not any(other != m and m in sp.preorder_traversal(other) for other in matches):
            uniq.append(m)
    if len(uniq) != 1:
        return None
    core = uniq[0]
    t = Dummy("t_super")
    outer_expr = expr.xreplace({core: t})
    if outer_expr.free_symbols & vars_set:
        return None
    return sp.Lambda(t, outer_expr), core


def _validate_ranges(ranges: list[tuple]) -> None:
    """Validate strict inner-first iterated-integral dependencies."""
    variables = [r[0] for r in ranges]
    if any(not isinstance(v, sp.Symbol) for v in variables):
        raise TypeError("integration variables must be SymPy Symbol objects")
    if len(set(variables)) != len(variables):
        raise ValueError("integration variables must be unique")

    for i, (var, lo, hi) in enumerate(ranges):
        lo_s = sp.sympify(lo)
        hi_s = sp.sympify(hi)
        if lo_s.has(sp.nan, sp.zoo) or hi_s.has(sp.nan, sp.zoo):
            raise ValueError(f"invalid bound for {var}: NaN and zoo are not allowed")
        # In inner-first order, bounds may depend on variables integrated later,
        # but never on the current variable or one already integrated earlier.
        prohibited = set(variables[: i + 1])
        bad = (lo_s.free_symbols | hi_s.free_symbols) & prohibited
        if bad:
            names = ", ".join(sorted(str(v) for v in bad))
            raise ValueError(
                f"invalid inner-first range for {var}: bound depends on {names}; "
                "bounds may depend only on variables appearing in later ranges"
            )


def _range_has_singularity(expr, var, lo, hi) -> bool:
    """Conservatively detect singularities that make parity shortcuts unsafe."""
    expr = sp.sympify(expr)
    try:
        singular = sp.singularities(expr, var)
    except (NotImplementedError, ValueError, TypeError):
        # If a denominator depends on the variable and singularity analysis is
        # unavailable, decline the shortcut rather than risk a false zero.
        _, den = sp.fraction(sp.together(expr))
        return var in den.free_symbols
    if singular is sp.S.EmptySet:
        return False
    if isinstance(singular, sp.FiniteSet):
        interval = sp.Interval(sp.sympify(lo), sp.sympify(hi))
        for point in singular:
            contains = interval.contains(point)
            if contains not in (False, sp.false):
                return True
        return False
    # Infinite/conditional singularity sets are unsafe for a symmetry fast path.
    return singular != sp.S.EmptySet
