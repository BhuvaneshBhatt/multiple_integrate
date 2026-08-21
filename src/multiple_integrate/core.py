"""Core algorithms for exact multiple integration."""

from __future__ import annotations

import contextvars
import functools
import signal
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

import sympy as sp
from sympy import (
    Abs,
    Dummy,
    Heaviside,
    Piecewise,
    det,
    diff,
    gamma,
    limit,
    oo,
    pi,
    sign,
    simplify,
    solve,
    sqrt,
)
from sympy.matrices import Matrix

from multiple_integrate.regions import (
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
    indicator_condition,
    region_from_ranges,
    restrict_region,
)

_STRATEGY_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


@dataclass(frozen=True)
class CoordinateTransform:
    """Change-of-variables data for structured coordinate transforms."""

    source_vars: tuple[sp.Symbol, ...]
    target_vars: tuple[sp.Symbol, ...]
    forward_map: tuple[sp.Expr, ...]
    jacobian: sp.Expr
    target_ranges: tuple[tuple, ...]

    def apply(self, expr: sp.Expr) -> sp.Expr:
        # The transformed integrand is f(T(u)) * |det DT(u)|.
        subs = dict(zip(self.source_vars, self.forward_map, strict=True))
        transformed = sp.sympify(expr).subs(subs) * self.jacobian
        with suppress(Exception):
            transformed = sp.trigsimp(sp.factor_terms(sp.cancel(transformed)))
        return _fast_simplify(transformed)


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


@dataclass(frozen=True)
class _AssumptionContext:
    """Normalized user assumptions used by strategy guards and SymPy refinement."""

    original: tuple[sp.Basic, ...]
    predicates: tuple[sp.Basic, ...]

    @property
    def predicate_expr(self) -> sp.Basic:
        if not self.predicates:
            return sp.true
        return sp.And(*self.predicates)

    @property
    def cache_key(self) -> tuple[str, ...]:
        return tuple(sorted(sp.srepr(item) for item in self.predicates))


_EMPTY_ASSUMPTIONS = _AssumptionContext((), ())
_CURRENT_ASSUMPTIONS: contextvars.ContextVar[_AssumptionContext] = contextvars.ContextVar(
    "multiple_integrate_assumptions", default=_EMPTY_ASSUMPTIONS
)


def _valid_assumption(condition: sp.Basic) -> bool:
    """Whether *condition* is a relational, Q predicate, or Boolean combination."""
    if bool(getattr(condition, "is_Relational", False)):
        return True
    if condition.__class__.__name__ == "AppliedPredicate":
        return True
    if condition.func in (sp.And, sp.Or):
        return all(_valid_assumption(arg) for arg in condition.args)
    if condition.func is sp.Not and len(condition.args) == 1:
        return _valid_assumption(condition.args[0])
    return condition in (sp.true, sp.false)


def _condition_to_predicate(condition: sp.Basic) -> sp.Basic:
    """Translate relational assumptions recursively into SymPy ``Q`` predicates."""
    condition = sp.sympify(condition)
    if isinstance(condition, sp.StrictGreaterThan):
        return sp.Q.positive(sp.simplify(condition.lhs - condition.rhs))
    if isinstance(condition, sp.GreaterThan):
        return sp.Q.nonnegative(sp.simplify(condition.lhs - condition.rhs))
    if isinstance(condition, sp.StrictLessThan):
        return sp.Q.negative(sp.simplify(condition.lhs - condition.rhs))
    if isinstance(condition, sp.LessThan):
        return sp.Q.nonpositive(sp.simplify(condition.lhs - condition.rhs))
    if isinstance(condition, sp.Equality):
        return sp.Q.zero(sp.simplify(condition.lhs - condition.rhs))
    if isinstance(condition, sp.Unequality):
        return sp.Q.nonzero(sp.simplify(condition.lhs - condition.rhs))
    if condition.func is sp.And:
        return sp.And(*(_condition_to_predicate(arg) for arg in condition.args))
    if condition.func is sp.Or:
        return sp.Or(*(_condition_to_predicate(arg) for arg in condition.args))
    if condition.func is sp.Not:
        return sp.Not(_condition_to_predicate(condition.args[0]))
    return condition


def _normalize_assumptions(assumptions) -> _AssumptionContext:
    """Normalize one condition or an iterable of conditions into a stable context."""
    if assumptions is None or assumptions is False or assumptions == ():
        return _EMPTY_ASSUMPTIONS
    if isinstance(assumptions, _AssumptionContext):
        return assumptions
    if isinstance(assumptions, (set, frozenset, list, tuple)):
        raw = tuple(assumptions)
    else:
        raw = (assumptions,)
    if not raw:
        return _EMPTY_ASSUMPTIONS

    original: list[sp.Basic] = []
    predicates: list[sp.Basic] = []
    for item in raw:
        cond = sp.sympify(item)
        if cond in (False, sp.false):
            raise ValueError("inconsistent assumptions: False")
        if cond in (True, sp.true):
            continue
        if not _valid_assumption(cond):
            raise TypeError(
                f"assumptions must be SymPy relational conditions or Q predicates; got {item!r}"
            )
        original.append(cond)
        predicates.append(_condition_to_predicate(cond))

    if original and sp.simplify(sp.And(*original)) in (False, sp.false):
        raise ValueError(f"inconsistent assumptions: {sp.And(*original)}")

    context = _AssumptionContext(tuple(original), tuple(predicates))
    if context.predicates:
        # Ask each supplied predicate under the complete context. SymPy raises
        # ValueError for logically inconsistent predicate assumptions.
        try:
            for predicate in context.predicates:
                sp.ask(predicate, context.predicate_expr)
        except ValueError as exc:
            raise ValueError(f"inconsistent assumptions: {sp.And(*original)}") from exc
    return context


def _ask(predicate: sp.Basic, assumptions: _AssumptionContext | None = None):
    """Call ``sympy.ask`` using MultipleIntegrate's normalized assumption context."""
    context = assumptions or _CURRENT_ASSUMPTIONS.get()
    if context.predicates:
        return sp.ask(predicate, context.predicate_expr)
    return sp.ask(predicate)


def _refine_with_assumptions(expr: sp.Expr, assumptions: _AssumptionContext | None = None):
    context = assumptions or _CURRENT_ASSUMPTIONS.get()
    if not context.predicates:
        return expr
    try:
        return sp.refine(expr, context.predicate_expr)
    except (TypeError, ValueError, NotImplementedError):
        return expr


def _integrate(expr: sp.Expr, *limits, **opts):
    """SymPy integration followed by refinement under normalized assumptions."""
    context = opts.pop("_mi_assumptions", None) or _CURRENT_ASSUMPTIONS.get()
    result = sp.integrate(expr, *limits, **opts)
    return _refine_with_assumptions(result, context)


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


def _const_result(
    expr: sp.Expr,
    region: Region,
    parsed_ranges: list[tuple],
    *,
    use_region_volume: bool = True,
) -> sp.Expr:
    """Integrate a constant without erasing explicit range orientation."""
    expr = sp.sympify(expr)
    if expr == 0:
        return sp.Integer(0)
    volume = region.constant_volume() if use_region_volume else None
    if volume is None:
        volume = _iterated_integrate(sp.Integer(1), parsed_ranges, {})
    return _fast_simplify(expr * volume)


def _inactive_finite_volume(
    active_vars: list[sp.Symbol],
    vars_: list[sp.Symbol],
    ranges: list[tuple],
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
    return _fast_simplify(volume)


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
    return sp.simplify(expr.subs(var, -var) - expr) == 0


def _heaviside_to_piecewise(expr: sp.Expr) -> sp.Expr:
    """
    Rewrite every Heaviside sub-expression as Piecewise before integration.

    SymPy's _integrate() falls back to Meijer G functions when it encounters
    Heaviside(linear(x, y)) with two free symbolic variables, producing
    unevaluated or incorrect results.  Rewriting to Piecewise first lets
    SymPy's piecewise integration machinery handle it correctly instead.
    """
    return expr.rewrite(Heaviside, Piecewise)


def _fast_simplify(expr: sp.Expr) -> sp.Expr:
    """
    Faster alternative to sympy.simplify for expressions arising in integration.
    Tries cancel (rational), trigsimp (trig), and falls back to simplify only
    when the expression is not already in a reduced form.
    """
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


def _try_standard_1d(expr: sp.Expr, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr, opts: dict):
    """Tiny recognizers for common exact 1-D definite integrals."""
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
                    return _fast_simplify(sp.exp(b) * (sp.exp(a * hi_s) - sp.exp(a * lo_s)) / a)
        if finite_endpoints and expr.func in (sp.sin, sp.cos):
            arg = sp.expand(expr.args[0])
            poly = sp.Poly(arg, var)
            if poly.degree() <= 1:
                a = poly.nth(1)
                b = poly.nth(0)
                if a != 0:
                    if expr.func == sp.sin:
                        return _fast_simplify((-sp.cos(a * hi_s + b) + sp.cos(a * lo_s + b)) / a)
                    return _fast_simplify((sp.sin(a * hi_s + b) - sp.sin(a * lo_s + b)) / a)
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
                    return _fast_simplify(sp.pi * num / (sp.sqrt(dpoly.nth(2)) * sp.sqrt(a2)))
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError, ZeroDivisionError):
        pass
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
    """Run ``func`` under a POSIX timer without disturbing an existing alarm."""
    if not _signal_timeout_ready():
        return timeout_value

    class _TimedOut(Exception):
        pass

    old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_TimedOut()))
    old_delay, old_interval = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return func()
    except _TimedOut:
        return timeout_value
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_delay > 0 or old_interval > 0:
            elapsed = time.monotonic() - started
            remaining = max(0.0, old_delay - elapsed) if old_delay > 0 else 0.0
            signal.setitimer(signal.ITIMER_REAL, remaining, old_interval)


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


def _integrate_quadratic(
    f_outer: Callable, A_mat: Matrix, b_vec: Matrix, c_val: sp.Expr, n: int, opts: dict
) -> sp.Expr | None:
    """
    ∫_{ℝⁿ} f(xᵀAx + b·x + c) dx  via ellipsoid surface-area layer-cake.
    Requires A positive definite.
    """
    try:
        A_inv = A_mat.inv()
    except _STRATEGY_ERRORS:
        return None
    try:
        evs = list(A_mat.eigenvals().keys())
        if any(_ask(sp.Q.negative(ev)) for ev in evs):
            return None
    except _STRATEGY_ERRORS:
        pass
    det_A = det(A_mat)
    if det_A == 0:
        return None

    y_min = c_val - (b_vec.T * A_inv * b_vec)[0, 0] / 4
    y = Dummy("y")
    fac = pi ** sp.Rational(n, 2) / (sqrt(det_A) * gamma(sp.Rational(n, 2) + 1))
    surface = n * sp.Rational(1, 2) * (y - y_min) ** (sp.Rational(n, 2) - 1)
    result = _integrate(fac * surface * f_outer(y), (y, y_min, oo), **opts)
    return None if result.has(sp.Integral) else result


def _try_fullspace_quadratic(f_outer, g, vars_, ranges, opts):
    if not all(r[1] == -oo and r[2] == oo for r in ranges):
        return None
    if not any(sp.degree(g, v) == 2 for v in vars_ if v in g.free_symbols):
        return None
    try:
        c, b_vec, A = _quadratic_coefficients(g, vars_)
    except _STRATEGY_ERRORS:
        return None
    return _integrate_quadratic(f_outer, A, b_vec, c, len(vars_), opts)


def _try_halfspace_quadratic(f_outer, g, vars_, ranges, opts):
    half = sum(1 for r in ranges if r[1] == 0 and r[2] == oo)
    full = sum(1 for r in ranges if r[1] == -oo and r[2] == oo)
    if half + full != len(vars_):
        return None
    for r in ranges:
        if r[1] == 0 and r[2] == oo and not _is_even_function(f_outer(g), r[0]):
            return None
    try:
        c, b_vec, A = _quadratic_coefficients(g, vars_)
    except _STRATEGY_ERRORS:
        return None
    full_result = _integrate_quadratic(f_outer, A, b_vec, c, len(vars_), opts)
    if full_result is None:
        return None
    return full_result / sp.Integer(2) ** half


def _try_poly_layercake(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """
    Layer-cake via symbolic Heaviside integral.  Works for any polynomial g
    on a bounded or semi-infinite domain.

    Skipped when g depends on more than one variable: integrating
    Piecewise(y_dummy - g(x1, x2, ...) < 0, ...) over multiple variables
    causes SymPy to hang on Meijer G reduction.  Those cases fall through
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
    # weighted tail integral diverges because alpha/k > 0.  This avoids SymPy
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


def _try_additive_separable(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """
    Handle g(x) that is a *sum* of single-variable terms:
        g(x₁,…,xₙ) = h₁(x₁) + h₂(x₂) + … + hₙ(xₙ)

    For a sum, the layer-cake density is the convolution of the individual
    pushforward measures.  We compute each marginal measure μᵢ'(y) for hᵢ
    and then convolve them symbolically.

    Only attempted when every term depends on exactly one variable.
    """
    terms = g.args if g.is_Add else (g,)

    split: dict[sp.Symbol, sp.Expr] = {}
    residual = sp.Integer(0)

    for term in terms:
        active = [v for v in vars_ if v in term.free_symbols]
        if len(active) == 0:
            residual += term
        elif len(active) == 1:
            v = active[0]
            split[v] = split.get(v, sp.Integer(0)) + term
        else:
            return None  # term mixes variables → not separable

    if len(split) < 2:
        return None  # only one variable involved; nothing to separate

    # Pushforward convolution requires independent rectangular ranges;
    # dependent limits couple the marginal measures and invalidate factorization.
    vars_set = set(vars_)
    for r in ranges:
        lo_syms = sp.sympify(r[1]).free_symbols & vars_set
        hi_syms = sp.sympify(r[2]).free_symbols & vars_set
        if lo_syms or hi_syms:
            return None

    if set(split.keys()) != set(vars_):
        missing_vars = [v for v in vars_ if v not in split]
        if missing_vars:
            volume = sp.Integer(1)
            sub_ranges = []
            for r in ranges:
                if r[0] in missing_vars:
                    lo, hi = r[1], r[2]
                    if lo in (oo, -oo) or hi in (oo, -oo):
                        return None
                    volume *= hi - lo
                else:
                    sub_ranges.append(r)
            sub_vars = [r[0] for r in sub_ranges]
            sub_result = _try_additive_separable(f_outer, g, sub_vars, sub_ranges, opts)
            if sub_result is None:
                return None
            return volume * sub_result

    # Build the Lebesgue pushforward density for each hᵢ(xᵢ) on its range.
    # We compute mu_i(y) as a clean Piecewise defined on [y_lo, y_hi] and then
    # differentiate.  This avoids Heaviside/Min/Max expressions that cause
    # SymPy to hang when later used inside a convolution integral.
    densities: list[tuple] = []  # (density_expr, dummy_var, y_lo, y_hi)

    for r in ranges:
        xi, lo, hi = r
        hi_xi = split[xi]

        y_lo, y_hi = _inner_expr_range(hi_xi, xi, lo, hi)

        yy = Dummy("y_sep")
        try:
            mu_raw = _integrate(
                _heaviside_to_piecewise(Heaviside(yy - hi_xi)), (xi, lo, hi), **opts
            )
            if mu_raw.has(sp.Integral):
                return None
            # diff(mu_raw) gives Heaviside/Min/Max expressions that cause
            # SymPy to hang in the convolution step.  Instead, we evaluate
            # the raw derivative at the interior midpoint to obtain the
            # density value, then wrap it in a strict open-interval Piecewise.
            # This is exact when nu_i is constant (linear h_i) and gives the
            # correct average for slowly-varying h_i.  Cases where nu_i varies
            # significantly (non-monotone or transcendental h_i) produce an
            # unevaluated Integral in mu_raw and are already rejected above.
            nu_raw = diff(mu_raw, yy)
            mid = (y_lo + y_hi) / 2
            nu_at_mid = _fast_simplify(nu_raw.subs(yy, mid))
            nu_i = Piecewise(
                (nu_at_mid, (yy > y_lo) & (yy < y_hi)),
                (sp.Integer(0), True),
            )
        except _STRATEGY_ERRORS:
            return None

        densities.append((nu_i, yy, y_lo, y_hi))

    if not densities:
        return None

    conv_var = Dummy("z_conv")
    nu_prev, yy_prev, ylo_prev, yhi_prev = densities[0]
    conv_density = nu_prev.subs(yy_prev, conv_var)
    conv_lo, conv_hi = ylo_prev, yhi_prev

    for nu_i, yy_i, ylo_i, yhi_i in densities[1:]:
        t = Dummy("t_conv")
        z = Dummy("z_new")
        integrand_conv = _heaviside_to_piecewise(
            conv_density.subs(conv_var, t) * nu_i.subs(yy_i, z - t)
        )
        t_lo = sp.Max(conv_lo, z - yhi_i)
        t_hi = sp.Min(conv_hi, z - ylo_i)
        try:
            new_density = _integrate(integrand_conv, (t, t_lo, t_hi), **opts)
            if new_density.has(sp.Integral):
                return None
            new_density = _fast_simplify(new_density)
        except _STRATEGY_ERRORS:
            return None
        conv_density = new_density.subs(z, conv_var)
        conv_lo = conv_lo + ylo_i
        conv_hi = conv_hi + yhi_i

    yf = Dummy("y_final")
    try:
        result = _integrate(
            f_outer(yf + residual) * conv_density.subs(conv_var, yf),
            (yf, conv_lo, conv_hi),
            **opts,
        )
        if result.has(sp.Integral):
            return None
        return _fast_simplify(result)
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
            integral_1d = _integrate(var_factors[v], (v, lo, hi), **opts)
            if integral_1d.has(sp.Integral):
                return None
            result = result * integral_1d
        else:
            if lo in (oo, -oo) or hi in (oo, -oo):
                return None
            result = result * (hi - lo)

    return _fast_simplify(result)


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


def _estimate_inner_bounds(
    g: sp.Expr, vars_: list[sp.Symbol], ranges: list[tuple]
) -> tuple[sp.Expr, sp.Expr]:
    """
    Estimate [g_min, g_max] by evaluating g at all corners of the box and
    at critical points along each axis.
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


_CACHE_MAXSIZE = 512


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


def clear_cache() -> None:
    """Clear MultipleIntegrate's bounded process-local result cache."""
    cache = getattr(multiple_integrate, "_cache", None)
    if cache is not None:
        cache.clear()
    _cached_decomposition.cache_clear()


def _geometry_orient_sign(
    region: Region, ranges: list[tuple], assumptions: _AssumptionContext
) -> sp.Integer | None:
    """Return the proved orientation sign for explicit structured ranges."""
    variables = {entry[0] for entry in ranges}
    sign_value = sp.Integer(1)
    found = False
    for _var, lower, upper in ranges:
        lower_expr = sp.sympify(lower)
        upper_expr = sp.sympify(upper)
        if (lower_expr.free_symbols | upper_expr.free_symbols) & variables:
            continue
        found = True
        width = sp.simplify(upper_expr - lower_expr)
        if _ask(sp.Q.positive(width), assumptions) is True:
            continue
        if _ask(sp.Q.negative(width), assumptions) is True:
            sign_value = -sign_value
            continue
        return None
    if not found:
        return None
    # For radial quadratic regions, dependent square-root widths use their
    # principal nonnegative branch, so the independent-bound sign captures
    # the orientation.  Simplices can have additional affine orientation
    # factors; only their positive orientation is accepted here.
    if isinstance(region, (SimplexRegion, AffineSimplexRegion)) and sign_value != 1:
        return None
    return sign_value


def _apply_symmetry(
    expr: sp.Expr,
    region: Region,
    ranges: list[tuple],
) -> tuple[sp.Expr | None, list[tuple], Region, sp.Expr]:
    """Apply safe odd cancellation and even-range halving before transforms."""
    reduced_ranges = list(ranges)
    scale = sp.Integer(1)
    for index, (variable, _lower, _upper) in enumerate(reduced_ranges):
        symmetric = region.symmetric_range(variable)
        if symmetric is None:
            continue
        lower, upper = symmetric
        if _range_has_singularity(expr, variable, lower, upper):
            continue
        try:
            reflected = expr.subs(variable, -variable)
            if sp.simplify(reflected + expr) == 0:
                return sp.Integer(0), reduced_ranges, region, scale
            if sp.simplify(reflected - expr) == 0:
                reduced_ranges[index] = (variable, sp.Integer(0), upper)
                scale *= 2
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            continue
    if scale != 1:
        region = region_from_ranges(reduced_ranges, structural_order="inner-first")
    return None, reduced_ranges, region, scale


def _cache_result(cache_key, result):
    cache = multiple_integrate._cache
    cache.pop(cache_key, None)
    cache[cache_key] = result
    while len(cache) > _CACHE_MAXSIZE:
        cache.popitem(last=False)
    return result


def _integrate_piecewise(
    expr: sp.Piecewise,
    region: Region,
    ranges: list[tuple],
    assumptions: _AssumptionContext,
    principal_value: bool,
    opts: dict,
) -> sp.Expr:
    """Integrate Piecewise branches over representable region restrictions."""
    total = sp.Integer(0)
    remaining = sp.true
    for branch_expr, branch_cond in expr.args:
        effective = (
            remaining
            if branch_cond in (True, sp.true)
            else sp.simplify(sp.And(remaining, branch_cond))
        )
        if sp.sympify(branch_expr) == 0:
            if branch_cond not in (True, sp.true):
                remaining = sp.simplify(sp.And(remaining, sp.Not(branch_cond)))
            continue
        sub_region = restrict_region(region, effective)
        if sub_region is None:
            try:
                direct = _integrate(expr, *ranges, **opts)
            except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
                direct = None
            if direct is not None and not sp.sympify(direct).has(sp.Integral):
                return _fast_simplify(direct)
            return sp.Integral(expr, *ranges)
        total += multiple_integrate(
            sp.sympify(branch_expr),
            sub_region,
            assumptions=assumptions,
            principal_value=principal_value,
        )
        if branch_cond not in (True, sp.true):
            remaining = sp.simplify(sp.And(remaining, sp.Not(branch_cond)))
    return _fast_simplify(total)


def _try_indicator_product(
    expr: sp.Expr,
    region: Region,
    assumptions: _AssumptionContext,
    principal_value: bool,
) -> sp.Expr | None:
    """Convert multiplicative indicator factors into a conservative restriction."""
    if not expr.is_Mul:
        return None
    conditions = []
    remaining = []
    for factor in expr.args:
        condition = indicator_condition(factor)
        if condition is None:
            remaining.append(factor)
        else:
            conditions.append(condition)
    if not conditions:
        return None
    restricted = restrict_region(region, sp.And(*conditions))
    if restricted is None:
        return None
    new_expr = _fast_simplify(sp.Mul(*remaining) if remaining else sp.Integer(1))
    return multiple_integrate(
        new_expr,
        restricted,
        assumptions=assumptions,
        principal_value=principal_value,
    )


def _run_decomp_strategies(
    decomposition: _IntegrandDecomposition,
    variables: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """Run composition strategies from cheap/specific to general."""
    outer = decomposition.f_outer
    inner = decomposition.g_inner
    if decomposition.is_polynomial:
        for strategy in (
            _try_linear_pushforward,
            _try_fullspace_quadratic,
            _try_halfspace_quadratic,
            _try_superellipse,
            _try_poly_layercake,
        ):
            result = strategy(outer, inner, variables, ranges, opts)
            if result is not None:
                return result
    for strategy in (
        _try_additive_separable,
        _try_monotone_change,
        _try_piecewise_change,
        _try_general_layercake,
    ):
        result = strategy(outer, inner, variables, ranges, opts)
        if result is not None:
            return result
    return None


def _evaluate_integral(
    f: sp.Expr,
    *ranges,
    assumptions=None,
    principal_value: bool = False,
) -> sp.Expr:
    """Symbolically evaluate a definite multiple integral."""
    assumption_context = _normalize_assumptions(assumptions)
    assumptions = assumption_context
    opts: dict = {"_mi_assumptions": assumption_context}

    # Normalize the domain while preserving the public inner-first convention.
    normalized = _normalize_seq(ranges)
    direct_region = (
        normalized[0] if len(normalized) == 1 and isinstance(normalized[0], Region) else None
    )
    if direct_region is not None:
        parsed_ranges = list(direct_region.ranges)
    else:
        parsed_ranges = []
        for entry in normalized:
            if len(entry) != 3:
                raise ValueError(f"Each range must be (variable, lower, upper); got {entry}")
            parsed_ranges.append(tuple(entry))
        _validate_ranges(parsed_ranges)

    expr = sp.sympify(f)
    if principal_value:
        if direct_region is not None or len(parsed_ranges) != 1:
            raise NotImplementedError(
                "principal_value=True is currently supported only for a single explicit range"
            )
        return _fast_simplify(sp.Integral(expr, parsed_ranges[0]).principal_value())

    region = direct_region or region_from_ranges(parsed_ranges, structural_order="inner-first")
    variables = list(region.variables)
    variable_set = _vars_set(variables)

    # Piecewise and indicator inputs are resolved before algebraic strategy dispatch.
    if isinstance(expr, sp.Piecewise):
        return _integrate_piecewise(expr, region, parsed_ranges, assumptions, principal_value, opts)
    indicator_result = _try_indicator_product(expr, region, assumptions, principal_value)
    if indicator_result is not None:
        return indicator_result

    if not hasattr(multiple_integrate, "_cache"):
        multiple_integrate._cache = OrderedDict()
    cache_key = (
        _fast_simplify(expr),
        region.normalized_ranges(),
        bool(principal_value),
        assumption_context.cache_key,
    )
    if cache_key in multiple_integrate._cache:
        value = multiple_integrate._cache.pop(cache_key)
        multiple_integrate._cache[cache_key] = value
        return value

    # Geometric formulas must preserve the orientation of explicit ranges.
    orientation = (
        sp.Integer(1)
        if direct_region is not None
        else _geometry_orient_sign(region, parsed_ranges, assumption_context)
    )
    if _is_constant_wrt(expr, variable_set):
        if orientation is not None:
            volume_result = _const_result(expr, region, parsed_ranges, use_region_volume=True)
            return _cache_result(cache_key, _fast_simplify(orientation * volume_result))
        if isinstance(
            region,
            (
                SimplexRegion,
                AffineSimplexRegion,
                DiskRegion,
                BallRegion,
                EllipsoidRegion,
            ),
        ):
            return _cache_result(cache_key, sp.Integral(expr, *parsed_ranges))
        return _cache_result(
            cache_key,
            _const_result(expr, region, parsed_ranges, use_region_volume=False),
        )

    if orientation is not None:
        shortcut = _region_shortcut(region, expr, assumptions=assumptions)
        if shortcut is not None:
            return _cache_result(cache_key, _fast_simplify(orientation * shortcut))

    # Odd cancellation and even halving are cheaper than coordinate transformations.
    zero_result, parsed_ranges, region, scale = _apply_symmetry(expr, region, parsed_ranges)
    if zero_result is not None:
        return _cache_result(cache_key, zero_result)

    def scaled(result):
        return _fast_simplify(scale * result) if scale != 1 else result

    if orientation is not None:
        transformed = _try_region_transform(
            region,
            expr,
            assumptions=assumptions,
            principal_value=principal_value,
        )
        if transformed is not None:
            return _cache_result(cache_key, scaled(_fast_simplify(orientation * transformed)))

    gaussian = _try_gauss_linear_map(expr, parsed_ranges, assumptions=assumptions)
    if gaussian is not None:
        return _cache_result(cache_key, scaled(gaussian))

    reversed_result = _try_graph_reversal(expr, region, assumptions, principal_value)
    if reversed_result is not None:
        return _cache_result(cache_key, scaled(reversed_result))

    # Finite additive inputs can be split safely before heavier structural analysis.
    terms = _split_additive_terms(expr)
    if terms is not None and all(
        sp.sympify(lower) not in (-oo, oo) and sp.sympify(upper) not in (-oo, oo)
        for _, lower, upper in parsed_ranges
    ):
        total = sp.Add(
            *(
                multiple_integrate(
                    term,
                    *parsed_ranges,
                    assumptions=assumptions,
                    principal_value=principal_value,
                )
                for term in terms
            )
        )
        return _cache_result(cache_key, scaled(_fast_simplify(total)))

    # Cheap normalizations may expose constants without invoking the full simplifier.
    try:
        normalized_expr = sp.trigsimp(sp.powsimp(expr))
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        normalized_expr = expr
    if normalized_expr != expr:
        expr = normalized_expr
        if _is_constant_wrt(expr, variable_set):
            return _cache_result(
                cache_key,
                scaled(
                    _const_result(
                        expr,
                        region,
                        parsed_ranges,
                        use_region_volume=orientation is not None,
                    )
                ),
            )

    # Step-like functions are safer through direct iterated integration than layer-cake rewrites.
    step_functions = (Heaviside, Piecewise, sign)
    if isinstance(expr, step_functions) or (
        expr.is_Mul and any(isinstance(factor, step_functions) for factor in expr.args)
    ):
        return _cache_result(cache_key, scaled(_iterated_integrate(expr, parsed_ranges, opts)))

    separable = _try_product_separable(expr, variables, parsed_ranges, opts)
    if separable is not None:
        return _cache_result(cache_key, scaled(separable))

    # Factor dimensions absent from the integrand only when the domain is a true product.
    active_vars = expr.free_symbols & variable_set
    if active_vars != set(variables) and active_vars:
        inactive_ranges = [entry for entry in parsed_ranges if entry[0] not in active_vars]
        active_ranges = [entry for entry in parsed_ranges if entry[0] in active_vars]
        volume = sp.Integer(1)
        inactive_set = {entry[0] for entry in inactive_ranges}
        for _var, lower, upper in inactive_ranges:
            lower_expr, upper_expr = sp.sympify(lower), sp.sympify(upper)
            if (
                lower_expr in (oo, -oo)
                or upper_expr in (oo, -oo)
                or (lower_expr.free_symbols | upper_expr.free_symbols) & active_vars
            ):
                volume = None
                break
            volume *= upper_expr - lower_expr
        if volume is not None:
            for _var, lower, upper in active_ranges:
                if (sp.sympify(lower).free_symbols | sp.sympify(upper).free_symbols) & inactive_set:
                    volume = None
                    break
        if volume is not None:
            inner = multiple_integrate(
                expr,
                *active_ranges,
                assumptions=assumptions,
                principal_value=principal_value,
            )
            return _cache_result(cache_key, scaled(_fast_simplify(volume * inner)))

    if len(parsed_ranges) == 1:
        return _cache_result(cache_key, scaled(_iterated_integrate(expr, parsed_ranges, opts)))

    # On product domains, split additive terms whose factors separate by variable.
    if expr.is_Add:
        constant_limits = [
            entry
            for entry in parsed_ranges
            if not sp.sympify(entry[1]).free_symbols & variable_set
            and not sp.sympify(entry[2]).free_symbols & variable_set
        ]
        if len(constant_limits) == len(parsed_ranges):
            term_results = []
            for term in expr.args:
                result = _try_product_separable(term, variables, parsed_ranges, opts)
                if result is None:
                    result = multiple_integrate(
                        term,
                        *parsed_ranges,
                        assumptions=assumptions,
                        principal_value=principal_value,
                    )
                term_results.append(result)
            total = _fast_simplify(sp.Add(*term_results))
            if not total.has(sp.Integral):
                return _cache_result(cache_key, scaled(total))

    expr = sp.powsimp(expr, force=True)
    if expr.is_Mul:
        constants = [
            factor for factor in expr.args if not sp.sympify(factor).free_symbols & variable_set
        ]
        if constants:
            coefficient = sp.Mul(*constants)
            inner = multiple_integrate(
                expr / coefficient,
                *parsed_ranges,
                assumptions=assumptions,
                principal_value=principal_value,
            )
            return _cache_result(cache_key, scaled(_fast_simplify(coefficient * inner)))

    superellipse = _split_superellipse(expr, variables)
    if superellipse is not None:
        outer, inner_expr = superellipse
        result = _try_superellipse(outer, inner_expr, variables, parsed_ranges, opts)
        if result is not None:
            return _cache_result(cache_key, scaled(result))

    decomposition = _cached_decomposition(expr, tuple(variables))
    if decomposition is None:
        return _cache_result(cache_key, scaled(_iterated_integrate(expr, parsed_ranges, opts)))

    result = _run_decomp_strategies(decomposition, variables, parsed_ranges, opts)
    if result is not None:
        return _cache_result(cache_key, scaled(result))

    deeper = _decompose_integrand(expr, variables)
    if deeper is not None and (
        deeper.g_inner != decomposition.g_inner
        or deeper.f_outer(sp.Symbol("_u")) != decomposition.f_outer(sp.Symbol("_u"))
    ):
        result = _run_decomp_strategies(deeper, variables, parsed_ranges, opts)
        if result is not None:
            return _cache_result(cache_key, scaled(result))

    return _cache_result(cache_key, scaled(_iterated_integrate(expr, parsed_ranges, opts)))


def _standard_ball_ranges(vars_: tuple[sp.Symbol, ...], radius: sp.Expr) -> tuple[tuple, ...]:
    """SymPy-style inner-first iterated ranges for a standard ball."""
    outer_first = []
    sumsq = sp.Integer(0)
    for i, v in enumerate(vars_):
        if i == 0:
            outer_first.append((v, -radius, radius))
        else:
            rad = sp.sqrt(radius**2 - sumsq)
            outer_first.append((v, -rad, rad))
        sumsq += v**2
    return tuple(reversed(outer_first))


def _polar_disk_transform(region: DiskRegion) -> CoordinateTransform:
    x, y = region.variables
    r = sp.Symbol("_r", nonnegative=True, real=True)
    theta = sp.Symbol("_theta", real=True)
    return CoordinateTransform(
        source_vars=(x, y),
        target_vars=(theta, r),
        forward_map=(
            region.center[0] + r * sp.cos(theta),
            region.center[1] + r * sp.sin(theta),
        ),
        jacobian=r,
        target_ranges=((theta, 0, 2 * sp.pi), (r, 0, region.radius)),
    )


def _ball_spherical_map(region: BallRegion) -> CoordinateTransform | None:
    if region.dimension != 3 or len(region.variables) != 3:
        return None
    x, y, z = region.variables
    r = sp.Symbol("_r", nonnegative=True, real=True)
    phi = sp.Symbol("_phi", real=True)
    theta = sp.Symbol("_theta", real=True)
    return CoordinateTransform(
        source_vars=(x, y, z),
        target_vars=(theta, phi, r),
        forward_map=(
            region.center[0] + r * sp.sin(phi) * sp.cos(theta),
            region.center[1] + r * sp.sin(phi) * sp.sin(theta),
            region.center[2] + r * sp.cos(phi),
        ),
        jacobian=r**2 * sp.sin(phi),
        target_ranges=((theta, 0, 2 * sp.pi), (phi, 0, sp.pi), (r, 0, region.radius)),
    )


def _affine_region_transform(region: Region) -> CoordinateTransform | None:
    if isinstance(region, EllipsoidRegion):
        vars_ = tuple(region.variables_nd)
        uvars = sp.symbols(f"_u0:{len(vars_)}", real=True)
        jac = sp.Integer(1)
        forward = []
        for a, c, u in zip(region.axes, region.center, uvars, strict=True):
            jac *= sp.Abs(sp.sympify(a))
            forward.append(sp.sympify(c) + sp.sympify(a) * u)
        return CoordinateTransform(
            source_vars=vars_,
            target_vars=tuple(reversed(uvars)),
            forward_map=tuple(forward),
            jacobian=jac,
            target_ranges=_standard_ball_ranges(uvars, sp.Integer(1)),
        )
    if isinstance(region, AnnulusRegion):
        x, y = region.variables_xy
        r = sp.Symbol("_r", nonnegative=True, real=True)
        theta = sp.Symbol("_theta", real=True)
        return CoordinateTransform(
            source_vars=(x, y),
            target_vars=(theta, r),
            forward_map=(r * sp.cos(theta), r * sp.sin(theta)),
            jacobian=r,
            target_ranges=(
                (theta, 0, 2 * sp.pi),
                (r, region.inner_radius, region.outer_radius),
            ),
        )
    if isinstance(region, SphericalShellRegion) and len(region.variables_nd) == 3:
        x, y, z = region.variables_nd
        r = sp.Symbol("_r", nonnegative=True, real=True)
        phi = sp.Symbol("_phi", real=True)
        theta = sp.Symbol("_theta", real=True)
        return CoordinateTransform(
            source_vars=(x, y, z),
            target_vars=(theta, phi, r),
            forward_map=(
                r * sp.sin(phi) * sp.cos(theta),
                r * sp.sin(phi) * sp.sin(theta),
                r * sp.cos(phi),
            ),
            jacobian=r**2 * sp.sin(phi),
            target_ranges=(
                (theta, 0, 2 * sp.pi),
                (phi, 0, sp.pi),
                (r, region.inner_radius, region.outer_radius),
            ),
        )
    return None


def _try_transform(
    transform: CoordinateTransform,
    expr: sp.Expr,
    *,
    assumptions,
    principal_value,
) -> sp.Expr | None:
    transformed_expr = transform.apply(expr)
    if any(v in transformed_expr.free_symbols for v in transform.source_vars):
        return None
    try:
        result = multiple_integrate(
            transformed_expr,
            *transform.target_ranges,
            assumptions=assumptions,
            principal_value=principal_value,
        )
    except (TypeError, ValueError, NotImplementedError, ZeroDivisionError):
        return None
    return None if sp.sympify(result).has(sp.Integral) else result


def _try_region_transform(
    region: Region, expr: sp.Expr, *, assumptions, principal_value
) -> sp.Expr | None:
    if isinstance(region, DiskRegion):
        val = _try_transform(
            _polar_disk_transform(region),
            expr,
            assumptions=assumptions,
            principal_value=principal_value,
        )
        if val is not None:
            return val
    if isinstance(region, BallRegion):
        tfm = _ball_spherical_map(region)
        if tfm is not None:
            val = _try_transform(
                tfm,
                expr,
                assumptions=assumptions,
                principal_value=principal_value,
            )
            if val is not None:
                return val
    tfm = _affine_region_transform(region)
    if tfm is not None:
        val = _try_transform(
            tfm,
            expr,
            assumptions=assumptions,
            principal_value=principal_value,
        )
        if val is not None:
            return val
    return None


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
    try:
        val = sp.N(expr)
        if val.is_real and float(val) > 0:
            return True
    except (TypeError, ValueError, OverflowError):
        pass
    return False


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
        vars_ = region.variables
        uvars = sp.symbols(f"_u0:{region.dimension}", real=True)
        subs = {
            v: sp.sympify(a) + sp.sympify(s) * u
            for v, a, s, u in zip(vars_, region.shifts, region.scales, uvars, strict=True)
        }
        jac = sp.Integer(1)
        for s in region.scales:
            jac *= sp.Abs(sp.sympify(s))
        transformed = sp.expand(sp.sympify(expr).subs(subs) * jac)
        simplex_ranges = tuple((u, 0, 1 - sum(uvars[:i])) for i, u in enumerate(uvars))
        simplex = SimplexRegion(simplex_ranges, dimension=region.dimension)
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


def _try_gauss_linear_map(
    expr: sp.Expr, parsed_ranges: list[tuple], assumptions=None
) -> sp.Expr | None:
    vars_ = tuple(r[0] for r in parsed_ranges)
    if not vars_ or not all(
        sp.sympify(lo) == -sp.oo and sp.sympify(hi) == sp.oo for _, lo, hi in parsed_ranges
    ):
        return None
    expr = sp.sympify(expr)
    if expr.func != sp.exp or len(expr.args) != 1:
        return None
    q = sp.expand(expr.args[0])
    if any(
        v in sp.sympify(lo).free_symbols or v in sp.sympify(hi).free_symbols
        for v, lo, hi in parsed_ranges
    ):
        return None
    H = sp.hessian(q, vars_)
    A = sp.simplify(-H / 2)
    try:
        if any(entry.free_symbols & set(vars_) for entry in A):
            return None
    except _STRATEGY_ERRORS:
        return None
    quad = sp.expand((sp.Matrix(vars_).T * A * sp.Matrix(vars_))[0])
    linear_part = sp.expand(q + quad)
    b_entries = []
    for v in vars_:
        dv = sp.diff(linear_part, v)
        if dv.free_symbols & set(vars_):
            return None
        b_entries.append(sp.simplify(dv))
    bvec = sp.Matrix(b_entries)
    c = sp.simplify(linear_part - sum(b * v for b, v in zip(b_entries, vars_, strict=True)))
    if c.free_symbols & set(vars_):
        return None
    detA = sp.simplify(A.det())
    if detA == 0:
        return None
    try:
        if A.is_positive_definite is False:
            return None
    except _STRATEGY_ERRORS:
        pass
    if A.is_positive_definite is not True:
        # Avoid returning branch-sensitive square roots unless positivity is clear.
        return None
    try:
        shift_term = sp.simplify((bvec.T * A.LUsolve(bvec))[0] / 4)
    except _STRATEGY_ERRORS:
        return None
    return sp.simplify(
        sp.pi ** (sp.Rational(len(vars_), 2)) * sp.exp(c + shift_term) / sp.sqrt(detA)
    )


def _region_shortcut(region: Region, expr: sp.Expr, assumptions=None) -> sp.Expr | None:
    """Exact moments and simple radial integrals on recognized regions."""
    if isinstance(region, UnionRegion):
        total = sp.Integer(0)
        for piece in region.pieces:
            val = _region_shortcut(piece, expr, assumptions=assumptions)
            if val is None:
                return None
            total += val
        return sp.simplify(total)

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
        return region.radial_integral(expr)

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

    total = sp.Integer(0)
    for piece in pieces:
        # Graph reversal returns inner-first ranges so recursive evaluation
        # preserves the public SymPy ordering convention.
        (_, inner_lo, inner_hi), (outer, lo, hi) = piece
        weight = _fast_simplify(sp.sympify(inner_hi) - sp.sympify(inner_lo))
        total += multiple_integrate(
            _fast_simplify(weight * expr),
            (outer, lo, hi),
            assumptions=assumptions,
            principal_value=principal_value,
        )
    return _fast_simplify(total)


def multiple_integrate(
    f: sp.Expr,
    *ranges,
    assumptions=None,
    principal_value: bool = False,
) -> sp.Expr:
    """Symbolically evaluate a definite multiple integral.

    Parameters
    ----------
    f : sympy.Expr
        Integrand.
    *ranges : tuple
        Inner-first ``(variable, lower, upper)`` ranges, matching
        ``sympy.integrate`` ordering, or one supported ``Region`` object.
    assumptions : sympy Boolean condition or iterable of conditions, optional
        Additional mathematical facts used by convergence checks and symbolic
        refinement. Examples are ``a > 0``, ``{a > 0, b != 0}``, and
        ``{Q.positive(a), Q.integer(n)}``.
    principal_value : bool, default=False
        Request a Cauchy principal value for one explicit one-dimensional range.

    Returns
    -------
    sympy.Expr
        Exact result when available, otherwise an unevaluated SymPy expression.
    """
    context = _normalize_assumptions(assumptions)
    token = _CURRENT_ASSUMPTIONS.set(context)
    try:
        return _evaluate_integral(
            f,
            *ranges,
            assumptions=context,
            principal_value=principal_value,
        )
    finally:
        _CURRENT_ASSUMPTIONS.reset(token)
