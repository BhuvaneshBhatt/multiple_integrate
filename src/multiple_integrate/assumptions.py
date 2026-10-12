"""Normalization and evaluation helpers for symbolic assumptions."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass

import sympy as sp


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


def _scoped_ask(predicate, facts=sp.true):
    """Use explicit facts; isolate recursive handlers from temporary global facts."""
    from sympy.assumptions import global_assumptions

    if global_assumptions:
        replacements = {
            v: sp.Dummy("integration_scope", **v.assumptions0)
            for v in predicate.free_symbols | facts.free_symbols
        }
        predicate, facts = predicate.xreplace(replacements), facts.xreplace(replacements)
    return sp.ask(predicate, facts, context=set())


def _normalize_assumptions(assumptions) -> _AssumptionContext:
    """Normalize one condition or an iterable of conditions into a stable context."""
    if assumptions is None or assumptions == ():
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
                _scoped_ask(predicate, context.predicate_expr)
        except ValueError as exc:
            raise ValueError(f"inconsistent assumptions: {sp.And(*original)}") from exc
    return context


def _ask(predicate: sp.Basic, assumptions: _AssumptionContext | None = None):
    """Call ``sympy.ask`` using MultipleIntegrate's normalized assumption context."""
    context = assumptions or _CURRENT_ASSUMPTIONS.get()
    if isinstance(predicate, sp.And):
        answers = [_ask(part, context) for part in predicate.args]
        return False if False in answers else True if all(a is True for a in answers) else None
    answer = _scoped_ask(predicate, context.predicate_expr)
    if (
        answer is True
        and context.predicates
        and isinstance(predicate, sp.AppliedPredicate)
        and predicate.function in (sp.Q.positive, sp.Q.negative)
    ):
        from .cell_proofs import nonnegative

        argument = predicate.arguments[0]
        if predicate.function == sp.Q.negative:
            argument = -argument
        if not nonnegative(argument, context, strict=True):
            return None
    if (
        answer is True
        and isinstance(predicate, sp.AppliedPredicate)
        and predicate.function == sp.Q.finite
    ):
        from .cell_proofs import finite_parameter_domain

        if not finite_parameter_domain(predicate.arguments[0], context):
            return None
    return answer


def _refine_with_assumptions(expr: sp.Expr, assumptions: _AssumptionContext | None = None):
    context = assumptions or _CURRENT_ASSUMPTIONS.get()
    if not context.predicates:
        return expr
    try:
        from sympy.assumptions import global_assumptions

        if global_assumptions:
            facts = context.predicate_expr
            replacements = {
                v: sp.Dummy("refinement_scope", **v.assumptions0)
                for v in expr.free_symbols | facts.free_symbols
            }
            refined = sp.refine(expr.xreplace(replacements), facts.xreplace(replacements))
            return refined.xreplace({value: key for key, value in replacements.items()})
        return sp.refine(expr, context.predicate_expr)
    except (TypeError, ValueError, NotImplementedError):
        return expr


def _integrate(expr: sp.Expr, *limits, **opts):
    """SymPy integration followed by refinement under normalized assumptions."""
    context = opts.pop("_mi_assumptions", None) or _CURRENT_ASSUMPTIONS.get()
    from .isolated_queries import isolated_query
    from .policy import _CURRENT_POLICY

    original_expr, original_limits = expr, limits
    orientation = sp.S.One
    # Rebind a singular right endpoint to positive distance so the backend
    # evaluates its left-hand limit, including oscillatory primitives.
    if len(limits) == 1 and isinstance(limits[0], (tuple, list, sp.Tuple)) and len(limits[0]) == 3:
        variable, lower, upper = map(sp.sympify, limits[0])
        if (upper - lower).is_negative is True:
            lower, upper = upper, lower
            limits = ((variable, lower, upper),)
            orientation = -sp.S.One
        if upper.is_finite is True and (upper - lower).is_positive is True:
            endpoint = sp.sympify(expr).subs(variable, upper)
            if endpoint.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                distance = sp.Dummy("endpoint_distance", positive=True)
                expr = sp.sympify(expr).subs(variable, upper - distance)
                limits = ((distance, sp.S.Zero, upper - lower),)
            elif lower.is_finite is True and lower != 0:
                endpoint = sp.sympify(expr).subs(variable, lower)
                if endpoint.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                    distance = sp.Dummy("endpoint_distance", positive=True)
                    expr = sp.sympify(expr).subs(variable, lower + distance)
                    limits = ((distance, sp.S.Zero, upper - lower),)

    seconds = _CURRENT_POLICY.get().backend_timeout_seconds
    try:
        result = (
            sp.integrate(expr, *limits, **opts)
            if seconds is None
            else isolated_query("sympy", "integrate", (expr, *limits), opts, seconds)
        )
    except RecursionError:
        return sp.Integral(original_expr, *original_limits)
    if result is None or (limits != original_limits and result.has(sp.Integral)):
        return sp.Integral(original_expr, *original_limits)
    return _refine_with_assumptions(orientation * result, context)


_EMPTY_ASSUMPTIONS = _AssumptionContext((), ())
_CURRENT_ASSUMPTIONS: contextvars.ContextVar[_AssumptionContext] = contextvars.ContextVar(
    "multiple_integrate_assumptions", default=_EMPTY_ASSUMPTIONS
)
