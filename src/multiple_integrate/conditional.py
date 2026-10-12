"""Conditional/stratified result lifting for symbolic integrals."""

from __future__ import annotations

import sympy as sp

from .results import IntegralCase, IntegrationStatus, ParameterStratum, UnevaluatedRegionIntegral


def _normalize_condition(condition: object) -> sp.Expr:
    condition = sp.sympify(condition)
    try:
        return sp.simplify_logic(condition)
    except (TypeError, ValueError, NotImplementedError):
        return condition


def _branch_status(value: sp.Expr) -> tuple[IntegrationStatus, bool | None]:
    value = sp.sympify(value)
    if value.has(sp.Integral, sp.Limit, UnevaluatedRegionIntegral):
        return IntegrationStatus.UNKNOWN, None
    if value.has(sp.AccumBounds):
        # Accumulation ranges are not single finite integral values.
        return IntegrationStatus.UNKNOWN, None
    if value.has(sp.nan):
        return IntegrationStatus.UNKNOWN, None
    series = value.atoms(sp.Sum, sp.Product)
    # Infinite index bounds are syntax, not infinite result values.
    visible = value.xreplace({term: sp.Dummy("series_value", finite=True) for term in series})
    if visible in (sp.oo, -sp.oo, sp.zoo) or visible.has(sp.oo, -sp.oo, sp.zoo):
        return IntegrationStatus.DIVERGENT, False
    if series and value.is_finite is not True:
        return IntegrationStatus.UNKNOWN, None
    return IntegrationStatus.EXACT, True


def complete_cases(cases: tuple[IntegralCase, ...], fallback: sp.Expr) -> tuple[IntegralCase, ...]:
    """Retain an unresolved complement when theorem cases do not cover all parameters."""
    covered = sp.Or(*(case.condition for case in cases))
    remaining = _normalize_condition(sp.Not(covered))
    if remaining is sp.false:
        return cases
    return cases + (IntegralCase(fallback, remaining, IntegrationStatus.UNKNOWN, None),)


def piecewise_cases(value: sp.Expr) -> tuple[IntegralCase, ...]:
    """Convert a Piecewise integral answer to disjoint explicit cases."""
    if not isinstance(value, sp.Piecewise):
        return ()
    previous = sp.false
    cases: list[IntegralCase] = []
    for branch_value, raw_condition in value.args:
        cond = sp.And(sp.Not(previous), raw_condition)
        cond = _normalize_condition(cond)
        if cond is sp.false:
            previous = sp.Or(previous, raw_condition)
            continue
        status, convergent = _branch_status(branch_value)
        cases.append(IntegralCase(sp.sympify(branch_value), cond, status, convergent))
        previous = sp.Or(previous, raw_condition)
    return complete_cases(tuple(cases), sp.nan)


def conditional_status(
    value: sp.Expr,
) -> tuple[IntegrationStatus, tuple[IntegralCase, ...], bool | None, bool | None]:
    """Classify a flat SymPy result without changing its mathematical content."""
    value = sp.sympify(value)
    if not isinstance(value, sp.Piecewise) and value.has(sp.Piecewise):
        value = sp.piecewise_fold(value)
    cases = piecewise_cases(value)
    if cases:
        has_exact = any(case.status is IntegrationStatus.EXACT for case in cases)
        has_unresolved = any(
            case.status in (IntegrationStatus.UNKNOWN, IntegrationStatus.UNEVALUATED)
            for case in cases
        )
        complete = not has_unresolved
        all_divergent = all(case.status is IntegrationStatus.DIVERGENT for case in cases)
        convergent = (
            False
            if all_divergent
            else (True if all(case.convergent is True for case in cases) else None)
        )
        return (
            IntegrationStatus.DIVERGENT
            if all_divergent
            else (IntegrationStatus.CONDITIONAL if has_exact else IntegrationStatus.UNKNOWN),
            cases,
            convergent,
            complete,
        )
    status, convergent = _branch_status(sp.sympify(value))
    return status, (), convergent, status is not IntegrationStatus.UNKNOWN


def parameter_strata(
    cases: tuple[IntegralCase, ...],
    *,
    integration_variables: tuple[sp.Symbol, ...] = (),
) -> tuple[ParameterStratum, ...]:
    """Lift conditional cases to explicit mutually exclusive parameter cells."""
    integration_set = set(integration_variables)
    strata: list[ParameterStratum] = []
    for case in cases:
        parameters = tuple(
            sorted(
                (case.condition.free_symbols | case.value.free_symbols) - integration_set,
                key=sp.default_sort_key,
            )
        )
        strata.append(
            ParameterStratum(
                condition=case.condition,
                value=case.value,
                status=case.status,
                convergent=case.convergent,
                parameters=parameters,
            )
        )
    return tuple(strata)
