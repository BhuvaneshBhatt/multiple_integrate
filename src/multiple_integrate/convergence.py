"""Conditional convergence theorems for parameter-dependent 1-D integrals."""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

from multiple_integrate.assumptions import (
    _AssumptionContext,
    _condition_to_predicate,
    _refine_with_assumptions,
)
from multiple_integrate.results import IntegralCase, IntegrationStatus

Rule = Callable[[sp.Expr, tuple], tuple[IntegralCase, ...]]


def _case(value, condition, *, convergent: bool) -> IntegralCase:
    status = IntegrationStatus.EXACT if convergent else IntegrationStatus.DIVERGENT
    return IntegralCase(value, condition, status, convergent)


def _coefficient_cases(coefficient, cases: tuple[IntegralCase, ...]) -> tuple[IntegralCase, ...]:
    """Separate the identically zero integrand from nonzero convergence tests."""
    if coefficient.is_zero is False:
        return cases
    zero = _case(sp.Integer(0), sp.Eq(coefficient, 0), convergent=True)
    nonzero = tuple(
        IntegralCase(
            case.value, sp.And(sp.Ne(coefficient, 0), case.condition), case.status, case.convergent
        )
        for case in cases
    )
    return (zero,) + nonzero


def _single_range(ranges):
    if len(ranges) != 1 or len(ranges[0]) != 3:
        return None
    var, lo, hi = ranges[0]
    return sp.sympify(var), sp.sympify(lo), sp.sympify(hi)


def _linear_exponential(expr: sp.Expr, ranges) -> tuple[IntegralCase, ...]:
    data = _single_range(ranges)
    if data is None:
        return ()
    var, lo, hi = data
    coefficient, exponential = expr.as_independent(var, as_Add=False)
    if lo != 0 or hi != sp.oo or exponential.func != sp.exp:
        return ()
    try:
        poly = sp.Poly(sp.expand(exponential.args[0]), var)
    except (sp.PolynomialError, TypeError, ValueError):
        return ()
    if poly.degree() != 1:
        return ()
    rate, constant = map(sp.simplify, (poly.nth(1), poly.nth(0)))
    if (
        rate.has(var)
        or rate.is_real is not True
        or not (rate.free_symbols or (coefficient.free_symbols and rate.is_nonnegative is True))
    ):
        return ()
    integral = sp.Integral(expr, (var, lo, hi))
    return _coefficient_cases(
        coefficient,
        (
            _case(-coefficient * sp.exp(constant) / rate, sp.Lt(rate, 0), convergent=True),
            _case(integral, sp.Ge(rate, 0), convergent=False),
        ),
    )


def _power_endpoint(expr: sp.Expr, ranges) -> tuple[IntegralCase, ...]:
    data = _single_range(ranges)
    if data is None:
        return ()
    var, lo, hi = data
    coefficient, power = expr.as_coeff_exponent(var)
    if (
        coefficient.has(var)
        or power.has(var)
        or not (
            power.free_symbols
            or (
                coefficient.free_symbols
                and (
                    (lo == 0 and hi == 1 and (power + 1).is_nonpositive is True)
                    or (lo == 1 and hi == sp.oo and (power + 1).is_nonnegative is True)
                )
            )
        )
        or power.is_real is not True
    ):
        return ()
    integral = sp.Integral(expr, (var, lo, hi))
    if lo == 1 and hi == sp.oo:
        return _coefficient_cases(
            coefficient,
            (
                _case(-coefficient / (power + 1), sp.Lt(power, -1), convergent=True),
                _case(integral, sp.Ge(power, -1), convergent=False),
            ),
        )
    if lo == 0 and hi == 1:
        return _coefficient_cases(
            coefficient,
            (
                _case(coefficient / (power + 1), sp.Gt(power, -1), convergent=True),
                _case(integral, sp.Le(power, -1), convergent=False),
            ),
        )
    return ()


def _gaussian(expr: sp.Expr, ranges) -> tuple[IntegralCase, ...]:
    data = _single_range(ranges)
    if data is None:
        return ()
    var, lo, hi = data
    coefficient, exponential = expr.as_independent(var, as_Add=False)
    if lo != -sp.oo or hi != sp.oo or exponential.func != sp.exp:
        return ()
    try:
        poly = sp.Poly(sp.expand(exponential.args[0]), var)
    except (sp.PolynomialError, TypeError, ValueError):
        return ()
    if poly.degree() != 2:
        return ()
    quad, linear, constant = map(sp.simplify, (poly.nth(2), poly.nth(1), poly.nth(0)))
    if (
        quad.has(var)
        or quad.is_real is not True
        or not (quad.free_symbols or (coefficient.free_symbols and quad.is_nonnegative is True))
    ):
        return ()
    value = sp.sqrt(sp.pi / (-quad)) * sp.exp(constant - linear**2 / (4 * quad))
    integral = sp.Integral(expr, (var, lo, hi))
    return _coefficient_cases(
        coefficient,
        (
            _case(sp.simplify(coefficient * value), sp.Lt(quad, 0), convergent=True),
            _case(integral, sp.Ge(quad, 0), convergent=False),
        ),
    )


def _gamma_tail(expr: sp.Expr, ranges) -> tuple[IntegralCase, ...]:
    """Recognize c*x**p*exp(-a*x) on [0, infinity)."""
    data = _single_range(ranges)
    if data is None:
        return ()
    var, lo, hi = data
    if lo != 0 or hi != sp.oo:
        return ()
    factors = list(sp.Mul.make_args(expr))
    exp_factor = next((item for item in factors if item.func == sp.exp), None)
    if exp_factor is None:
        return ()
    try:
        poly = sp.Poly(sp.expand(exp_factor.args[0]), var)
    except (sp.PolynomialError, TypeError, ValueError):
        return ()
    if poly.degree() != 1 or poly.nth(0) != 0:
        return ()
    rate = sp.simplify(-poly.nth(1))
    rest = sp.simplify(expr / exp_factor)
    coefficient, power = rest.as_coeff_exponent(var)
    if any(item.has(var) for item in (coefficient, rate, power)):
        return ()
    if not (rate.free_symbols or power.free_symbols):
        return ()
    if rate.is_real is not True or power.is_real is not True:
        return ()
    condition = sp.And(sp.Gt(rate, 0), sp.Gt(power, -1))
    value = sp.simplify(coefficient * sp.gamma(power + 1) / rate ** (power + 1))
    return _coefficient_cases(coefficient, (_case(value, condition, convergent=True),))


def _beta_interval(expr: sp.Expr, ranges) -> tuple[IntegralCase, ...]:
    """Recognize c*x**a*(1-x)**b on [0, 1]."""
    data = _single_range(ranges)
    if data is None:
        return ()
    var, lo, hi = data
    if lo != 0 or hi != 1:
        return ()
    powers = expr.as_powers_dict()
    a = sp.simplify(powers.get(var, 0))
    b = sp.simplify(powers.get(1 - var, 0))
    core = sp.simplify(var**a * (1 - var) ** b)
    coefficient = sp.simplify(expr / core)
    if coefficient.has(var) or (not a.free_symbols and not b.free_symbols):
        return ()
    if a.is_real is not True or b.is_real is not True:
        return ()
    condition = sp.And(sp.Gt(a, -1), sp.Gt(b, -1))
    value = sp.simplify(coefficient * sp.beta(a + 1, b + 1))
    return _coefficient_cases(coefficient, (_case(value, condition, convergent=True),))


CONVERGENCE_RULES: tuple[Rule, ...] = (
    _linear_exponential,
    _power_endpoint,
    _gamma_tail,
    _beta_interval,
    _gaussian,
)


def conditional_cases(expr: sp.Expr, ranges) -> tuple[IntegralCase, ...]:
    """Return the first applicable theorem's cases; a complement may remain unresolved."""
    expression = sp.sympify(expr)
    for rule in CONVERGENCE_RULES:
        cases = tuple(c for c in rule(expression, ranges) if c.condition is not sp.false)
        if cases:
            return cases
    return ()


def specialized_value(
    expr: sp.Expr,
    ranges,
    assumptions: _AssumptionContext,
) -> sp.Expr | None:
    """Return a theorem value when supplied assumptions certify one exact case.

    This is a planner fast path for parameter-conditioned improper integrals.
    It avoids generic function-property and symbolic integration work when a
    registered convergence theorem already proves the applicable branch.
    """
    if not assumptions.predicates:
        return None
    for case in conditional_cases(expr, ranges):
        predicate = _condition_to_predicate(sp.sympify(case.condition))
        from multiple_integrate.cell_proofs import native_predicate_true

        if native_predicate_true(predicate, assumptions):
            return _refine_with_assumptions(sp.sympify(case.value), assumptions)
    return None


__all__ = ["CONVERGENCE_RULES", "conditional_cases", "specialized_value"]
