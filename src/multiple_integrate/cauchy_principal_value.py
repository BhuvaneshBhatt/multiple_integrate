"""One-dimensional and iterated Cauchy principal values."""

from __future__ import annotations

import sympy as sp

from .assumptions import _integrate, _normalize_assumptions, _refine_with_assumptions
from .planner import record_coverage, record_strategy
from .principal_value_prescriptions import IteratedPrincipalValue
from .rational_cutoff import certified_order, real_rational_cutoff, requires_real_rational_cutoff
from .recognition import _fast_simplify
from .results import UnresolvedIntegration


def _rational_poles(expr: sp.Expr, var: sp.Symbol) -> tuple[sp.Expr, ...]:
    try:
        _num, den = sp.fraction(sp.cancel(expr))
        if den == 1 or not den.has(var):
            return ()
        roots = sp.solve(sp.Eq(den, 0), var)
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        return ()
    return tuple(sp.sympify(root) for root in roots if not sp.sympify(root).has(var))


def _inside_interval(point, lower, upper, assumptions) -> bool:
    predicate = getattr(assumptions, "predicate_expr", sp.true)
    left = True if lower is -sp.oo else sp.ask(sp.Q.positive(point - lower), predicate)
    right = True if upper is sp.oo else sp.ask(sp.Q.positive(upper - point), predicate)
    return left is True and right is True


def _symmetric_cutoff_pv(expr, var, lower, upper, points, *, assumptions=None):
    eps = sp.Dummy("pv_epsilon", positive=True, real=True)
    ordered = certified_order(points, assumptions=assumptions)
    canonical = real_rational_cutoff(expr, var, lower, upper, ordered, eps, assumptions=assumptions)
    if canonical is not None:
        return sp.limit(canonical, eps, 0, dir="+")
    if (
        lower not in (-sp.oo, sp.oo)
        and upper not in (-sp.oo, sp.oo)
        and requires_real_rational_cutoff(expr, var, assumptions=assumptions)
    ):
        raise UnresolvedIntegration(
            "a canonical real rational cutoff primitive could not be certified"
        )
    total = sp.Integer(0)
    start = lower
    for point in ordered:
        total += sp.integrate(expr, (var, start, point - eps))
        start = point + eps
    total += sp.integrate(expr, (var, start, upper))
    return sp.limit(total, eps, 0, dir="+")


def one_dimensional_pv(expr, range_, *, assumptions=None) -> sp.Expr:
    """Evaluate one explicit 1-D Cauchy principal value.

    Rational interior poles are handled with one shared symmetric cutoff, which
    avoids complex-log branch artifacts for parameter-dependent real poles.
    Other expressions fall back to SymPy's Cauchy-PV evaluator.
    """
    var, lower, upper = range_
    expr = sp.sympify(expr)
    points = tuple(
        p
        for p in _rational_poles(expr, var)
        if p not in (-sp.oo, sp.oo) and _inside_interval(p, lower, upper, assumptions)
    )
    if points:
        value = _symmetric_cutoff_pv(expr, var, lower, upper, points, assumptions=assumptions)
    else:
        value = sp.Integral(expr, (var, lower, upper)).principal_value()
    value = _refine_with_assumptions(value, assumptions)
    if var.is_real is True and expr.is_rational_function(var):
        numerator, denominator = sp.fraction(sp.cancel(expr))
        coefficients = sp.Poly(numerator, var).all_coeffs() + sp.Poly(denominator, var).all_coeffs()
        if all(_refine_with_assumptions(c, assumptions).is_real is True for c in coefficients):
            # Only certified real rational functions permit removal of log
            # branch artifacts. Unknown complex coefficients must be retained.
            value = _refine_with_assumptions(sp.re(value), assumptions)
    record_coverage(
        "cutoff_domain",
        True,
        "principal_value",
        basis="all interval segments or whole-interval SymPy PV",
        variable=var,
        lower=lower,
        upper=upper,
    )
    record_strategy("principal_value", reason=f"Cauchy PV in {var}")
    return _fast_simplify(value)


def iterated_pv(expr, ranges, prescription: IteratedPrincipalValue, *, assumptions=None) -> sp.Expr:
    """Evaluate nested one-dimensional PVs in the package's inner-first order."""
    selected = set(prescription.variables)
    all_variables = not selected
    known = {sp.sympify(r[0]) for r in ranges}
    missing = selected - known
    if missing:
        names = ", ".join(map(str, sorted(missing, key=sp.default_sort_key)))
        raise ValueError(f"iterated PV variables are not integration variables: {names}")
    value = sp.sympify(expr)
    outer_constraints = []
    for _var, lo, hi in ranges[1:]:
        outer_constraints.extend((_var > lo, _var < hi))
    base_original = tuple(getattr(assumptions, "original", ()))
    for index, range_ in enumerate(ranges):
        var = sp.sympify(range_[0])
        local_conditions = base_original + tuple(outer_constraints[index * 2 :])
        local_assumptions = _normalize_assumptions(local_conditions)
        if all_variables or var in selected:
            value = one_dimensional_pv(value, range_, assumptions=local_assumptions)
        else:
            value = _integrate(value, range_, _mi_assumptions=local_assumptions)
    record_strategy("iterated_principal_value", reason="nested one-dimensional Cauchy PV")
    return _fast_simplify(value)


__all__ = ["one_dimensional_pv", "iterated_pv"]
