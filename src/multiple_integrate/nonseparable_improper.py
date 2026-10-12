"""Absolute full-space moments after proved global polynomial shear charts."""

from functools import lru_cache

import sympy as sp

from .assumptions import _ask, _normalize_assumptions
from .fast_rules import _polynomial
from .planner import record_convergence, record_coverage, record_strategy
from .policy import _normalize_policy, _with_policy
from .quadratic_radial import _moment


def _chart(candidate, variables, context):
    """Recognize a triangular sum of positive even powers; determinant is one."""
    q = sp.expand(candidate)
    forward = {}
    powers = []
    coefficients = []
    for i in reversed(range(len(variables))):
        x = variables[i]
        try:
            p = sp.Poly(q, x)
        except sp.PolynomialError:
            return None
        m = p.degree()
        d = p.LC()
        if (
            m < 2
            or m % 2
            or d.has(*variables)
            or _ask(sp.Q.positive(d) & sp.Q.finite(d), context) is not True
        ):
            return None
        center = sp.cancel(-p.nth(m - 1) / (m * d))
        if i:
            center_poly = _polynomial(center, variables[:i])
            if center_poly is None:
                return None
            center_coefficients = center_poly.coeffs()
        else:
            if center.has(*variables):
                return None
            center_coefficients = (center,)
        if not all(
            _ask(sp.Q.real(c) & sp.Q.finite(c), context) is True for c in center_coefficients
        ):
            return None
        if center.free_symbols & set(variables[i:]):
            return None
        u = sp.Dummy("improper_coordinate", real=True)
        transformed = sp.expand(q.subs(x, u + center))
        residual = sp.expand(transformed - d * u**m)
        if residual.has(u):
            return None
        forward[x] = (u, center)
        powers.insert(0, m)
        coefficients.insert(0, d)
        q = residual
    if q.has(*variables) or _ask(sp.Q.real(q) & sp.Q.finite(q), context) is not True:
        return None
    mapping = {}
    for x in variables:
        u, center = forward[x]
        mapping[x] = u + center.subs(mapping, simultaneous=True)
    return tuple(forward[x][0] for x in variables), mapping, tuple(powers), tuple(coefficients), q


@_with_policy
def improper_data(expr, ranges, assumptions=None, policy=None):
    """Prove absolute origin/tail thresholds before allowing angular cancellation."""
    if len(ranges) < 2 or not all(lo == -sp.oo and hi == sp.oo for _, lo, hi in ranges):
        return None
    variables = tuple(r[0] for r in ranges)
    context = _normalize_assumptions(assumptions)
    budget = _normalize_policy(policy)
    if not budget.allow_expensive_exact and len(variables) > budget.semialg.max_variables:
        return None
    for factor in sp.Mul.make_args(sp.sympify(expr)):
        candidate = (
            -factor.args[0]
            if factor.func == sp.exp
            else factor.base
            if factor.is_Pow and factor.exp.has(*variables) is False
            else None
        )
        if candidate is None or _polynomial(candidate, variables) is None:
            continue
        chart = _chart(candidate, variables, context)
        if chart is None:
            chart = _chart(candidate, tuple(reversed(variables)), context)
        if chart is None:
            continue
        local, mapping, powers, coefficients, shift = chart
        q = sp.expand(candidate - shift)
        polynomial = sp.S.One
        scale = sp.S.One
        gaussian = None
        power = None
        valid = True
        qpoly = sp.Poly(q, *variables)
        monomial = qpoly.monoms()[0]
        for term in sp.Mul.make_args(sp.sympify(expr)):
            if not term.has(*variables):
                if _ask(sp.Q.finite(term), context) is not True:
                    valid = False
                    break
                scale *= term
                continue
            if _polynomial(term, variables) is not None:
                polynomial *= term
                continue
            arg = term.args[0] if term.func == sp.exp else term.base if term.is_Pow else None
            if arg is None or _polynomial(arg, variables) is None:
                valid = False
                break
            ratio = sp.cancel(
                sp.Poly(arg, *variables).coeff_monomial(monomial) / qpoly.coeff_monomial(monomial)
            )
            offset = sp.simplify(arg - ratio * q)
            if (
                offset.has(*variables)
                or _ask(sp.Q.real(offset) & sp.Q.finite(offset), context) is not True
            ):
                valid = False
                break
            if term.func == sp.exp:
                if (
                    gaussian is not None
                    or _ask(sp.Q.negative(ratio) & sp.Q.finite(ratio), context) is not True
                ):
                    valid = False
                    break
                gaussian = -ratio
                scale *= sp.exp(offset)
            else:
                if (
                    power is not None
                    or term.exp.has(*variables)
                    or _ask(
                        sp.Q.positive(ratio)
                        & sp.Q.finite(ratio)
                        & sp.Q.real(term.exp)
                        & sp.Q.finite(term.exp),
                        context,
                    )
                    is not True
                ):
                    valid = False
                    break
                b = sp.cancel(offset / ratio)
                if _ask(sp.Q.zero(b), context) is True:
                    b = sp.S.Zero
                if b != 0 and _ask(sp.Q.positive(b), context) is not True:
                    valid = False
                    break
                if (
                    gaussian is not None
                    and b != 0
                    and _ask(sp.Q.negative(term.exp), context) is not True
                ):
                    valid = False
                    break
                power = (b, term.exp)
                scale *= ratio**term.exp
        if not valid or (gaussian is None and power is None):
            continue
        if (
            gaussian is not None
            and power is not None
            and power[0] != 0
            and _ask(sp.Q.negative(power[1]), context) is not True
        ):
            continue
        translated = _polynomial(polynomial.subs(mapping, simultaneous=True), local)
        if translated is None or not all(
            _ask(sp.Q.finite(c), context) is True for c in translated.coeffs()
        ):
            continue
        b, exponent = power if power is not None else (sp.S.Zero, sp.S.Zero)
        for monomial, c in translated.terms():
            if c == 0:
                continue
            alpha = sum(
                (sp.Rational(p + 1, m) for p, m in zip(monomial, powers, strict=True)), sp.S.Zero
            )
            if (b == 0 and _ask(sp.Q.positive(alpha + exponent), context) is not True) or (
                gaussian is None and _ask(sp.Q.negative(alpha + exponent), context) is not True
            ):
                valid = False
                break
        if valid:
            return translated, scale, gaussian, power, powers, coefficients, mapping
    return None


def try_nonseparable_improper(expr, ranges, assumptions=None):
    data = improper_data(expr, ranges, assumptions)
    if data is None:
        return None
    poly, scale, gaussian, power, powers, coefficients, mapping = data

    @lru_cache(None)
    def radial(alpha):
        return 2 * _moment(alpha, gaussian, power, sp.S.Zero, sp.oo)

    total = sp.S.Zero
    for monomial, c in poly.terms():
        if any(p % 2 for p in monomial):
            continue
        alpha = tuple(sp.Rational(p + 1, m) for p, m in zip(monomial, powers, strict=True))
        weight = sum(alpha, sp.S.Zero)
        angular = sp.prod(
            2 * sp.gamma(a) / (m * d**a)
            for a, m, d in zip(alpha, powers, coefficients, strict=True)
        ) / sp.gamma(weight)
        total += c * angular * radial(weight)
    record_strategy("nonseparable_improper")
    record_coverage(
        "integration_domain",
        True,
        "nonseparable_improper",
        basis="global triangular polynomial bijection and complete weighted orthant decomposition",
        forward_map=tuple(mapping.values()),
    )
    record_coverage(
        "absolute_convergence",
        True,
        "nonseparable_improper",
        basis="every transformed polynomial monomial satisfies weighted origin and tail thresholds",
    )
    for kind in ("absolute", "ordinary"):
        record_convergence(kind, True, "nonseparable_improper")
    return sp.simplify(scale * total)
