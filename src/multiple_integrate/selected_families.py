"""Budgeted beta, generalized Laplace and rational Mellin moment theorems."""

from __future__ import annotations

import sympy as sp

from .assumptions import _ask, _normalize_assumptions
from .convergence_evidence import _finite
from .policy import _normalize_policy, _with_policy


def _true(predicate, context):
    return _ask(predicate, context) is True


def _poly(expr, variables):
    from .fast_rules import _polynomial

    budget = _normalize_policy(None)
    if not budget.allow_expensive_exact and len(variables) > budget.semialg.max_variables:
        return None
    return _polynomial(expr, variables)


def _powers(expr, var, context):
    """Bounded finite Laurent/real-power moments, without branch recombination."""
    budget = _normalize_policy(None)
    parts = sp.Add.make_args(expr)
    if not budget.allow_expensive_exact and len(parts) > budget.semialg.max_pieces:
        return None
    terms = {}
    for part in parts:
        power = part.as_powers_dict().get(var, sp.S.Zero)
        if power.has(var) or not _true(sp.Q.real(power) & sp.Q.finite(power), context):
            return None
        polynomial = _poly(part / var**power, (var,))
        if polynomial is None or not all(_finite(c, context) for c in polynomial.coeffs()):
            return None
        for (degree,), coefficient in polynomial.terms():
            exponent = power + degree
            terms[exponent] = terms.get(exponent, sp.S.Zero) + coefficient
        if not budget.allow_expensive_exact and len(terms) > budget.semialg.max_pieces:
            return None
    return tuple((p, c) for p, c in terms.items() if c != 0)


def _beta(expr, var, lo, hi, context):
    if not all(_finite(b, context) and _true(sp.Q.real(b), context) for b in (lo, hi)):
        return None
    sign = 1
    if _true(sp.Q.negative(hi - lo), context):
        lo, hi, sign = hi, lo, -1
    if not _true(sp.Q.positive(hi - lo), context):
        return None
    left, right = var - lo, hi - var
    powers = expr.as_powers_dict()
    p, q = powers.get(left, sp.S.Zero), powers.get(right, sp.S.Zero)
    if p == q == 0 or all(e.is_integer is True and e.is_nonnegative is True for e in (p, q)):
        return None
    if any(e.has(var) or not _finite(e, context) for e in (p, q)) or not _true(
        sp.Q.real(p) & sp.Q.real(q), context
    ):
        return None
    polynomial = _poly(expr / (left**p * right**q), (var,))
    if polynomial is None or not all(_finite(c, context) for c in polynomial.coeffs()):
        return None
    t = sp.Dummy("beta_coordinate", real=True)
    shifted = _poly(polynomial.as_expr().subs(var, lo + (hi - lo) * t), (t,))
    if shifted is None:
        return None
    # Exact polynomial division proves endpoint orders without sampling or roots.
    left_order = min(k[0] for k, c in shifted.terms()) if not shifted.is_zero else 0
    shifted = shifted.exquo(sp.Poly(t**left_order, t))
    if shifted is None:
        return None
    right_order = 0
    divisor = sp.Poly(1 - t, t)
    while not shifted.is_zero and shifted.eval(1) == 0:
        quotient, remainder = shifted.div(divisor)
        if not remainder.is_zero:
            break
        shifted, right_order = quotient, right_order + 1
    p, q = p + left_order, q + right_order
    if not _true(sp.Q.positive(p + 1) & sp.Q.positive(q + 1), context):
        return None
    return (
        sign,
        hi - lo,
        p,
        q,
        tuple((k, c / (hi - lo) ** (left_order + right_order)) for k, c in shifted.terms()),
    )


def _laplace(expr, var, lo, hi, context):
    factors = [f for f in sp.Mul.make_args(expr) if f.func is sp.exp]
    if len(factors) != 1:
        return None
    exponential = factors[0]
    phase = _poly(exponential.args[0], (var,))
    if phase is None or phase.is_zero or phase.degree() < 1:
        return None
    degree = int(phase.degree())
    center = sp.cancel(-phase.nth(degree - 1) / (degree * phase.LC())) if degree > 1 else sp.S.Zero
    if not _finite(center, context) or not _true(sp.Q.real(center), context):
        return None
    t = sp.Dummy("laplace_coordinate", real=True)
    shifted = _poly(phase.as_expr().subs(var, t + center), (t,))
    if shifted is None:
        return None
    rate, constant = -shifted.LC(), shifted.nth(0)
    if (
        any(shifted.nth(k) != 0 for k in range(1, degree))
        or not all(_finite(c, context) for c in (rate, constant))
        or not _true(sp.Q.positive(sp.re(rate)), context)
    ):
        return None
    sign = 1
    if (
        hi == -sp.oo
        or lo == sp.oo
        or (lo != -sp.oo and hi != sp.oo and _true(sp.Q.negative(hi - lo), context))
    ):
        lo, hi, sign = hi, lo, -1
    full = (lo, hi) == (-sp.oo, sp.oo)
    lower, upper = lo - center, hi - center
    if not full and not (
        _finite(lower, context)
        and _true(sp.Q.real(lower) & sp.Q.nonnegative(lower), context)
        and (
            upper == sp.oo
            or (
                _finite(upper, context)
                and _true(sp.Q.real(upper) & sp.Q.positive(upper - lower), context)
            )
        )
    ):
        return None
    terms = _powers((expr / exponential).subs(var, t + center), t, context)
    if terms is None:
        return None
    # A strictly positive cutoff removes the origin singularity. Upper-gamma
    # tails remain valid for all finite real orders, including order zero.
    if (full or not _true(sp.Q.positive(lower), context)) and not all(
        _true(sp.Q.positive(p + 1), context) for p, _ in terms
    ):
        return None
    if full and (degree % 2 or any(p.is_integer is not True for p, _ in terms)):
        return None
    return sign, full, degree, rate, constant, terms, lower, upper


def _mellin(expr, var, lo, hi, context):
    sign = 1
    if (lo, hi) == (sp.oo, 0):
        lo, hi, sign = hi, lo, -1
    if (lo, hi) != (0, sp.oo):
        return None
    for factor in sp.Mul.make_args(expr):
        if not factor.is_Pow or factor.exp.has(var) or not _finite(factor.exp, context):
            continue
        q = -factor.exp
        if not _true(sp.Q.real(q) & sp.Q.positive(q), context):
            continue
        base = _poly(factor.base, (var,))
        if base is None or base.degree() < 1:
            continue
        degree = int(base.degree())
        a, c = base.LC(), base.nth(0)
        if any(base.nth(k) != 0 for k in range(1, degree)) or not all(
            _finite(b, context) and _true(sp.Q.positive(b), context) for b in (a, c)
        ):
            continue
        terms = _powers(expr / factor, var, context)
        if terms is not None and all(
            _true(sp.Q.positive(p + 1) & sp.Q.positive(degree * q - p - 1), context)
            for p, _ in terms
        ):
            return sign, degree, a, c, q, terms
    return None


@_with_policy
def selected_interval_rule(expr, var, lo, hi, assumptions=None, *, evaluate=True, policy=None):
    """Return a proved family and value; evaluate=False performs recognition only."""
    context = _normalize_assumptions(assumptions)
    expr, lo, hi = map(sp.sympify, (expr, lo, hi))
    if lo.has(var) or hi.has(var):
        return None
    data = _laplace(expr, var, lo, hi, context)
    if data is not None:
        if not evaluate:
            return "generalized_laplace_moments", None
        sign, full, degree, rate, constant, terms, lower, upper = data
        value = sp.Add(
            *(
                coefficient
                * (
                    sp.gamma((p + 1) / degree)
                    if full
                    else sp.uppergamma((p + 1) / degree, rate * lower**degree)
                    - (
                        0
                        if upper == sp.oo
                        else sp.uppergamma((p + 1) / degree, rate * upper**degree)
                    )
                )
                / (degree * rate ** ((p + 1) / degree))
                for p, coefficient in terms
                if not full or p.is_even is True
            )
        )
        return "generalized_laplace_moments", sign * (2 if full else 1) * sp.exp(constant) * value
    data = _mellin(expr, var, lo, hi, context)
    if data is not None:
        if not evaluate:
            return "rational_mellin_moments", None
        sign, degree, a, c, q, terms = data
        value = sp.Add(
            *(
                coefficient
                * c ** ((p + 1) / degree - q)
                / (degree * a ** ((p + 1) / degree))
                * sp.beta((p + 1) / degree, q - (p + 1) / degree)
                for p, coefficient in terms
            )
        )
        return "rational_mellin_moments", sign * value
    data = _beta(expr, var, lo, hi, context)
    if data is not None:
        if not evaluate:
            return "endpoint_beta_moments", None
        sign, width, p, q, terms = data
        value = sp.Add(*(c * sp.beta(p + k[0] + 1, q + 1) for k, c in terms))
        return "endpoint_beta_moments", sign * width ** (p + q + 1) * value
    return None


@_with_policy
def coercive_exponential_absolute(expr, ranges, assumptions=None, *, policy=None):
    """Nonnegative mixed even terms improve separable stretched-exponential decay."""
    variables = tuple(r[0] for r in ranges)
    if not all(lo == -sp.oo and hi == sp.oo for _, lo, hi in ranges):
        return None
    factors = [f for f in sp.Mul.make_args(expr) if f.func is sp.exp]
    if len(factors) != 1:
        return None
    context = _normalize_assumptions(assumptions)
    phase = _poly(factors[0].args[0], variables)
    amplitude = _poly(expr / factors[0], variables)
    if (
        phase is None
        or amplitude is None
        or not all(_finite(c, context) for c in amplitude.coeffs())
    ):
        return None
    confined = set()
    for monomial, coefficient in phase.terms():
        if not _finite(coefficient, context):
            return None
        active = [i for i, p in enumerate(monomial) if p]
        if not active:
            continue
        if any(p % 2 for p in monomial) or not _true(sp.Q.nonpositive(coefficient), context):
            return None
        if len(active) == 1 and _true(sp.Q.negative(coefficient), context):
            confined.add(active[0])
    return True if len(confined) == len(variables) else None


@_with_policy
def positive_even_tail_absolute(expr, ranges, assumptions=None, *, policy=None):
    """Discard nonnegative terms to obtain an integrable anisotropic majorant."""
    from .anisotropic import anisotropic_absolute

    variables = tuple(r[0] for r in ranges)
    # The same whole-space majorant covers the positive orthant too.
    if not all(lo == -sp.oo and hi == sp.oo for _, lo, hi in ranges) and not all(
        lo == 0 and hi == sp.oo for _, lo, hi in ranges
    ):
        return None
    context = _normalize_assumptions(assumptions)
    for factor in sp.Mul.make_args(expr):
        if (
            not factor.is_Pow
            or factor.exp.has(*variables)
            or not _true(sp.Q.negative(factor.exp) & sp.Q.finite(factor.exp), context)
        ):
            continue
        base = _poly(factor.base, variables)
        amplitude = _poly(expr / factor, variables)
        if (
            base is None
            or amplitude is None
            or not all(_finite(c, context) for c in amplitude.coeffs())
        ):
            continue
        zero = (0,) * len(variables)
        constant = base.coeff_monomial(zero)
        if not _finite(constant, context) or not _true(sp.Q.positive(constant), context):
            continue
        pure = {}
        valid = True
        for monomial, coefficient in base.terms():
            active = [i for i, p in enumerate(monomial) if p]
            if not active:
                continue
            if (
                any(p % 2 for p in monomial)
                or not _finite(coefficient, context)
                or not _true(sp.Q.nonnegative(coefficient), context)
            ):
                valid = False
                break
            if len(active) == 1 and _true(sp.Q.positive(coefficient), context):
                i = active[0]
                if i not in pure or monomial[i] > pure[i][0]:
                    pure[i] = monomial[i], coefficient
        if not valid or len(pure) != len(variables):
            continue
        lower = constant + sp.Add(*(c * variables[i] ** degree for i, (degree, c) in pure.items()))
        majorant = amplitude.as_expr() * lower**factor.exp
        full_ranges = tuple((v, -sp.oo, sp.oo) for v in variables)
        if anisotropic_absolute(majorant, full_ranges, assumptions, policy) is True:
            return True
    return None
