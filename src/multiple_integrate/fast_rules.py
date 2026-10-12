"""Bounded exact moment recurrences and elementary definite-integral rules."""

from __future__ import annotations

from math import comb

import sympy as sp

from .assumptions import _ask, _normalize_assumptions
from .cell_proofs import finite_external_parameters, nonnegative, nonzero
from .convergence_evidence import _finite
from .planner import record_convergence, record_coverage
from .policy import _normalize_policy


def _polynomial(expr, variables):
    budget = _normalize_policy(None)
    try:
        # Bound expansion before constructing Poly for factored input.
        def estimate(node):
            if node.is_Add:
                return sum(estimate(a) for a in node.args)
            if node.is_Mul:
                count = 1
                for a in node.args:
                    count *= estimate(a)
                    if count > budget.semialg.max_pieces:
                        return count
                return count
            if node.is_Pow and node.exp.is_Integer and node.exp >= 0:
                count = estimate(node.base)
                if count > budget.semialg.max_pieces:
                    return count
                if count > 1:
                    degree = int(node.exp)
                    if degree > budget.semialg.max_pieces:
                        return budget.semialg.max_pieces + 1
                    return comb(count + degree - 1, degree)
            return 1

        if (
            not budget.allow_expensive_exact
            and estimate(sp.sympify(expr)) > budget.semialg.max_pieces
        ):
            return None
        poly = sp.Poly(expr, *variables)
        if not budget.allow_expensive_exact and (
            poly.total_degree() > budget.semialg.max_polynomial_degree
            or len(poly.terms()) > budget.semialg.max_pieces
        ):
            return None
        return poly
    except (sp.PolynomialError, ValueError, TypeError):
        return None


def box_polynomial_moment(expr, ranges, assumptions=None):
    """Sparse Cartesian moments with finite coefficients and bounded expansion."""
    context = _normalize_assumptions(assumptions)
    variables = tuple(r[0] for r in ranges)
    if not variables or any(sp.sympify(b).has(*variables) for r in ranges for b in r[1:]):
        return None
    if not all(
        _finite(sp.sympify(b), context) and _ask(sp.Q.real(b), context) is True
        for r in ranges
        for b in r[1:]
    ):
        return None
    polynomial = _polynomial(expr, variables)
    if polynomial is None or not all(_finite(c, context) for c in polynomial.coeffs()):
        return None
    return sp.Add(
        *(
            c
            * sp.prod(
                (sp.sympify(hi) ** (k + 1) - sp.sympify(lo) ** (k + 1)) / sp.Integer(k + 1)
                for (_, lo, hi), k in zip(ranges, powers, strict=True)
            )
            for powers, c in polynomial.terms()
        )
    )


def _certificate(
    method,
    *,
    absolute=True,
    basis="proved moment rule: compact continuity or positive Gaussian decay",
):
    record_coverage(
        "integration_domain",
        True,
        method,
        basis="exact whole-interval formula with proved prerequisites",
    )
    if absolute is True:
        record_coverage(
            "absolute_convergence",
            True,
            method,
            basis=basis,
        )
    else:
        record_convergence("ordinary", True, method)
        record_convergence("absolute", absolute, method)


def _affine_moments(polynomial, a, b, lo, hi, function, context):
    n = int(polynomial.degree()) if not polynomial.is_zero else 0
    zero = sp.Add(
        *(c * (hi ** (k + 1) - lo ** (k + 1)) / sp.Integer(k + 1) for (k,), c in polynomial.terms())
    ) * function(b)
    if _ask(sp.Q.zero(a), context) is True:
        return zero
    if function is sp.exp:
        left, right = sp.exp(a * lo + b), sp.exp(a * hi + b)
        moments = [(right - left) / a]
        for k in range(1, n + 1):
            moments.append((hi**k * right - lo**k * left - k * moments[-1]) / a)
    else:
        sl, sh = sp.sin(a * lo + b), sp.sin(a * hi + b)
        cl, ch = sp.cos(a * lo + b), sp.cos(a * hi + b)
        sine, cosine = [(cl - ch) / a], [(sh - sl) / a]
        for k in range(1, n + 1):
            sine.append((lo**k * cl - hi**k * ch + k * cosine[k - 1]) / a)
            cosine.append((hi**k * sh - lo**k * sl - k * sine[k - 1]) / a)
        moments = sine if function is sp.sin else cosine
    value = sp.Add(*(c * moments[k] for (k,), c in polynomial.terms()))
    return value if nonzero(a, context) else sp.Piecewise((zero, sp.Eq(a, 0)), (value, True))


def polynomial_function_moment(expr, var, lo, hi, assumptions=None):
    """Polynomial times affine exp/trig or a proved decaying quadratic exp."""
    context = _normalize_assumptions(assumptions)
    lo, hi = sp.sympify(lo), sp.sympify(hi)
    if not all(
        b in (-sp.oo, sp.oo)
        or (_finite(b, context) and _ask(sp.Q.real(b), context) is True and not b.has(var))
        for b in (lo, hi)
    ):
        return None
    factors = sp.Mul.make_args(expr)
    special = [f for f in factors if f.func in (sp.exp, sp.sin, sp.cos)]
    if len(special) != 1:
        return None
    f = special[0]
    polynomial = _polynomial(expr / f, (var,))
    phase = _polynomial(f.args[0], (var,))
    if (
        polynomial is None
        or phase is None
        or not all(_finite(c, context) for c in (*polynomial.coeffs(), *phase.coeffs()))
    ):
        return None
    if phase.degree() <= 1:
        if lo in (-sp.oo, sp.oo) or hi in (-sp.oo, sp.oo):
            return None
        return _affine_moments(polynomial, phase.nth(1), phase.nth(0), lo, hi, f.func, context)
    if f.func is not sp.exp or phase.degree() != 2:
        return None
    alpha = -phase.nth(2)
    if not nonnegative(alpha, context, strict=True):
        return None
    center = phase.nth(1) / (2 * alpha)
    constant = phase.nth(0) + alpha * center**2
    t = sp.Dummy("gaussian_centered")
    shifted = sp.Poly(polynomial.as_expr().subs(var, t + center), t)
    left, right = lo - center, hi - center

    def endpoint_power(endpoint, centered, degree):
        return (
            sp.S.Zero
            if endpoint in (-sp.oo, sp.oo)
            else centered**degree * sp.exp(-alpha * centered**2)
        )

    el, er = endpoint_power(lo, left, 0), endpoint_power(hi, right, 0)

    def endpoint_erf(endpoint, centered):
        if endpoint == sp.oo:
            return sp.S.One
        if endpoint == -sp.oo:
            return -sp.S.One
        return sp.erf(sp.sqrt(alpha) * centered)

    moments = [
        sp.sqrt(sp.pi) / (2 * sp.sqrt(alpha)) * (endpoint_erf(hi, right) - endpoint_erf(lo, left)),
        (el - er) / (2 * alpha),
    ]
    for k in range(2, int(shifted.degree()) + 1 if not shifted.is_zero else 1):
        moments.append(
            (
                endpoint_power(lo, left, k - 1)
                - endpoint_power(hi, right, k - 1)
                + (k - 1) * moments[k - 2]
            )
            / (2 * alpha)
        )
    return sp.exp(constant) * sp.Add(*(c * moments[k] for (k,), c in shifted.terms()))


def root_free_rational(expr, var, lo, hi, assumptions=None):
    """Real linear/quadratic denominators, certified root-free on a finite interval."""
    context = _normalize_assumptions(assumptions)
    lo, hi = sp.sympify(lo), sp.sympify(hi)
    if not all(
        _finite(b, context) and _ask(sp.Q.real(b), context) is True and not b.has(var)
        for b in (lo, hi)
    ):
        return None
    num, den = sp.fraction(expr)
    numerator, denominator = _polynomial(num, (var,)), _polynomial(den, (var,))
    if numerator is None or denominator is None or denominator.degree() not in (1, 2):
        return None
    if not all(_finite(c, context) for c in numerator.coeffs()) or not all(
        _finite(c, context) and _ask(sp.Q.real(c), context) is True for c in denominator.coeffs()
    ):
        return None
    degree = int(denominator.degree())
    a = denominator.nth(degree)
    if not nonzero(a, context):
        return None
    if degree == 1:
        b = denominator.nth(0)
        left, right = a * lo + b, a * hi + b
        if not (
            nonnegative(left, context, strict=True)
            and nonnegative(right, context, strict=True)
            or nonnegative(-left, context, strict=True)
            and nonnegative(-right, context, strict=True)
        ):
            return None
        primitive = sp.log(sp.Abs(a * var + b)) / a
    else:
        b, c = denominator.nth(1), denominator.nth(0)
        d = 4 * a * c - b * b
        if nonnegative(d, context, strict=True):
            primitive = 2 / sp.sqrt(d) * sp.atan((2 * a * var + b) / sp.sqrt(d))
        else:

            def avoids_zero(linear):
                left, right = linear.subs(var, lo), linear.subs(var, hi)
                return (
                    nonnegative(left, context, strict=True)
                    and nonnegative(right, context, strict=True)
                    or nonnegative(-left, context, strict=True)
                    and nonnegative(-right, context, strict=True)
                )

            if _ask(sp.Q.zero(d), context) is True:
                if not avoids_zero(2 * a * var + b):
                    return None
                primitive = -2 / (2 * a * var + b)
            elif nonnegative(-d, context, strict=True):
                root = sp.sqrt(-d)
                minus, plus = 2 * a * var + b - root, 2 * a * var + b + root
                if not avoids_zero(minus) or not avoids_zero(plus):
                    return None
                primitive = sp.log(sp.Abs(minus / plus)) / root
            else:
                return None
    quotient, remainder = sp.div(numerator, denominator)
    if degree == 1:
        tail = remainder.nth(0) * (primitive.subs(var, hi) - primitive.subs(var, lo))
    else:
        derivative_weight = remainder.nth(1) / (2 * a)
        left, right = denominator.eval(lo), denominator.eval(hi)
        tail = derivative_weight * sp.log(sp.Abs(right) / sp.Abs(left)) + (
            remainder.nth(0) - derivative_weight * b
        ) * (primitive.subs(var, hi) - primitive.subs(var, lo))
    return box_polynomial_moment(quotient.as_expr(), ((var, lo, hi),), context) + tail


def quartic_rational_moment(expr, var, lo, hi, assumptions=None):
    """Absolutely integrable degree-two moments of shifted positive biquadratics."""
    if (lo, hi) not in ((-sp.oo, sp.oo), (sp.oo, -sp.oo)):
        return None
    context = _normalize_assumptions(assumptions)
    num, den = sp.fraction(expr)
    numerator, denominator = _polynomial(num, (var,)), _polynomial(den, (var,))
    if (
        numerator is None
        or denominator is None
        or denominator.degree() != 4
        or numerator.degree() > 2
    ):
        return None
    if not all(_finite(c, context) for c in numerator.coeffs()) or not all(
        _finite(c, context) and _ask(sp.Q.real(c), context) is True for c in denominator.coeffs()
    ):
        return None
    a = denominator.nth(4)
    if nonnegative(-a, context, strict=True):
        numerator, denominator, a = -numerator, -denominator, -a
    if not nonnegative(a, context, strict=True):
        return None
    center = -denominator.nth(3) / (4 * a)
    t = sp.Dummy("quartic_centered", real=True)
    shifted = sp.Poly(sp.expand(denominator.as_expr().subs(var, t + center)), t)
    if any(sp.simplify(shifted.nth(k)) != 0 for k in (1, 3)):
        return None
    b, c = sp.simplify(shifted.nth(2)), sp.simplify(shifted.nth(0))
    # This avoids asking a large SAT query about a nested radical. For B<0,
    # C>B**2/(4*A) proves positivity at the quadratic's minimum in t**2.
    positive_remainder = nonnegative(sp.factor(c - b * b / (4 * a)), context, strict=True)
    if not (positive_remainder or nonnegative(c, context, strict=True) and nonnegative(b, context)):
        return None
    weight = sp.Poly(sp.expand(numerator.as_expr().subs(var, t + center)), t)
    d = b + 2 * sp.sqrt(a * c)
    value = sp.pi / sp.sqrt(d) * (weight.nth(0) / sp.sqrt(c) + weight.nth(2) / sp.sqrt(a))
    return value if lo == -sp.oo else -value


def reciprocal_phase_rule(expr, var, lo, hi, assumptions=None):
    """Ordinary one-sided Si/Ci limits for sin(c/(x-d))/(x-d) and cos analogues."""
    context = _normalize_assumptions(assumptions)
    factors = sp.Mul.make_args(expr)
    trig = next((f for f in factors if f.func in (sp.sin, sp.cos)), None)
    if trig is None:
        return None
    rest = sp.cancel(expr / trig)
    num, den = sp.fraction(rest)
    polynomial = _polynomial(den, (var,))
    if polynomial is None or polynomial.degree() != 1 or num.has(var):
        return None
    slope = polynomial.nth(1)
    if not nonzero(slope, context):
        return None
    center = -polynomial.nth(0) / slope
    coefficient = num / slope
    c = sp.cancel(trig.args[0] * (var - center))
    if (
        c.has(var)
        or not all(_finite(v, context) for v in (c, center, coefficient))
        or _ask(sp.Q.real(c) & sp.Q.real(center), context) is not True
    ):
        return None
    lo, hi = sp.sympify(lo), sp.sympify(hi)
    if not all(_finite(v, context) and _ask(sp.Q.real(v), context) is True for v in (lo, hi)):
        return None
    if lo == hi:
        return sp.S.Zero, True
    if not nonzero(c, context):
        return None
    sign = (
        1
        if nonnegative(c, context, strict=True)
        else -1
        if nonnegative(-c, context, strict=True)
        else None
    )
    if sign is None:
        return None

    def primitive(endpoint, side):
        distance = sp.simplify(endpoint - center)
        if distance == 0:
            return -sign * side * sp.pi / 2 if trig.func is sp.sin else sp.S.Zero
        if nonnegative(distance, context, strict=True) or nonnegative(
            -distance, context, strict=True
        ):
            return -sp.Si(c / distance) if trig.func is sp.sin else -sp.Ci(sp.Abs(c / distance))
        return None

    left, right = sp.simplify(lo - center), sp.simplify(hi - center)

    def side(t):
        return (
            1
            if nonnegative(t, context, strict=True)
            else -1
            if nonnegative(-t, context, strict=True)
            else 0
            if t == 0
            else None
        )

    sl, sr = side(left), side(right)
    if sl is None or sr is None:
        return None
    if sl == 0:
        value = primitive(hi, sr) - primitive(lo, sr)
    elif sr == 0:
        value = primitive(hi, sl) - primitive(lo, sl)
    elif sl != sr:
        value = (
            primitive(center, sl) - primitive(lo, sl) + primitive(hi, sr) - primitive(center, sr)
        )
    else:
        value = primitive(hi, sr) - primitive(lo, sl)
    singular = sl == 0 or sr == 0 or sl != sr
    absolute = (
        True
        if not singular or _ask(sp.Q.zero(coefficient), context) is True
        else False
        if nonzero(coefficient, context)
        else None
    )
    return coefficient * value, absolute


def periodic_trigonometric_tail(expr, var, lo, hi, context):
    """Dirichlet tails and positive periodic-mean divergence, without a backend."""
    lo, hi = sp.sympify(lo), sp.sympify(hi)
    if hi != sp.oo or not (lo == 0 or nonnegative(lo, context, strict=True)):
        return None
    absolute = False
    factors = sp.Mul.make_args(expr)
    trig = next(
        (
            f
            for f in factors
            if f.func in (sp.sin, sp.cos)
            or (f.func is sp.Abs and f.args[0].func in (sp.sin, sp.cos))
        ),
        None,
    )
    if trig is None:
        return None
    coefficient = sp.cancel(expr * var / trig)
    if (
        coefficient.has(var)
        or not _finite(coefficient, context)
        or not nonzero(coefficient, context)
    ):
        return None
    if trig.func is sp.Abs:
        absolute = True
        trig = trig.args[0]
    phase = _polynomial(trig.args[0], (var,))
    if phase is None or phase.degree() != 1:
        return None
    frequency, offset = phase.nth(1), phase.nth(0)
    if not all(
        _finite(c, context) and _ask(sp.Q.real(c), context) is True for c in (frequency, offset)
    ):
        return None
    sign = (
        1
        if nonnegative(frequency, context, strict=True)
        else -1
        if nonnegative(-frequency, context, strict=True)
        else None
    )
    if sign is None:
        return None
    if absolute:
        # A nonconstant absolute sinusoid has strictly positive periodic mean;
        # comparison over successive periods proves harmonic-tail divergence.
        if nonnegative(coefficient, context, strict=True):
            return sp.oo, False
        if nonnegative(-coefficient, context, strict=True):
            return -sp.oo, False
        return None
    sine_weight = sp.cos(offset) if trig.func is sp.sin else -sp.sin(offset)
    cosine_weight = sp.sin(offset) if trig.func is sp.sin else sp.cos(offset)
    if lo == 0:
        if cosine_weight != 0 and _ask(sp.Q.zero(cosine_weight), context) is not True:
            return None
        return coefficient * sine_weight * sign * sp.pi / 2, True
    return coefficient * (
        sine_weight * sign * (sp.pi / 2 - sp.Si(sp.Abs(frequency) * lo))
        - cosine_weight * sp.Ci(sp.Abs(frequency) * lo)
    ), True


def fast_interval_rule(expr, var, lo, hi, assumptions=None, *, evidence=True):
    """Return (strategy, value) before generic singularity/backend analysis."""
    context = _normalize_assumptions(assumptions)
    if not finite_external_parameters(expr, (var,), context):
        return None
    tail = periodic_trigonometric_tail(expr, var, lo, hi, context)
    if tail is not None:
        value, ordinary = tail
        method = "periodic_trigonometric_tail"
        if evidence:
            record_coverage(
                "integration_domain",
                True,
                method,
                basis="Dirichlet tail or positive periodic-mean comparison",
            )
            record_convergence("ordinary", ordinary, method)
            record_convergence("absolute", False, method)
        return method, value
    value = polynomial_function_moment(expr, var, lo, hi, context)
    if value is not None:
        method = "polynomial_function_moments"
        if evidence:
            _certificate(method)
        return method, value
    value = root_free_rational(expr, var, lo, hi, context)
    if value is not None:
        method = "root_free_rational"
        if evidence:
            _certificate(method)
        return method, value
    value = quartic_rational_moment(expr, var, lo, hi, context)
    if value is not None:
        method = "quartic_rational_moments"
        if evidence:
            _certificate(
                method,
                basis="positive shifted biquadratic denominator and numerator degree at most two give an integrable quadratic tail",
            )
        return method, value
    from .selected_families import selected_interval_rule

    selected = selected_interval_rule(expr, var, lo, hi, context)
    if selected is not None:
        method, value = selected
        if evidence:
            _certificate(
                method,
                basis="proved endpoint orders and exponential decay or positive Mellin denominator",
            )
        return method, value
    oscillatory = reciprocal_phase_rule(expr, var, lo, hi, context)
    if oscillatory is not None:
        value, absolute = oscillatory
        method = "reciprocal_phase"
        if evidence:
            _certificate(method, absolute=absolute)
        return method, value
    return None
