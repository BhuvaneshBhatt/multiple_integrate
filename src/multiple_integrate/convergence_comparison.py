"""Bounded-multiplier, affine-power and positive-quadratic comparison proofs."""

from __future__ import annotations

import sympy as sp

from .assumptions import _ask, _normalize_assumptions
from .quadratic_radial import positive_quadratic


def _true(predicate, assumptions):
    return _ask(predicate, _normalize_assumptions(assumptions)) is True


def scaled_quadratic_power(f, ranges, assumptions):
    """Pure powers of translated SPD quadratics; sign-constant divergence only."""
    coords = tuple(r[0] for r in ranges)
    candidates = [
        factor
        for factor in sp.Mul.make_args(f)
        if factor.is_Pow and factor.exp.is_real is True and not factor.exp.has(*coords)
    ]
    if len(candidates) != 1:
        return None
    factor = candidates[0]
    coefficient = sp.simplify(f / factor)
    if coefficient.has(*coords) or not _true(sp.Q.finite(coefficient), assumptions):
        return None
    data = positive_quadratic(factor.base, coords, assumptions)
    if data is None:
        return None
    _, center, minimum = data
    full = all(lo == -sp.oo and hi == sp.oo for _, lo, hi in ranges)
    local = minimum == 0 and all(
        _true(
            sp.Q.finite(lo) & sp.Q.finite(hi) & sp.Q.positive(c - lo) & sp.Q.positive(hi - c),
            assumptions,
        )
        for (_, lo, hi), c in zip(ranges, center, strict=True)
    )
    if not local and not (full and _true(sp.Q.positive(minimum), assumptions)):
        return None
    threshold = 2 * factor.exp + len(coords)
    good = sp.Q.positive(threshold) if local else sp.Q.negative(threshold)
    bad = sp.Q.nonpositive(threshold) if local else sp.Q.nonnegative(threshold)
    if _true(good, assumptions):
        return True
    # Nonzero constant multiples have sign/phase constant, so here an
    # absolute-divergence claim also establishes ordinary divergence.
    if _true(bad, assumptions) and _true(sp.Q.nonzero(coefficient), assumptions):
        return False
    return None


def polynomial_exponential_bound(phase, coords, ranges, assumptions):
    """Bound exp(Re(phase)) by its constant term on proved Cartesian signs."""
    fresh = {v: sp.Dummy(v.name, real=True) for v in coords}
    from .fast_rules import _polynomial

    polynomial = _polynomial(sp.re(phase.xreplace(fresh)), tuple(fresh.values()))
    if polynomial is None:
        return False
    if not all(_true(sp.Q.real(c) & sp.Q.finite(c), assumptions) for c in polynomial.coeffs()):
        return False
    for monomial, coefficient in polynomial.terms():
        if not any(monomial):
            continue
        sign = 1
        for degree, (_, lo, hi) in zip(monomial, ranges, strict=True):
            if degree % 2:
                if _true(sp.Q.nonnegative(sp.Min(sp.sympify(lo), sp.sympify(hi))), assumptions):
                    pass
                elif _true(sp.Q.nonpositive(sp.Max(sp.sympify(lo), sp.sympify(hi))), assumptions):
                    sign = -sign
                else:
                    return False
        if not _true(sp.Q.nonpositive(sign * coefficient), assumptions):
            return False
    return True


def bounded_multiplier(factor, coords, ranges, assumptions, compact_rule):
    """Return a sufficient uniform bound, without guessing asymptotic cancellation."""
    if not factor.has(*coords):
        return _true(sp.Q.finite(factor), assumptions)
    if (
        factor.is_Pow
        and not factor.exp.has(*coords)
        and _true(
            sp.Q.real(factor.exp) & sp.Q.finite(factor.exp) & sp.Q.nonnegative(factor.exp),
            assumptions,
        )
    ):
        return bounded_multiplier(factor.base, coords, ranges, assumptions, compact_rule)
    if factor.func in (sp.sin, sp.cos):
        fresh = {v: sp.Dummy(v.name, real=True) for v in coords}
        arg = factor.args[0].xreplace(fresh)
        return _true(sp.Q.real(arg), assumptions)
    if factor.func == sp.exp:
        fresh = {v: sp.Dummy(v.name, real=True) for v in coords}
        phase = sp.simplify(factor.args[0] / sp.I).xreplace(fresh)
        return _true(sp.Q.real(phase), assumptions) or polynomial_exponential_bound(
            factor.args[0], coords, ranges, assumptions
        )
    return compact_rule(factor, ranges, assumptions=assumptions)


def affine_power_absolute(f, range_, assumptions):
    v, lo, hi = map(sp.sympify, range_)
    if hi == sp.oo and _true(sp.Q.positive(lo) & sp.Q.finite(lo), assumptions):
        coefficient, power = f.as_coeff_exponent(v)
        if (
            not coefficient.has(v)
            and not power.has(v)
            and _true(
                sp.Q.finite(coefficient) & sp.Q.real(power) & sp.Q.negative(power + 1), assumptions
            )
        ):
            return True
    finite_interval = _true(sp.Q.positive(hi - lo) & sp.Q.finite(lo) & sp.Q.finite(hi), assumptions)
    right_tail = hi == sp.oo and _true(sp.Q.real(lo) & sp.Q.finite(lo), assumptions)
    left_tail = lo == -sp.oo and _true(sp.Q.real(hi) & sp.Q.finite(hi), assumptions)
    if not (finite_interval or right_tail or left_tail):
        return None
    powers = [a for a in sp.Mul.make_args(f) if a.is_Pow and a.base.has(v)]
    if len(powers) != 1:
        return None
    factor = powers[0]
    coefficient = sp.simplify(f / factor)
    if (
        coefficient.has(v)
        or factor.exp.has(v)
        or not _true(sp.Q.finite(coefficient) & sp.Q.real(factor.exp), assumptions)
    ):
        return None
    try:
        base = sp.Poly(factor.base, v)
    except sp.PolynomialError:
        return None
    if base.degree() != 1:
        return None
    slope = base.nth(1)
    if not all(_true(sp.Q.real(c) & sp.Q.finite(c), assumptions) for c in base.coeffs()):
        return None
    if right_tail or left_tail:
        endpoint = lo if right_tail else hi
        inward_slope = slope if right_tail else -slope
        value = sp.simplify(factor.base.subs(v, endpoint))
        if _true(
            sp.Q.positive(inward_slope) & sp.Q.positive(value) & sp.Q.negative(factor.exp + 1),
            assumptions,
        ):
            return True
        return None
    left = sp.simplify(factor.base.subs(v, lo))
    right = sp.simplify(factor.base.subs(v, hi))
    if (
        (left == 0 and _true(sp.Q.positive(slope), assumptions))
        or (right == 0 and _true(sp.Q.negative(slope), assumptions))
    ) and _true(sp.Q.positive(factor.exp + 1), assumptions):
        return True
    return None


def power_log_absolute(f, range_, assumptions):
    """Power-log comparison at zero or infinity, away from log's other zero."""
    v, lo, hi = map(sp.sympify, range_)
    power = sp.S.Zero
    log_power = sp.S.Zero
    has_log = False
    coefficient = sp.S.One
    for factor in sp.Mul.make_args(f):
        if factor == v or (factor.is_Pow and factor.base == v):
            power += sp.S.One if factor == v else factor.exp
        elif factor == sp.log(v) or (factor.is_Pow and factor.base in (sp.log(v), -sp.log(v))):
            exponent = sp.S.One if factor == sp.log(v) else factor.exp
            if not _true(sp.Q.real(exponent) & sp.Q.finite(exponent), assumptions):
                return None
            # Real powers of opposite log branches have the same magnitude.
            # Account for every factor; retaining only the last exponent can
            # turn the divergent logarithmic threshold into a convergence proof.
            log_power += exponent
            has_log = True
        elif not factor.has(v):
            coefficient *= factor
        else:
            return None
    if (
        not has_log
        or any(p.has(v) for p in (power, log_power))
        or not _true(
            sp.Q.real(power) & sp.Q.real(log_power) & sp.Q.finite(coefficient), assumptions
        )
    ):
        return None
    if (
        hi == sp.oo
        and _true(sp.Q.positive(lo - 1) & sp.Q.finite(lo), assumptions)
        and (
            _true(sp.Q.negative(power + 1), assumptions)
            or (power == -1 and _true(sp.Q.negative(log_power + 1), assumptions))
        )
    ):
        return True
    if (
        lo == 0
        and _true(sp.Q.positive(hi) & sp.Q.positive(1 - hi), assumptions)
        and (
            _true(sp.Q.positive(power + 1), assumptions)
            or (power == -1 and _true(sp.Q.negative(log_power + 1), assumptions))
        )
    ):
        return True
    return None


def polynomial_majorant(f, coords, assumptions):
    """Positive even monomial remainders can only improve negative-power decay."""
    if not f.is_Pow or f.exp.has(*coords) or not _true(sp.Q.negative(f.exp), assumptions):
        return None
    try:
        polynomial = sp.Poly(f.base, *coords)
    except sp.PolynomialError:
        return None
    low = sp.S.Zero
    remainder = []
    for monomial, c in polynomial.terms():
        term = c * sp.prod(v**p for v, p in zip(coords, monomial, strict=True))
        if sum(monomial) <= 2:
            low += term
        else:
            remainder.append((monomial, c))
    if not remainder or not all(
        all(p % 2 == 0 for p in m) and _true(sp.Q.nonnegative(c), assumptions) for m, c in remainder
    ):
        return None
    data = positive_quadratic(low, coords, assumptions)
    if data is None or not _true(sp.Q.positive(data[2]), assumptions):
        return None
    return low**f.exp


def local_polynomial_comparison(f, ranges, assumptions):
    """Finite polynomial vanishing order controls an SPD point singularity."""
    coords = tuple(r[0] for r in ranges)
    if not all(
        _true(sp.Q.real(lo) & sp.Q.real(hi) & sp.Q.finite(lo) & sp.Q.finite(hi), assumptions)
        for _, lo, hi in ranges
    ):
        return None
    for factor in sp.Mul.make_args(f):
        if not factor.is_Pow or factor.exp.has(*coords) or factor.exp.is_real is not True:
            continue
        data = positive_quadratic(factor.base, coords, assumptions)
        if data is None or data[2] != 0:
            continue
        _, center, _ = data
        try:
            polynomial = sp.Poly(sp.cancel(f / factor), *coords)
            local = tuple(sp.Dummy(f"vanishing{i}", real=True) for i in range(len(coords)))
            shifted = sp.Poly(
                sp.expand(
                    polynomial.as_expr().subs(
                        {v: u + c for v, u, c in zip(coords, local, center, strict=True)},
                        simultaneous=True,
                    )
                ),
                *local,
            )
        except sp.PolynomialError:
            continue
        terms = [(m, c) for m, c in shifted.terms() if c != 0]
        if terms and all(_true(sp.Q.finite(c), assumptions) for _, c in terms):
            order = min(sum(m) for m, _ in terms)
            if _true(sp.Q.positive(order + 2 * factor.exp + len(coords)), assumptions):
                return True
    return None


def affine_laurent_log_absolute(f, range_, assumptions):
    """Finite Laurent amplitudes times log powers, at a proved affine endpoint/tail."""
    from .selected_families import _powers

    v, lo, hi = map(sp.sympify, range_)
    if hi == -sp.oo or lo == sp.oo or _true(sp.Q.negative(hi - lo), assumptions):
        lo, hi = hi, lo
    context = _normalize_assumptions(assumptions)
    logs = f.atoms(sp.log)
    if len(logs) != 1:
        return None
    logarithm = next(iter(logs))
    try:
        base = sp.Poly(logarithm.args[0], v)
    except sp.PolynomialError:
        return None
    if base.degree() != 1 or not all(
        _true(sp.Q.real(c) & sp.Q.finite(c), assumptions) for c in base.coeffs()
    ):
        return None
    slope = base.nth(1)
    if not _true(sp.Q.nonzero(slope), assumptions):
        return None
    # Work with fresh real coordinates; Symbol flags do not constrain integration ranges.
    t = sp.Dummy("log_comparison_coordinate", real=True)
    mapped = f.subs(v, (t - base.nth(0)) / slope)
    exponent = sp.S.Zero
    amplitude = sp.S.One
    for factor in sp.Mul.make_args(mapped):
        log_base = factor.base if factor.is_Pow else factor
        if log_base in (sp.log(t), -sp.log(t)):
            q = factor.exp if factor.is_Pow else sp.S.One
            if not _true(sp.Q.real(q) & sp.Q.finite(q), assumptions):
                return None
            exponent += q
        else:
            amplitude *= factor
    terms = _powers(amplitude, t, context)
    if terms is None:
        return None
    ends = [sp.simplify(logarithm.args[0].subs(v, end)) for end in (lo, hi)]
    if _true(sp.Q.positive(slope), assumptions):
        lower, upper = ends
    elif _true(sp.Q.negative(slope), assumptions):
        upper, lower = ends
    else:
        return None
    tail = upper == sp.oo and _true(sp.Q.positive(lower - 1) & sp.Q.finite(lower), assumptions)
    origin = lower == 0 and _true(sp.Q.positive(upper) & sp.Q.positive(1 - upper), assumptions)
    if not (tail or origin):
        return None
    for power, _ in terms:
        good = sp.Q.negative(power + 1) if tail else sp.Q.positive(power + 1)
        if not _true(good, assumptions) and not (
            _true(sp.Q.zero(power + 1), assumptions)
            and _true(sp.Q.negative(exponent + 1), assumptions)
        ):
            return None
    return True
