"""Exact shared-product cutoffs on certified convex downward-closed polygons."""

from __future__ import annotations

from functools import cmp_to_key

import sympy as sp

from .assumptions import _normalize_assumptions
from .cell_proofs import native_predicate_true as _true
from .policy import _CURRENT_POLICY, StrategyBudgetExceeded


def linear_segment_moment(p, q, A, B, left, right, e, assumptions=None):
    """Integrate u**p*v**q between v=e/u and v=B+A*u on a proved interval."""
    from .singular_maps import _rational_primitive

    u = sp.Dummy("polygon_coordinate", positive=True)

    def primitive(expr):
        value = _rational_primitive(expr, u)
        if value is not None and A != 0 and _true(sp.Q.negative(A), assumptions):
            value = value.xreplace({sp.log(u + B / A): sp.log(-u - B / A)})
        return value

    if q != -1:
        a = primitive(u**p * (B + A * u) ** (q + 1))
        b = primitive(u ** (p - q - 1))
        if a is None or b is None:
            return None
        antiderivative = (a - e ** (q + 1) * b) / (q + 1)
    elif p == -1:
        antiderivative = (
            sp.log(B) * sp.log(u)
            - sp.polylog(2, -A * u / B)
            + sp.log(u) ** 2 / 2
            - sp.log(e) * sp.log(u)
        )
    else:
        correction = primitive(A * u ** (p + 1) / (B + A * u))
        if correction is None:
            return None
        antiderivative = (
            (u ** (p + 1) * sp.log(B + A * u) - correction) / (p + 1)
            + u ** (p + 1) * (sp.log(u) / (p + 1) - 1 / sp.Integer(p + 1) ** 2)
            - sp.log(e) * u ** (p + 1) / (p + 1)
        )
    result = antiderivative.subs(u, right) - antiderivative.subs(u, left)
    replacements = {}
    for term in result.atoms(sp.polylog):
        if term.args[0] == 2 and sp.simplify(term.args[1].subs(e, 0) - 1) == 0:
            z = term.args[1]
            replacements[term] = sp.pi**2 / 6 - sp.polylog(2, 1 - z) - sp.log(z) * sp.log(1 - z)
    return sp.expand(result.xreplace(replacements))


def polygon_segments(halfspaces, su, sv, assumptions=None):
    """Certify an orthant polygon as the lower envelope of decreasing positive lines."""
    from .symbolic_geometry import _compare, _Uncertain

    context = _normalize_assumptions(assumptions)
    lines = []
    caps = []
    for a, b, c in halfspaces:
        a, b = sp.simplify(a * su), sp.simplify(b * sv)
        if not _true(sp.Q.real(c) & sp.Q.finite(c) & sp.Q.nonnegative(c), assumptions):
            return None
        if _true(sp.Q.nonnegative(a) & sp.Q.nonnegative(b), assumptions):
            continue
        if not _true(sp.Q.nonpositive(a) & sp.Q.nonpositive(b), assumptions):
            return None
        if c == 0:
            return ()
        if _true(sp.Q.negative(b), assumptions):
            A, B = sp.simplify(-a / b), sp.simplify(-c / b)
            lines.append((A, B))
            if _true(sp.Q.negative(A), assumptions):
                caps.append(-B / A)
        elif b == 0 and _true(sp.Q.negative(a), assumptions):
            caps.append(-c / a)
        else:
            return None
    if not lines or not caps:
        return None
    budget = _CURRENT_POLICY.get()
    if not budget.allow_expensive_exact and len(lines) ** 2 > budget.hypersurface.max_pieces:
        raise StrategyBudgetExceeded("polygon envelope exceeds pair budget")
    try:
        U = sorted(caps, key=cmp_to_key(lambda a, b: _compare(a, b, context)))[0]
        if _compare(U, sp.S.Zero, context) <= 0:
            return ()
        breaks = [sp.S.Zero, U]
        for i, (A, B) in enumerate(lines):
            for C, D in lines[i + 1 :]:
                if sp.simplify(A - C) == 0:
                    continue
                if not _true(sp.Q.nonzero(A - C), assumptions):
                    return None
                cross = sp.simplify((D - B) / (A - C))
                if _compare(cross, sp.S.Zero, context) > 0 and _compare(cross, U, context) < 0:
                    breaks.append(cross)
        ordered = []
        for endpoint in sorted(breaks, key=cmp_to_key(lambda a, b: _compare(a, b, context))):
            if not ordered or _compare(endpoint, ordered[-1], context) != 0:
                ordered.append(endpoint)
        segments = []
        for lo, hi in zip(ordered, ordered[1:], strict=False):
            midpoint = (lo + hi) / 2
            A, B = sorted(
                lines,
                key=cmp_to_key(
                    lambda a, b, midpoint=midpoint: _compare(
                        a[0] * midpoint + a[1], b[0] * midpoint + b[1], context
                    )
                ),
            )[0]
            # Affine differences cannot change order inside an interval with no crossings.
            if not all(
                _compare(C * endpoint + D, A * endpoint + B, context) >= 0
                for C, D in lines
                for endpoint in (lo, hi)
            ):
                return None
            if (
                _compare(A * lo + B, sp.S.Zero, context) < 0
                or _compare(A * hi + B, sp.S.Zero, context) < 0
            ):
                return None
            if segments and segments[-1][2:] == (A, B):
                segments[-1] = (segments[-1][0], hi, A, B)
            else:
                segments.append((lo, hi, A, B))
        return tuple(segments)
    except _Uncertain:
        return None


def clipped_polygon_cutoff(
    num, den_powers, den_coefficient, halfspaces, coefficient, variables, epsilon, assumptions=None
):
    """Sum all occupied orthants under a single original defining-equation cutoff."""
    e = sp.Dummy("polygon_cutoff", positive=True)
    total = sp.S.Zero
    all_segments = []
    windows = []
    occupied = 0
    budget = _CURRENT_POLICY.get()
    for su in (-1, 1):
        for sv in (-1, 1):
            # Sign constraints with zero intercept can remove an entire open orthant.
            if any(
                c == 0
                and (
                    (b == 0 and _true(sp.Q.negative(a * su), assumptions))
                    or (a == 0 and _true(sp.Q.negative(b * sv), assumptions))
                )
                for a, b, c in halfspaces
            ):
                continue
            segments = polygon_segments(halfspaces, su, sv, assumptions)
            if segments is None:
                return None
            if not segments:
                continue
            occupied += 1
            for index, (lo, hi, A, B) in enumerate(segments):
                left = lo
                right = hi
                if lo == 0:
                    left = e / B if A == 0 else 2 * e / (B + sp.sqrt(B**2 + 4 * A * e))
                else:
                    windows.append(lo * (A * lo + B))
                if sp.simplify(A * hi + B) == 0:
                    right = (-B - sp.sqrt(B**2 + 4 * A * e)) / (2 * A)
                else:
                    windows.append(hi * (A * hi + B))
                if A != 0:
                    windows.append(-(B**2) / (4 * A))
                all_segments.append((su, sv, lo, hi, A, B))
                if (
                    not budget.allow_expensive_exact
                    and len(all_segments) * len(num.terms()) > budget.hypersurface.max_pieces
                ):
                    raise StrategyBudgetExceeded("polygon product moments exceed piece budget")
                for powers, c in num.terms():
                    p, q = (a - b for a, b in zip(powers, den_powers, strict=True))
                    if (
                        not budget.allow_expensive_exact
                        and max(abs(p), abs(q)) > budget.hypersurface.max_polynomial_degree
                    ):
                        return None
                    moment = linear_segment_moment(p, q, A, B, left, right, e, assumptions)
                    if moment is None:
                        return None
                    total += (
                        c / den_coefficient * sp.Integer(su) ** p * sp.Integer(sv) ** q * moment
                    )
    if not occupied or not windows:
        return None
    if not all(_true(sp.Q.positive(w) & sp.Q.finite(w), assumptions) for w in windows):
        return None
    return sp.expand(total.subs(e, epsilon / sp.Abs(coefficient))), dict(
        realization="clipped_product_cutoff",
        backend="exact_polygon_product_moments",
        polygon_segments=tuple(all_segments),
        orthants=occupied,
        moment_pieces=len(all_segments) * len(num.terms()),
        coverage_complete=True,
        cutoff_valid_for=epsilon < sp.Abs(coefficient) * sp.Min(*windows),
        assumptions=assumptions,
    )
