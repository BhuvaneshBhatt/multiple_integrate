"""Exact polynomial moments on planar ellipsoid wedges with proved angular cells."""

import sympy as sp
from sympy.core.relational import Relational

from .assumptions import _ask, _normalize_assumptions
from .fast_rules import _polynomial
from .policy import _normalize_policy, _with_policy, check_semialg_budget
from .quadratic_radial import positive_quadratic
from .rational_cutoff import certified_order
from .results import (
    ConvergenceEvidence,
    CoverageEvidence,
    IntegrationResult,
    IntegrationStatus,
    UnresolvedIntegration,
)


@_with_policy
def integrate_quadratic_sector(
    f, formula, variables, *, assumptions=None, policy=None, measure_dimension="ambient"
):
    """Single planar SPD ellipsoid clipped by center-passing affine halfplanes.

    Angular boundaries must be exact algebraic constants; translated centers
    and positive symbolic radii are allowed. Offset and uncertain clips decline.
    """
    variables = tuple(variables)
    if (
        len(variables) != 2
        or not isinstance(formula, sp.And)
        or measure_dimension not in ("ambient", "intrinsic", "top", 2)
    ):
        return None
    budget = _normalize_policy(policy)
    context = _normalize_assumptions(assumptions)
    if not check_semialg_budget(formula, variables, budget)[0]:
        return None

    def proved(q):
        return _ask(q, context) is True

    density = _polynomial(sp.sympify(f), variables)
    if density is None or not all(proved(sp.Q.finite(c)) for c in density.coeffs()):
        return None
    data = None
    clips = []
    for atom in sp.And.make_args(formula):
        if not isinstance(atom, Relational) or atom.rel_op not in ("<", "<=", ">", ">="):
            return None
        q = atom.rhs - atom.lhs if atom.rel_op in ("<", "<=") else atom.lhs - atom.rhs
        poly = _polynomial(q, variables)
        if poly is None:
            return None
        if poly.total_degree() == 2:
            if data is not None:
                return None
            data = positive_quadratic(-q, variables, assumptions)
            if data is None:
                return None
        elif poly.total_degree() <= 1:
            clips.append((q, atom.rel_op in ("<", ">")))
        else:
            return None
    if data is None or not clips:
        return None
    matrix, center, minimum = data
    radius2 = -minimum
    if not proved(sp.Q.positive(radius2)):
        return None
    if not all(a.is_algebraic is True and a.is_real is True for a in matrix):
        return None
    transform = matrix.cholesky().T.inv()
    u, v = sp.symbols("_sector_u _sector_v", real=True, cls=sp.Dummy)
    mapping = {
        x: c + w for x, c, w in zip(variables, center, transform * sp.Matrix([u, v]), strict=True)
    }
    t = sp.Dummy("angular_tangent", real=True)
    angular = []
    roots = []
    for q, strict in clips:
        target = sp.expand(q.subs(mapping, simultaneous=True))
        poly = sp.Poly(target, u, v)
        if sp.simplify(poly.nth(0, 0)) != 0:
            return None
        a, b = poly.nth(1, 0), poly.nth(0, 1)
        if not all(c.is_algebraic is True and c.is_real is True for c in (a, b)):
            return None
        if a == b == 0:
            if strict:
                return None
            continue
        boundary = sp.Poly(a * (1 - t * t) + 2 * b * t, t)
        solutions = sp.roots(boundary.as_expr(), t)
        if sum(solutions.values()) != boundary.degree():
            return None
        for root in solutions:
            if root.is_real is True:
                roots.append(root)
            elif root.is_real is not False:
                return None
        angular.append(boundary.as_expr())
    try:
        ordered = certified_order(roots)
    except UnresolvedIntegration:
        return None
    if not budget.allow_expensive_exact and len(ordered) + 1 > budget.semialg.max_pieces:
        return None
    intervals = []
    endpoints = (-sp.oo, *ordered, sp.oo)
    for lo, hi in zip(endpoints, endpoints[1:], strict=False):
        sample = (
            hi - 1
            if lo == -sp.oo and hi != sp.oo
            else lo + 1
            if hi == sp.oo and lo != -sp.oo
            else sp.S.Zero
            if lo == -sp.oo
            else (lo + hi) / 2
        )
        signs = [sp.simplify(g.subs(t, sample)) for g in angular]
        if any(z.is_negative is True for z in signs):
            continue
        if not all(z.is_nonnegative is True for z in signs):
            return None
        intervals.append((lo, hi))
    if not intervals and measure_dimension not in ("ambient", 2):
        return None
    target = _polynomial(sp.sympify(f).subs(mapping, simultaneous=True), (u, v))
    if target is None:
        return None
    value = sp.S.Zero
    for (p, q), c in target.terms():
        integrand = 2 * (1 - t * t) ** p * (2 * t) ** q / (1 + t * t) ** (p + q + 1)
        primitive = sp.integrate(integrand, t)
        if primitive.has(sp.Integral) or sp.cancel(sp.diff(primitive, t) - integrand) != 0:
            return None

        def at(bound, primitive=primitive):
            return (
                sp.limit(primitive, t, bound)
                if bound in (-sp.oo, sp.oo)
                else primitive.subs(t, bound)
            )

        moment = sum((at(hi) - at(lo) for lo, hi in intervals), sp.S.Zero)
        value += (
            c * radius2 ** sp.Rational(p + q + 2, 2) * moment / (p + q + 2) / sp.sqrt(matrix.det())
        )
    return IntegrationResult(
        sp.simplify(value),
        IntegrationStatus.EXACT,
        "quadratic_sector",
        convergent=True,
        complete=True,
        exact=True,
        assumptions=context.predicate_expr,
        coverage=(
            CoverageEvidence(
                "integration_domain",
                True,
                "quadratic_sector",
                (
                    (
                        "basis",
                        "exact ordered angular roots and constant signs on each root-free interval",
                    ),
                    ("angular_intervals", tuple(intervals)),
                ),
            ),
        ),
        convergence_evidence=tuple(
            ConvergenceEvidence(k, True, "compact_polynomial_sector")
            for k in ("ordinary", "absolute")
        ),
        diagnostics={"angular_intervals": tuple(intervals), "whitening_map": transform},
    )
