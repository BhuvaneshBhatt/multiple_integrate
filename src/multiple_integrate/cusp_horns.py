"""Exact cusp-horn blowup integrals with explicit local convergence thresholds."""

from __future__ import annotations

import sympy as sp
from sympy.core.relational import Relational

from .assumptions import _normalize_assumptions
from .cell_proofs import native_predicate_true as _true
from .cell_proofs import nonnegative
from .planner import record_convergence, record_coverage, record_strategy
from .policy import _CURRENT_POLICY


def try_cusp_horn(expr, formula, variables, assumptions=None):
    """P(x,y)/(x**m-y**n)**s on a weighted horn, n positive even."""
    if len(variables) != 2:
        return None
    for x, y in (tuple(variables), tuple(reversed(variables))):
        upper = None
        k = None
        lower_seen = False
        exponents = None
        valid = True
        for atom in sp.And.make_args(formula):
            if not isinstance(atom, Relational) or atom.rel_op not in ("<", "<=", ">", ">="):
                valid = False
                break
            f = (
                sp.expand(atom.lhs - atom.rhs)
                if atom.rel_op in (">", ">=")
                else sp.expand(atom.rhs - atom.lhs)
            )
            try:
                poly = sp.Poly(f, x, y)
            except sp.PolynomialError:
                valid = False
                break
            if not f.has(y) and poly.total_degree() == 1:
                slope = poly.nth(1, 0)
                constant = poly.nth(0, 0)
                if constant == 0 and _true(sp.Q.positive(slope), assumptions):
                    lower_seen = True
                elif _true(sp.Q.negative(slope), assumptions):
                    bound = -constant / slope
                    if upper is not None and sp.simplify(upper - bound) != 0:
                        valid = False
                        break
                    upper = sp.simplify(bound)
                else:
                    valid = False
                    break
            elif len(poly.monoms()) == 2 and all(p == 0 or q == 0 for p, q in poly.monoms()):
                m = next((p for p, q in poly.monoms() if p > 0 and q == 0), None)
                n = next((q for p, q in poly.monoms() if q > 0 and p == 0), None)
                if (
                    m is None
                    or n is None
                    or n % 2
                    or not _true(sp.Q.negative(poly.nth(0, n)), assumptions)
                ):
                    valid = False
                    break
                if exponents is not None and exponents != (m, n):
                    valid = False
                    break
                exponents = (m, n)
                candidate = -poly.nth(m, 0) / poly.nth(0, n)
                if k is not None and sp.simplify(k - candidate) != 0:
                    valid = False
                    break
                k = sp.simplify(candidate)
            else:
                valid = False
                break
        if (
            not valid
            or not lower_seen
            or upper is None
            or k is None
            or not _true(
                sp.Q.positive(upper)
                & sp.Q.finite(upper)
                & sp.Q.positive(k)
                & sp.Q.nonnegative(1 - k),
                assumptions,
            )
        ):
            continue
        m, n = exponents
        budget = _CURRENT_POLICY.get()
        if not budget.allow_expensive_exact and max(m, n) > budget.semialg.max_polynomial_degree:
            continue
        base = x**m - y**n
        power = None
        poly = None
        scale = sp.S.One
        for factor in sp.Mul.make_args(expr):
            if not factor.is_Pow or factor.exp.has(x, y):
                continue
            try:
                arg = sp.Poly(factor.base, x, y)
            except sp.PolynomialError:
                continue
            ratio = arg.nth(m, 0)
            if sp.expand(factor.base - ratio * base) != 0 or not _true(
                sp.Q.positive(ratio)
                & sp.Q.finite(ratio)
                & sp.Q.real(factor.exp)
                & sp.Q.finite(factor.exp),
                assumptions,
            ):
                continue
            try:
                poly = sp.Poly(sp.cancel(expr / factor), x, y)
            except sp.PolynomialError:
                continue
            scale = ratio**factor.exp
            power = -factor.exp
            break
        if (
            poly is None
            or power is None
            or not all(_true(sp.Q.finite(c), assumptions) for c in poly.coeffs())
        ):
            continue
        budget = _CURRENT_POLICY.get()
        if not budget.allow_expensive_exact and (
            poly.total_degree() > budget.semialg.max_polynomial_degree
            or len(poly.terms()) > budget.semialg.max_pieces
        ):
            continue
        # Strip exact defining-equation vanishing before testing angular poles.
        vanishing_order = 0
        while not poly.is_zero:
            quotient, remainder = poly.div(sp.Poly(base, x, y))
            if not remainder.is_zero:
                break
            poly = quotient
            vanishing_order += 1
            power -= 1
        angular_finite = _true(sp.Q.positive(1 - power), assumptions)
        boundary_excluded = nonnegative(1 - k, _normalize_assumptions(assumptions), strict=True)
        boundary_contact = _true(sp.Q.zero(1 - k), assumptions)
        thresholds = [
            1 + p + sp.Rational(m * (q + 1), n) - m * power for (p, q), c in poly.terms() if c != 0
        ]
        radial_finite = all(_true(sp.Q.positive(t), assumptions) for t in thresholds)
        if not radial_finite or not (boundary_excluded or angular_finite):
            positive_terms = all(
                q % 2 == 0 and _true(sp.Q.positive(c), assumptions) for (p, q), c in poly.terms()
            )
            negative_terms = all(
                q % 2 == 0 and _true(sp.Q.negative(c), assumptions) for (p, q), c in poly.terms()
            )
            proved_divergence = any(
                _true(sp.Q.nonpositive(t), assumptions) for t in thresholds
            ) or (boundary_contact and _true(sp.Q.nonnegative(power - 1), assumptions))
            if (positive_terms or negative_terms) and proved_divergence:
                coefficient = scale if positive_terms else -scale
                if _true(sp.Q.positive(coefficient), assumptions):
                    value = sp.oo
                elif _true(sp.Q.negative(coefficient), assumptions):
                    value = -sp.oo
                else:
                    continue
                record_strategy("cusp_horn")
                for kind in ("ordinary", "absolute"):
                    record_convergence(kind, False, "cusp_blowup_threshold")
                return value
            continue
        value = sp.S.Zero
        for (p, q), c in poly.terms():
            if q % 2:
                continue
            order = 1 + p + sp.Rational(m * (q + 1), n) - m * power
            value += (
                c
                * scale
                * upper**order
                / order
                * sp.Rational(2, n)
                * (
                    sp.beta(sp.Rational(q + 1, n), 1 - power)
                    if boundary_contact
                    else sp.betainc(sp.Rational(q + 1, n), 1 - power, 0, k)
                )
            )
        record_strategy("cusp_horn")
        record_coverage(
            "absolute_convergence",
            True,
            "cusp_horn",
            basis="triangular cusp blowup and all local monomial thresholds",
            parameterization=(x, x ** sp.Rational(m, n) * sp.Symbol("t", real=True)),
            thresholds=tuple(thresholds),
            angular_threshold=1 - power,
            defining_equation_vanishing_order=vanishing_order,
        )
        record_coverage(
            "integration_domain",
            True,
            "cusp_horn",
            basis="exact horn bounds; angular boundary poles are excluded or proved integrable",
        )
        return sp.simplify(value)
    return None
