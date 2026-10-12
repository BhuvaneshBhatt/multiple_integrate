"""Weighted blowup cutoffs for selected tangent polynomial graph pairs."""

import sympy as sp
from sympy.core.relational import Relational

from .adapters.semialg import semialgebraic_domain
from .assumptions import _ask, _normalize_assumptions
from .fast_rules import _polynomial
from .policy import _normalize_policy
from .singular_maps import _rational_primitive


def _log_angular_primitive(first, t):
    """Primitive of F(t)*(1/t-1/(1-t)) with only endpoint rational poles."""
    left, right = sp.log(t), sp.log(1 - t)
    expanded = sp.expand(first)
    a, b = expanded.coeff(left), expanded.coeff(right)
    if a.has(t) or b.has(t):
        return None
    rational = sp.cancel(expanded - a * left - b * right)
    if not rational.is_rational_function(t):
        return None
    primitive = _rational_primitive(rational * (1 / t - 1 / (1 - t)), t)
    if primitive is None:
        return None
    return (
        primitive
        + a * (left**2 / 2 - sp.polylog(2, 1 - t))
        + b * (-sp.polylog(2, t) + right**2 / 2)
    )


def tangent_graph_cutoff(expr, region, level, epsilon, variables, assumptions=None):
    """Shared |K*y*(y-c*x**m)| cutoff on 0<x<A, 0<y<c*x**m.

    The chart x=u, y=c*u**m*t has a proved inverse for u>0.
    Integer radial orders give rational or logarithmic/dilogarithmic angular primitives.
    Other implicit contacts require a proved lowering chart.
    """
    variables = tuple(variables)
    if len(variables) != 2:
        return None
    budget = _normalize_policy(None)
    context = _normalize_assumptions(assumptions)

    def proved(predicate):
        return _ask(predicate, context) is True

    formula, _ = semialgebraic_domain(region, variables)
    for x, y in (variables, tuple(reversed(variables))):
        upper = c = m = None
        lower_x = lower_y = False
        valid = True
        for atom in sp.And.make_args(formula):
            if not isinstance(atom, Relational) or atom.rel_op not in ("<", "<=", ">", ">="):
                valid = False
                break
            positive = atom.lhs - atom.rhs if atom.rel_op in (">", ">=") else atom.rhs - atom.lhs
            poly = _polynomial(positive, (x, y))
            if poly is None or not all(
                proved(sp.Q.real(a) & sp.Q.finite(a)) for a in poly.coeffs()
            ):
                valid = False
                break
            if poly.total_degree() == 1 and not positive.has(y):
                slope, offset = poly.nth(1, 0), poly.nth(0, 0)
                if offset == 0 and proved(sp.Q.positive(slope)):
                    lower_x = True
                elif proved(sp.Q.negative(slope)):
                    bound = sp.cancel(-offset / slope)
                    if upper is not None and sp.simplify(upper - bound) != 0:
                        valid = False
                        break
                    upper = bound
                else:
                    valid = False
                    break
            elif poly.monoms() == [(0, 1)] and proved(sp.Q.positive(poly.nth(0, 1))):
                lower_y = True
            elif len(poly.terms()) == 2 and proved(sp.Q.negative(poly.nth(0, 1))):
                degree = next((p for (p, q), _ in poly.terms() if p >= 2 and q == 0), None)
                if degree is None:
                    valid = False
                    break
                candidate = sp.cancel(-poly.nth(degree, 0) / poly.nth(0, 1))
                if m is not None and (m != degree or sp.simplify(c - candidate) != 0):
                    valid = False
                    break
                m, c = degree, candidate
            else:
                valid = False
                break
        if not valid or not lower_x or not lower_y or upper is None or m is None:
            continue
        if not proved(
            sp.Q.positive(upper) & sp.Q.finite(upper) & sp.Q.positive(c) & sp.Q.finite(c)
        ):
            continue
        if not budget.allow_expensive_exact and m > budget.hypersurface.max_polynomial_degree:
            continue
        level_polynomial = _polynomial(level, (x, y))
        if level_polynomial is None:
            continue
        coefficient = sp.cancel(level / (y * (y - c * x**m)))
        if coefficient.has(x, y) or not proved(
            sp.Q.real(coefficient) & sp.Q.finite(coefficient) & sp.Q.nonzero(coefficient)
        ):
            continue
        u = sp.Dummy("tangent_radial", positive=True)
        t = sp.Dummy("tangent_angular", positive=True)
        mapping = {x: u, y: c * u**m * t}
        source_num, source_den = sp.sympify(expr).as_numer_denom()
        if _polynomial(source_num, (x, y)) is None or _polynomial(source_den, (x, y)) is None:
            continue
        density = sp.cancel(sp.sympify(expr).subs(mapping, simultaneous=True) * c * u**m)
        numerator, denominator = density.as_numer_denom()
        num, den = _polynomial(numerator, (u, t)), _polynomial(denominator, (u, t))
        if (
            num is None
            or den is None
            or not all(_finite_coefficient(a, context) for a in num.coeffs())
        ):
            continue
        try:
            constant, factors = sp.factor_list(denominator, u, t)
        except (sp.PolynomialError, ValueError):
            continue
        powers = [0, 0, 0]
        for factor, multiplicity in factors:
            for i, reference in enumerate((u, t, 1 - t)):
                ratio = sp.cancel(factor / reference)
                if not ratio.has(u, t) and proved(sp.Q.nonzero(ratio) & sp.Q.finite(ratio)):
                    constant *= ratio**multiplicity
                    powers[i] += multiplicity
                    break
            else:
                valid = False
                break
        if not valid or not proved(sp.Q.finite(constant) & sp.Q.nonzero(constant)):
            continue
        regulator_scale = sp.Abs(coefficient) * c**2
        delta = epsilon / (regulator_scale * upper ** (2 * m))
        lower = (1 - sp.sqrt(1 - 4 * delta)) / 2
        higher = 1 - lower
        total = sp.S.Zero
        logarithmic_terms = 0
        for (p, q), amplitude in num.terms():
            radial = p - powers[0]
            angular = q - powers[1]
            other = -powers[2]
            order = sp.Rational(radial + 1, 2 * m)
            if not order.is_Integer or (
                not budget.allow_expensive_exact
                and max(abs(angular), abs(other), abs(angular - order), abs(other - order))
                > budget.hypersurface.max_polynomial_degree
            ):
                valid = False
                break
            first = _rational_primitive(t**angular * (1 - t) ** other, t)
            if first is None:
                valid = False
                break
            if order == 0:
                log_primitive = _log_angular_primitive(first, t)
                if log_primitive is None:
                    valid = False
                    break
                total += (
                    amplitude
                    / constant
                    / (2 * m)
                    * (log_primitive.subs(t, lower) - log_primitive.subs(t, higher))
                )
                logarithmic_terms += 1
                continue
            second = _rational_primitive(t ** (angular - order) * (1 - t) ** (other - order), t)
            if first is None or second is None:
                valid = False
                break
            primitive = (
                upper ** (radial + 1) * first - (epsilon / regulator_scale) ** order * second
            ) / (radial + 1)
            total += amplitude / constant * (primitive.subs(t, higher) - primitive.subs(t, lower))
        if not valid:
            continue
        total = total.xreplace(
            {
                sp.polylog(2, higher): sp.pi**2 / 6
                - sp.polylog(2, lower)
                - sp.log(lower) * sp.log(higher)
            }
        )
        target_level = sp.factor(level.subs(mapping, simultaneous=True))
        return sp.expand(total), dict(
            realization="tangent_graph_cutoff",
            backend="exact_weighted_blowup_moments",
            source_formula=formula,
            source_variables=variables,
            source_density=sp.sympify(expr),
            source_level=level,
            target_variables=(u, t),
            forward_map=tuple(mapping[v] for v in variables),
            inverse_map=(x, y / (c * x**m)),
            chart_jacobian=c * u**m,
            chart_valid_for=sp.And(u > 0, t > 0, t < 1),
            transformed_level=target_level,
            transformed_cutoff=sp.Abs(target_level) > epsilon,
            regulator_scale=regulator_scale,
            tangent_order=m,
            logarithmic_radial_terms=logarithmic_terms,
            coverage_complete=True,
            orthants=1,
            coverage_basis="invertible weighted blowup on the requested horn; excluded vertex has g=0",
            cutoff_valid_for=epsilon < regulator_scale * upper ** (2 * m) / 4,
            assumptions=context.original,
        )
    return None


def _finite_coefficient(coefficient, context):
    return _ask(sp.Q.finite(coefficient), context) is True
