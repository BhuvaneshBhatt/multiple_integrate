"""Certified triangular polynomial maps and two-dimensional clipped product cutoffs."""

from __future__ import annotations

import sympy as sp
from sympy.core.relational import Relational

from .assumptions import _ask, _normalize_assumptions
from .policy import _CURRENT_POLICY, StrategyBudgetExceeded
from .region_types import BoxRegion


def _true(predicate, assumptions):
    return _ask(predicate, _normalize_assumptions(assumptions)) is True


def _rational_primitive(expr, t):
    primitive = sp.integrate(sp.apart(expr, t), t)
    primitive = primitive.xreplace({sp.log(t - 1): sp.log(1 - t)})
    if primitive.has(sp.Integral) or sp.cancel(sp.diff(primitive, t) - expr) != 0:
        return None
    return primitive


def triangle_moment(p, q, e):
    """Integral of u**p*v**q on u,v>0, u+v<1, u*v>e, 0<e<1/4."""
    t = sp.Dummy("triangle_coordinate", positive=True)
    lower = (1 - sp.sqrt(1 - 4 * e)) / 2
    upper = 1 - lower
    if q != -1:
        a = _rational_primitive(t**p * (1 - t) ** (q + 1), t)
        b = _rational_primitive(t ** (p - q - 1), t)
        if a is None or b is None:
            return None
        primitive = (a - e ** (q + 1) * b) / (q + 1)
    else:
        if p == -1:
            primitive = sp.log(t) ** 2 / 2 - sp.polylog(2, t) - sp.log(e) * sp.log(t)
        else:
            r = _rational_primitive(t ** (p + 1) / (1 - t), t)
            if r is None:
                return None
            primitive = (
                t ** (p + 1) * (sp.log(t) / (p + 1) - 1 / (p + 1) ** 2)
                + t ** (p + 1) * sp.log(1 - t) / (p + 1)
                + r / (p + 1)
                - sp.log(e) * t ** (p + 1) / (p + 1)
            )
    result = primitive.subs(t, upper) - primitive.subs(t, lower)
    result = result.xreplace(
        {sp.polylog(2, upper): sp.pi**2 / 6 - sp.polylog(2, lower) - sp.log(lower) * sp.log(upper)}
    )
    return sp.expand(result)


def polygon_product_cutoff(expr, formula, variables, level, epsilon, assumptions=None):
    """Recognize axis-intercept triangles in each orthant of a clipped affine domain."""
    if len(variables) != 2:
        return None
    u, v = variables
    try:
        g = sp.Poly(level, u, v)
        if len(g.terms()) != 1 or g.monoms()[0] != (1, 1):
            return None
        coefficient = g.coeffs()[0]
        if not _true(
            sp.Q.real(coefficient) & sp.Q.nonzero(coefficient) & sp.Q.finite(coefficient),
            assumptions,
        ):
            return None
        halfspaces = []
        for atom in sp.And.make_args(formula):
            if atom is sp.true:
                continue
            if not isinstance(atom, Relational) or atom.rel_op not in ("<", "<=", ">", ">="):
                return None
            f = (
                sp.expand(atom.lhs - atom.rhs)
                if atom.rel_op in (">", ">=")
                else sp.expand(atom.rhs - atom.lhs)
            )
            poly = sp.Poly(f, u, v)
            if poly.total_degree() > 1:
                return None
            if not all(_true(sp.Q.real(c) & sp.Q.finite(c), assumptions) for c in poly.coeffs()):
                return None
            halfspaces.append((poly.nth(1, 0), poly.nth(0, 1), poly.nth(0, 0)))
        density = sp.cancel(expr)
        numerator, denominator = density.as_numer_denom()
        num = sp.Poly(numerator, u, v)
        den = sp.Poly(denominator, u, v)
        if len(den.terms()) != 1:
            return None
        den_powers, den_coefficient = den.terms()[0]
        if not _true(
            sp.Q.nonzero(den_coefficient) & sp.Q.finite(den_coefficient), assumptions
        ) or not all(_true(sp.Q.finite(c), assumptions) for c in num.coeffs()):
            return None
        triangles = []
        for su in (-1, 1):
            for sv in (-1, 1):
                # An orthant absent from the domain may be certified by a sign constraint.
                if any(
                    c == 0
                    and (
                        (b == 0 and _true(sp.Q.negative(a * su), assumptions))
                        or (a == 0 and _true(sp.Q.negative(b * sv), assumptions))
                    )
                    for a, b, c in halfspaces
                ):
                    continue
                candidate = None
                for a, b, c in halfspaces:
                    if _true(
                        sp.Q.negative(a * su) & sp.Q.negative(b * sv) & sp.Q.positive(c),
                        assumptions,
                    ):
                        L = -c / (a * su)
                        H = -c / (b * sv)
                        if all(
                            _true(sp.Q.nonnegative(z), assumptions)
                            for A, B, C in halfspaces
                            for z in (C, C + A * su * L, C + B * sv * H)
                        ):
                            candidate = (su, sv, sp.simplify(L), sp.simplify(H))
                            break
                if candidate is None:
                    from .polygon_moments import clipped_polygon_cutoff

                    return clipped_polygon_cutoff(
                        num,
                        den_powers,
                        den_coefficient,
                        halfspaces,
                        coefficient,
                        variables,
                        epsilon,
                        assumptions,
                    )
                triangles.append(candidate)
        budget = _CURRENT_POLICY.get()
        if (
            not budget.allow_expensive_exact
            and len(triangles) * len(num.terms()) > budget.hypersurface.max_pieces
        ):
            raise StrategyBudgetExceeded("clipped product cutoff exceeds triangle/moment budget")
        if not triangles:
            return None
        total = sp.S.Zero
        thresholds = []
        for su, sv, L, H in triangles:
            scale = sp.Abs(coefficient) * L * H
            thresholds.append(scale / 4)
            for powers, c in num.terms():
                p, q = (a - b for a, b in zip(powers, den_powers, strict=True))
                if (
                    not budget.allow_expensive_exact
                    and max(abs(p), abs(q)) > budget.hypersurface.max_polynomial_degree
                ):
                    return None
                moment = triangle_moment(p, q, epsilon / scale)
                if moment is None:
                    return None
                total += (
                    c
                    / den_coefficient
                    * sp.Integer(su) ** p
                    * sp.Integer(sv) ** q
                    * L ** (p + 1)
                    * H ** (q + 1)
                    * moment
                )
        return sp.expand(total), dict(
            realization="clipped_product_cutoff",
            backend="exact_triangle_product_moments",
            triangles=tuple(triangles),
            orthants=len(triangles),
            moment_pieces=len(triangles) * len(num.terms()),
            coverage_complete=True,
            cutoff_valid_for=epsilon < sp.Min(*thresholds),
            source_density=expr,
            source_level=level,
            source_formula=formula,
            source_variables=variables,
            assumptions=_normalize_assumptions(assumptions).original,
        )
    except (sp.PolynomialError, ValueError, TypeError):
        return None


def _graph_product_cutoff(expr, region, level, epsilon, variables, assumptions=None):
    """Flatten y=P(x) by a global triangular polynomial map, with proved Jacobian."""
    from .adapters.semialg import semialgebraic_domain
    from .singular_geometry import product_level_cutoff
    from .symbolic_geometry import symbolic_boxes

    formula, variables = semialgebraic_domain(region, variables)
    if len(variables) != 2:
        return None
    x, y = variables
    try:
        _, factors = sp.factor_list(level, x, y)
        graph = []
        xroots = []
        for factor, multiplicity in factors:
            poly = sp.Poly(factor, y)
            if poly.degree() == 1 and not poly.nth(1).has(x):
                graph.append((factor, poly.nth(1), poly.nth(0)))
            elif not factor.has(y) and sp.Poly(factor, x).degree() == 1:
                xroots.append(-factor.subs(x, 0) / sp.diff(factor, x))
            else:
                return None
        if len(graph) != 1 or len(xroots) > 1:
            return None
        _, slope, offset = graph[0]
        if not _true(sp.Q.real(slope) & sp.Q.finite(slope) & sp.Q.nonzero(slope), assumptions):
            return None
        curve = sp.Poly(-offset / slope, x)
        budget = _CURRENT_POLICY.get()
        if not budget.allow_expensive_exact and (
            curve.degree() > budget.hypersurface.max_polynomial_degree
            or len(curve.terms()) > budget.hypersurface.max_pieces
        ):
            return None
        if not all(_true(sp.Q.real(c) & sp.Q.finite(c), assumptions) for c in curve.coeffs()):
            return None
        center = xroots[0] if xroots else sp.S.Zero
        if not _true(sp.Q.real(center) & sp.Q.finite(center), assumptions):
            return None
        u, v = sp.Dummy("graph_base", real=True), sp.Dummy("graph_normal", real=True)
        mapping = {x: u + center, y: (v - offset.subs(x, u + center)) / slope}
        transformed = formula.subs(mapping, simultaneous=True).replace(
            lambda a: isinstance(a, Relational),
            lambda a: a.func(sp.expand(a.lhs), sp.expand(a.rhs)),
        )
        boxes = symbolic_boxes(transformed, (u, v), assumptions=assumptions)
        target_density = sp.cancel(expr.subs(mapping, simultaneous=True)) / sp.Abs(slope)
        target_level = sp.factor(level.subs(mapping, simultaneous=True))
        if boxes is None:
            result = polygon_product_cutoff(
                target_density, transformed, (u, v), target_level, epsilon, assumptions
            )
        else:
            if not boxes:
                return None
            results = []
            for entries in boxes:
                box = BoxRegion(tuple((w, i.lower, i.upper) for w, i in entries))
                part = product_level_cutoff(
                    target_density, box, target_level, epsilon, assumptions=assumptions
                )
                if part is None:
                    return None
                results.append(part)
            details = dict(results[0][1])
            details["mapped_box_pieces"] = len(results)
            details["cutoff_valid_for"] = sp.And(*(part[1]["cutoff_valid_for"] for part in results))
            details["orthants"] = sum(part[1]["orthants"] for part in results)
            result = (sp.Add(*(part[0] for part in results)), details)
        if result is None:
            return None
        value, details = result
        details.update(
            realization="polynomial_graph_cutoff",
            backend="exact_polynomial_graph_product_moments",
            source_formula=formula,
            source_variables=variables,
            source_density=expr,
            source_level=level,
            forward_map=tuple(mapping[w] for w in variables),
            graph_jacobian=1 / sp.Abs(slope),
            graph_degree=curve.degree(),
        )
        return value, details
    except (sp.PolynomialError, ValueError, TypeError):
        return None


def graph_product_cutoff(expr, region, level, epsilon, variables=None, assumptions=None):
    """Try both coordinate orders for a global polynomial graph cutoff map."""
    from .adapters.semialg import semialgebraic_domain

    formula, coordinates = semialgebraic_domain(region, variables)
    if len(coordinates) != 2:
        return None
    for order in (coordinates, tuple(reversed(coordinates))):
        result = _graph_product_cutoff(
            expr, formula, level, epsilon, assumptions=assumptions, variables=order
        )
        if result is not None:
            return result
    return None
