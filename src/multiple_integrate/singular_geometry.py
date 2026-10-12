"""Exact shared cutoffs for coordinate-plane singularities on centered boxes.

A product level set can have intersections or repeated defining factors. Its
|g| cutoff is evaluated directly, without assuming a regular implicit atlas.
"""

from __future__ import annotations

from collections import Counter

import sympy as sp

from .assumptions import _ask, _normalize_assumptions
from .policy import _CURRENT_POLICY, StrategyBudgetExceeded
from .region_types import BoxRegion
from .results import UnresolvedIntegration


def _exponential_simplex(rates, powers, length):
    """Integral over sum(m_i*t_i)<length of exp(-sum(rate_i*t_i)).

    Its Laplace transform in length is 1/s * product(1/(rate_i+m_i*s)).
    Repeated rational poles are inverted exactly, including logarithmic cases.
    """
    s = sp.Dummy("laplace", real=True)
    transform = 1 / s
    poles = Counter({sp.Integer(0): 1})
    for rate, power in zip(rates, powers, strict=True):
        transform /= rate + power * s
        poles[-sp.Rational(rate, power)] += 1
    result = sp.Integer(0)
    for pole, multiplicity in poles.items():
        local = sp.cancel(transform * (s - pole) ** multiplicity)
        for k in range(1, multiplicity + 1):
            coefficient = sp.diff(local, s, multiplicity - k).subs(s, pole) / sp.factorial(
                multiplicity - k
            )
            result += coefficient * length ** (k - 1) * sp.exp(pole * length) / sp.factorial(k - 1)
    return sp.expand(result)


def _proved(predicate, context):
    return _ask(predicate, context) is True


def _affine_product_cutoff(expr, region, polynomial, epsilon, variables, assumptions):
    """Lower a certified affine image of a box to the coordinate-product route."""
    from sympy.core.relational import Relational

    from .adapters.semialg import semialgebraic_domain
    from .symbolic_geometry import symbolic_boxes

    context = _normalize_assumptions(assumptions)
    formula, variables = semialgebraic_domain(region, variables)
    try:
        _, factors = sp.factor_list(polynomial, *variables)
        forms = []
        for factor, _ in factors:
            poly = sp.Poly(factor, *variables)
            if poly.total_degree() != 1:
                return None
            forms.append(factor)
        if not forms or len(forms) > len(variables):
            return None
        rows = [tuple(sp.diff(f, v) for v in variables) for f in forms]
        if sp.Matrix(rows).rank() != len(rows):
            return None
        candidates = [sp.expand(a.lhs - a.rhs) for a in formula.atoms(Relational)] + list(variables)
        for candidate in candidates:
            if len(forms) == len(variables):
                break
            poly = sp.Poly(candidate, *variables)
            if poly.total_degree() != 1:
                continue
            row = tuple(sp.diff(candidate, v) for v in variables)
            if sp.Matrix([*rows, row]).rank() > len(rows):
                rows.append(row)
                forms.append(candidate)
        if len(forms) != len(variables):
            return None
        matrix = sp.ImmutableMatrix(rows)
        determinant = sp.simplify(matrix.det())
        constants = sp.ImmutableMatrix([f.subs(dict.fromkeys(variables, 0)) for f in forms])
        if not all(
            _proved(sp.Q.real(c) & sp.Q.finite(c), context) for c in (*matrix, *constants)
        ) or not _proved(sp.Q.nonzero(determinant), context):
            return None
        targets = tuple(sp.Dummy(f"affine_normal{i}", real=True) for i in range(len(variables)))
        inverse = matrix.inv()
        forward = inverse * (sp.ImmutableMatrix(targets) - constants)
        mapping = dict(zip(variables, forward, strict=True))
        transformed_formula = formula.subs(mapping, simultaneous=True).replace(
            lambda a: isinstance(a, Relational),
            lambda a: a.func(sp.expand(a.lhs), sp.expand(a.rhs)),
        )
        boxes = symbolic_boxes(transformed_formula, targets, assumptions=assumptions)
        density = sp.cancel(expr.subs(mapping, simultaneous=True)) / sp.Abs(determinant)
        level = sp.factor(polynomial.subs(mapping, simultaneous=True))
        if boxes is None:
            from .singular_maps import polygon_product_cutoff

            result = polygon_product_cutoff(
                density, transformed_formula, targets, level, epsilon, assumptions
            )
        elif len(boxes) == 1:
            box = BoxRegion(tuple((v, i.lower, i.upper) for v, i in boxes[0]))
            result = product_level_cutoff(density, box, level, epsilon, assumptions=assumptions)
        else:
            return None
        if result is None:
            return None
        value, details = result
        details.update(
            source_density=sp.sympify(expr),
            realization="affine_product_cutoff",
            affine_matrix=matrix,
            inverse_matrix=sp.ImmutableMatrix(inverse),
            affine_jacobian=1 / sp.Abs(determinant),
            source_variables=variables,
            target_variables=targets,
            forward_map=tuple(forward),
            source_formula=formula,
            transformed_formula=transformed_formula,
            source_level=sp.sympify(polynomial),
            transformed_level=level,
        )
        return value, details
    except (sp.PolynomialError, ValueError, ZeroDivisionError):
        return None


def product_level_cutoff(expr, region, polynomial, epsilon, *, assumptions=None, variables=None):
    """Exact shared cutoffs on asymmetric boxes and certified affine box images.

    All finite rational poles must lie on the excluded coordinate planes. Each
    signed interval uses exact product moments and inclusion-exclusion; no odd
    cancellation is assumed for asymmetric domains.
    """
    if not isinstance(region, BoxRegion):
        if variables is None:
            variables = tuple(getattr(region, "variables", ()))
        if not variables:
            return None
        # Resolve proved tangent/contact charts before attempting generic affine
        # or graph decompositions.  The general routes may perform expensive
        # geometric queries (or decline under a budget) even when an exact
        # shared-|g| cutoff is already available in the original coordinates.
        # The chart methods preserve both the regulator and its scale.
        from .tangent_cutoff import tangent_graph_cutoff

        tangent = tangent_graph_cutoff(expr, region, polynomial, epsilon, variables, assumptions)
        if tangent is not None:
            return tangent
        from .contact_charts import polynomial_contact_cutoff

        contact = polynomial_contact_cutoff(
            expr, region, polynomial, epsilon, variables, assumptions
        )
        if contact is not None:
            return contact
        affine = _affine_product_cutoff(
            sp.sympify(expr), region, polynomial, epsilon, variables, assumptions
        )
        if affine is not None:
            return affine
        from .singular_maps import graph_product_cutoff

        graph = graph_product_cutoff(
            sp.sympify(expr), region, polynomial, epsilon, variables, assumptions
        )
        if graph is not None:
            return graph
        from .triangular_intersections import triangular_product_cutoff

        return triangular_product_cutoff(
            sp.sympify(expr), region, polynomial, epsilon, variables, assumptions
        )
    context = _normalize_assumptions(assumptions)
    variables = tuple(region.variables)
    budget = _CURRENT_POLICY.get().hypersurface
    if not _CURRENT_POLICY.get().allow_expensive_exact and len(variables) > budget.max_variables:
        raise StrategyBudgetExceeded("singular product cutoff exceeds variable budget")
    ranges = tuple(tuple(map(sp.sympify, r)) for r in region.ranges)
    roots = {}
    try:
        _, factors = sp.factor_list(polynomial, *variables)
        for factor, _ in factors:
            active = [v for v in variables if factor.has(v)]
            if len(active) != 1 or sp.Poly(factor, active[0]).degree() != 1:
                # Prefer exact regulator-preserving contact charts to generic
                # transformations; retain the latter for unmatched families.
                from .tangent_cutoff import tangent_graph_cutoff

                tangent = tangent_graph_cutoff(
                    expr, region, polynomial, epsilon, variables, assumptions
                )
                if tangent is not None:
                    return tangent
                from .contact_charts import polynomial_contact_cutoff

                contact = polynomial_contact_cutoff(
                    expr, region, polynomial, epsilon, variables, assumptions
                )
                if contact is not None:
                    return contact
                affine = _affine_product_cutoff(
                    sp.sympify(expr), region, polynomial, epsilon, variables, assumptions
                )
                if affine is not None:
                    return affine
                from .singular_maps import graph_product_cutoff

                graph = graph_product_cutoff(
                    sp.sympify(expr), region, polynomial, epsilon, variables, assumptions
                )
                if graph is not None:
                    return graph
                from .triangular_intersections import triangular_product_cutoff

                return triangular_product_cutoff(
                    sp.sympify(expr), region, polynomial, epsilon, variables, assumptions
                )
            v = active[0]
            root = sp.simplify(-factor.subs(v, 0) / sp.diff(factor, v))
            if v in roots and sp.simplify(root - roots[v]) != 0:
                return None
            roots[v] = root
        centers = tuple(roots.get(v, sp.simplify((lo + hi) / 2)) for v, lo, hi in ranges)
        if not all(_proved(sp.Q.real(c) & sp.Q.finite(c), context) for c in centers):
            return None
        if not all(
            _proved(
                sp.Q.real(lo)
                & sp.Q.real(hi)
                & sp.Q.finite(lo)
                & sp.Q.finite(hi)
                & sp.Q.positive(hi - lo),
                context,
            )
            for _, lo, hi in ranges
        ):
            return None
        locals_ = tuple(sp.Dummy("normal_coordinate", real=True) for _ in variables)
        mapping = {v: u + c for v, u, c in zip(variables, locals_, centers, strict=True)}
        level = sp.Poly(sp.expand(polynomial.subs(mapping, simultaneous=True)), *locals_)
        if len(level.terms()) != 1:
            return None
        powers, coefficient = level.terms()[0]
        if sum(powers) <= 1 or not _proved(
            sp.Q.real(coefficient) & sp.Q.finite(coefficient) & sp.Q.nonzero(coefficient), context
        ):
            return None
        if (
            not _CURRENT_POLICY.get().allow_expensive_exact
            and sum(powers) > budget.max_polynomial_degree
        ):
            raise StrategyBudgetExceeded("singular product cutoff exceeds polynomial degree budget")
        density = sp.cancel(expr.subs(mapping, simultaneous=True))
        numerator, denominator = density.as_numer_denom()
        den = sp.Poly(denominator, *locals_)
        num = sp.Poly(numerator, *locals_)
        if len(den.terms()) != 1:
            return None
        den_powers, den_coefficient = den.terms()[0]
        if not _proved(sp.Q.nonzero(den_coefficient), context) or not all(
            _proved(sp.Q.finite(c), context) for c in (den_coefficient, *num.coeffs())
        ):
            raise UnresolvedIntegration(
                "singular product density coefficients must be certified finite with a nonzero denominator"
            )
        if any(d and not m for d, m in zip(den_powers, powers, strict=True)):
            return None
        signed = []
        for (_, lo, hi), center in zip(ranges, centers, strict=True):
            lo, hi = sp.simplify(lo - center), sp.simplify(hi - center)
            intervals = []
            if _proved(sp.Q.negative(lo), context):
                if _proved(sp.Q.nonnegative(hi), context):
                    intervals.append((-1, sp.S.Zero, -lo))
                elif _proved(sp.Q.negative(hi), context):
                    intervals.append((-1, -hi, -lo))
                else:
                    return None
            elif not _proved(sp.Q.nonnegative(lo), context):
                return None
            if _proved(sp.Q.positive(hi), context):
                intervals.append(
                    (1, sp.S.Zero if _proved(sp.Q.nonpositive(lo), context) else lo, hi)
                )
            elif not _proved(sp.Q.nonpositive(hi), context):
                return None
            signed.append(intervals)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    from itertools import product

    orthants = list(product(*signed))
    work_count = sum(
        sp.prod(1 + (low != 0) for _, low, _ in orthant) for orthant in orthants
    ) * len(num.terms())
    if not _CURRENT_POLICY.get().allow_expensive_exact and work_count > budget.max_pieces:
        raise StrategyBudgetExceeded("singular product cutoff exceeds density/orthant piece budget")
    total = sp.S.Zero
    thresholds = []
    for orthant in orthants:
        edges = [((high, 1),) if low == 0 else ((high, 1), (low, -1)) for _, low, high in orthant]
        for endpoints in product(*edges):
            upper = tuple(edge[0] for edge in endpoints)
            inclusion_sign = sp.prod(edge[1] for edge in endpoints)
            threshold = sp.Abs(coefficient) * sp.prod(
                w**m for w, m in zip(upper, powers, strict=True)
            )
            thresholds.append(threshold)
            length = (
                sp.log(sp.Abs(coefficient))
                + sum(m * sp.log(w) for w, m in zip(upper, powers, strict=True))
                - sp.log(epsilon)
            )
            for exponents, c in num.terms():
                exponents = tuple(p - d for p, d in zip(exponents, den_powers, strict=True))
                factor = inclusion_sign * c / den_coefficient
                rates, active_powers = [], []
                for p, m, width, (sign, _, _) in zip(
                    exponents, powers, upper, orthant, strict=True
                ):
                    factor *= sp.Integer(sign) ** p * width ** (p + 1)
                    if m:
                        rates.append(p + 1)
                        active_powers.append(m)
                    else:
                        factor /= p + 1
                total += factor * _exponential_simplex(rates, active_powers, length)
    details = {
        "source_density": sp.sympify(expr),
        "source_level": sp.sympify(polynomial),
        "source_ranges": ranges,
        "assumptions": context.original,
        "coverage_complete": True,
        "realization": "coordinate_product_cutoff",
        "geometry_kind": "intersecting_or_repeated_coordinate_planes",
        "centers": centers,
        "widths": tuple(sp.simplify((hi - lo) / 2) for _, lo, hi in ranges),
        "signed_intervals": tuple(tuple(group) for group in signed),
        "level_powers": powers,
        "orthants": len(orthants),
        "moment_pieces": int(work_count),
        "cutoff_valid_for": sp.And(epsilon > 0, epsilon < sp.Min(*thresholds)),
        "backend": "exact_product_cutoff_moments",
    }
    return sp.expand(total), details
