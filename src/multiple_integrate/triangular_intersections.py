"""Global triangular polynomial charts for normal-crossing singular products."""

from __future__ import annotations

from itertools import permutations

import sympy as sp
from sympy.core.relational import Relational

from .adapters.semialg import semialgebraic_domain
from .assumptions import _ask, _normalize_assumptions
from .cell_proofs import nonzero
from .policy import _normalize_policy
from .region_types import BoxRegion


def triangular_product_cutoff(expr, region, level, epsilon, variables, assumptions=None):
    """Flatten independent polynomial graph factors in arbitrary bounded dimension.

    Each target coordinate is linear in its own source coordinate with a proved
    constant nonzero slope, and polynomial in previous coordinates. This gives
    a global polynomial inverse, constant absolute Jacobian and exact shared
    product cutoff, including repeated factors and singular intersections.
    """
    from .singular_geometry import product_level_cutoff
    from .symbolic_geometry import symbolic_boxes

    variables = tuple(variables)
    budget = _normalize_policy(None)
    if len(variables) < 3 or (
        not budget.allow_expensive_exact and len(variables) > budget.hypersurface.max_variables
    ):
        return None
    context = _normalize_assumptions(assumptions)
    try:
        coefficient, factor_list = sp.factor_list(level, *variables)
        factors = [factor for factor, _ in factor_list]
        if not factors or len(factors) > len(variables) or not nonzero(coefficient, context):
            return None
        if not budget.allow_expensive_exact and (
            sp.Poly(level, *variables).total_degree() > budget.hypersurface.max_polynomial_degree
        ):
            return None
        formula, _ = semialgebraic_domain(region, variables)
        for attempt, order in enumerate(permutations(variables)):
            if not budget.allow_expensive_exact and attempt >= budget.hypersurface.max_pieces:
                return None
            remaining = list(factors)
            targets = tuple(sp.Dummy("triangular_normal", real=True) for _ in order)
            mapping = {}
            forms = []
            jacobian = sp.S.One
            for v, target in zip(order, targets, strict=True):
                available = set(mapping) | {v}
                eligible = []
                for factor in remaining:
                    polynomial = sp.Poly(factor, v)
                    if (
                        factor.free_symbols & set(variables) <= available
                        and polynomial.degree() == 1
                        and not polynomial.nth(1).has(*variables)
                    ):
                        eligible.append(factor)
                if len(eligible) > 1:
                    break
                if not eligible:
                    # A later factor can still depend on this free coordinate.
                    form = v
                else:
                    form = eligible[0]
                    remaining.remove(form)
                polynomial = sp.Poly(form, v)
                slope, offset = polynomial.nth(1), polynomial.nth(0)
                if not nonzero(slope, context) or not all(
                    _ask(sp.Q.real(c) & sp.Q.finite(c), context) is True
                    for c in polynomial.coeffs()
                    if not c.has(*variables)
                ):
                    break
                offset_poly = sp.Poly(offset, *order)
                if not all(
                    _ask(sp.Q.real(c) & sp.Q.finite(c), context) is True
                    for c in offset_poly.coeffs()
                ):
                    break
                mapping[v] = sp.expand((target - offset.subs(mapping, simultaneous=True)) / slope)
                jacobian /= sp.Abs(slope)
                forms.append(form)
            else:
                if remaining:
                    continue
                transformed = formula.subs(mapping, simultaneous=True).replace(
                    lambda atom: isinstance(atom, Relational),
                    lambda atom: atom.func(sp.expand(atom.lhs), sp.expand(atom.rhs)),
                )
                boxes = symbolic_boxes(transformed, targets, assumptions=assumptions)
                if not boxes:
                    continue
                density = sp.cancel(expr.subs(mapping, simultaneous=True)) * jacobian
                target_level = sp.factor(level.subs(mapping, simultaneous=True))
                # A successful chart must actually yield a coordinate monomial.
                target_poly = sp.Poly(target_level, *targets)
                if len(target_poly.terms()) != 1:
                    continue
                parts = []
                for box in boxes:
                    native = BoxRegion(tuple((v, i.lower, i.upper) for v, i in box))
                    part = product_level_cutoff(
                        density, native, target_level, epsilon, assumptions=assumptions
                    )
                    if part is None:
                        break
                    parts.append(part)
                else:
                    details = dict(parts[0][1])
                    details.update(
                        realization="triangular_polynomial_cutoff",
                        backend="exact_triangular_product_moments",
                        source_formula=formula,
                        source_variables=variables,
                        source_density=expr,
                        source_level=level,
                        coordinate_order=order,
                        target_variables=targets,
                        defining_forms=tuple(forms),
                        forward_map=tuple(mapping[v] for v in variables),
                        triangular_jacobian=jacobian,
                        transformed_formula=transformed,
                        transformed_level=target_level,
                        assumptions=context.original,
                        mapped_box_pieces=len(parts),
                        orthants=sum(p[1]["orthants"] for p in parts),
                        cutoff_valid_for=sp.And(*(p[1]["cutoff_valid_for"] for p in parts)),
                    )
                    return sp.Add(*(p[0] for p in parts)), details
        return None
    except (sp.PolynomialError, ValueError, TypeError, ZeroDivisionError):
        return None
