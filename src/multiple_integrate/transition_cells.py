"""Budgeted polynomial event partitions without sampling parameter fibers."""

from __future__ import annotations

from functools import cmp_to_key
from itertools import combinations

import sympy as sp
from sympy.core.relational import Relational

from .adapters.semialg import is_satisfiable, semialgebraic_domain
from .parameter_cells import ParameterCell
from .policy import StrategyBudgetExceeded, _normalize_policy


def linear_quadratic_cells(
    *ranges, parameters, region=None, variables=None, bounds=None, policy=None
):
    """Split selected degree drops, discriminants and coincident boundaries.

    One-parameter rational polynomial roots and multi-parameter polynomial
    sign conditions are exact within budgets. Coupled quadratic forms track
    selected principal-minor and critical-level events. Children must certify
    their geometry; sign conditions need not be connected or topology-stable.
    """
    parameters = tuple(parameters)
    if (
        not parameters
        or len(set(parameters)) != len(parameters)
        or any(not isinstance(p, sp.Symbol) or p.is_real is not True for p in parameters)
    ):
        raise NotImplementedError("automatic transition cells require distinct real parameters")
    budget = _normalize_policy(policy)
    if not budget.allow_expensive_exact and len(parameters) > budget.semialg.max_variables:
        raise StrategyBudgetExceeded("automatic parameter count exceeds max_variables")
    coordinates = tuple(
        variables or getattr(region, "variables", ()) or tuple(r[0] for r in ranges)
    )
    if (
        not coordinates
        or len(set(coordinates)) != len(coordinates)
        or any(not isinstance(v, sp.Symbol) or v in parameters for v in coordinates)
    ):
        raise ValueError("automatic transition coordinates require distinct nonparameter Symbols")
    groups = {v: [] for v in coordinates}
    linear_rows = []
    parallel_groups = {}
    candidates = []
    graph_groups = {}

    def candidate(expr, *, track=True):
        for part in sp.cancel(expr).as_numer_denom():
            if not part.has(*parameters):
                continue
            if part.free_symbols - set(parameters):
                raise NotImplementedError("transition coefficients involve undeclared parameters")
            try:
                polynomial = sp.Poly(part, *parameters, domain=sp.QQ)
            except (sp.PolynomialError, sp.CoercionFailed) as error:
                raise NotImplementedError(
                    "transition coefficients require exact rational polynomials"
                ) from error
            if len(parameters) == 1:
                if (
                    not budget.allow_expensive_exact
                    and polynomial.degree() > budget.semialg.max_polynomial_degree
                ):
                    raise StrategyBudgetExceeded(
                        "transition parameter degree exceeds max_polynomial_degree"
                    )
                pieces = [polynomial]
            else:
                pieces = [factor for factor, _ in polynomial.factor_list()[1]]
                if not budget.allow_expensive_exact and any(
                    factor.total_degree() > budget.semialg.max_polynomial_degree
                    for factor in pieces
                ):
                    raise StrategyBudgetExceeded(
                        "transition parameter degree exceeds max_polynomial_degree"
                    )
            for piece in pieces:
                normalized = piece.monic().as_expr()
                if track and normalized not in candidates:
                    candidates.append(normalized)
            if not budget.allow_expensive_exact and len(candidates) > budget.semialg.max_pieces:
                raise StrategyBudgetExceeded("automatic transition count exceeds max_pieces")

    def boundary(expr):
        active = [v for v in coordinates if expr.has(v)]
        if not active:
            candidate(expr)
            return
        numerator, denominator = sp.cancel(expr).as_numer_denom()
        if denominator.has(*coordinates):
            raise NotImplementedError("geometry denominators depend on coordinates")
        candidate(denominator)
        full_poly = sp.Poly(numerator, *coordinates)
        active = [v for v in coordinates if full_poly.degree(v) > 0]
        if not active:
            candidate(numerator)
            return
        if len(coordinates) == 2:
            for dependent in coordinates:
                independent = next(v for v in coordinates if v != dependent)
                p = sp.Poly(numerator, dependent)
                if p.degree() == 1 and not p.nth(1).has(*coordinates):
                    graph_groups.setdefault(independent, []).append(sp.cancel(-p.nth(0) / p.nth(1)))
        if full_poly.total_degree() <= 1:
            row = tuple(full_poly.coeff_monomial(v) for v in coordinates)
            if row not in linear_rows:
                linear_rows.append(row)
            pivot = next(c for c in row if c != 0)
            normal = tuple(sp.cancel(c / pivot) for c in row)
            offset = sp.cancel(full_poly.coeff_monomial((0,) * len(coordinates)) / pivot)
            parallel_groups.setdefault(normal, []).append(offset)
        elif len(active) > 1:
            if full_poly.total_degree() != 2:
                if len(coordinates) != 2:
                    raise NotImplementedError(
                        "higher-degree projection currently requires planar graph geometry"
                    )
                for dependent in coordinates:
                    independent = next(v for v in coordinates if v != dependent)
                    p = sp.Poly(numerator, dependent)
                    slope = p.nth(1)
                    if p.degree() == 1 and not slope.has(*coordinates):
                        candidate(slope)
                        graph = sp.cancel(-p.nth(0) / slope)
                        for c in sp.Poly(graph, independent).coeffs():
                            candidate(c, track=False)
                        return
                raise NotImplementedError(
                    "linear or quadratic boundaries required unless a polynomial graph projection is proved"
                )
            matrix = sp.hessian(numerator, coordinates) / 2
            zero = dict.fromkeys(coordinates, 0)
            linear = sp.Matrix([sp.diff(numerator, v).subs(zero) / 2 for v in coordinates])
            augmented = matrix.row_join(linear).col_join(
                linear.T.row_join(sp.Matrix([[numerator.subs(zero)]]))
            )
            for index in range(1, len(coordinates) + 1):
                candidate(matrix[:index, :index].det())
            candidate(augmented.det())
            for c in full_poly.coeffs():
                candidate(c, track=False)
            return
        if len(active) > 1:
            for c in full_poly.coeffs():
                candidate(c, track=c in row)
            return
        v = active[0]
        polynomial = sp.Poly(numerator, v)
        if (
            not budget.allow_expensive_exact
            and polynomial.degree() > budget.semialg.max_polynomial_degree
        ):
            raise StrategyBudgetExceeded("geometry coordinate degree exceeds max_polynomial_degree")
        for index, c in enumerate(polynomial.all_coeffs()):
            candidate(
                c, track=polynomial.degree() != 1 or index == 0 or polynomial.LC().has(*parameters)
            )
        if polynomial.degree() >= 2:
            candidate(sp.discriminant(polynomial.as_expr(), v))
        if polynomial.as_expr() not in groups[v]:
            groups[v].append(polynomial.as_expr())

    if region is not None:
        formula, coords = semialgebraic_domain(region, coordinates or None)
        if tuple(coords) != coordinates:
            raise ValueError("region coordinates differ from supplied coordinates")
        for atom in formula.atoms(Relational):
            boundary(atom.lhs - atom.rhs)
    for v, lo, hi in ranges:
        boundary(v - sp.sympify(lo))
        boundary(v - sp.sympify(hi))
    if bounds is not None:
        from collections.abc import Mapping

        entries = (
            ((v, *limits) for v, limits in bounds.items())
            if isinstance(bounds, Mapping)
            else bounds
        )
        names = {str(v): v for v in coordinates}
        for v, lo, hi in entries:
            v = names.get(v, v) if isinstance(v, str) else v
            if v not in groups:
                raise ValueError("bound coordinate is not declared")
            boundary(v - sp.sympify(lo))
            boundary(v - sp.sympify(hi))
    pairs = 0
    for v, graphs in graph_groups.items():
        for first, second in combinations(graphs, 2):
            pairs += 1
            if not budget.allow_expensive_exact and pairs > budget.semialg.max_pieces:
                raise StrategyBudgetExceeded("graph projection count exceeds max_pieces")
            polynomial = sp.Poly(first - second, v)
            if polynomial.is_zero:
                continue
            candidate(polynomial.LC())
            if polynomial.degree() > 1:
                candidate(sp.discriminant(polynomial.as_expr(), v))
            for boundary_poly in groups[v]:
                candidate(sp.resultant(polynomial.as_expr(), boundary_poly, v))

    for offsets in parallel_groups.values():
        for first, second in combinations(offsets, 2):
            pairs += 1
            if not budget.allow_expensive_exact and pairs > budget.semialg.max_pieces:
                raise StrategyBudgetExceeded("parallel boundary count exceeds max_pieces")
            candidate(first - second)
    for rows in combinations(linear_rows, len(coordinates)):
        pairs += 1
        if not budget.allow_expensive_exact and pairs > budget.semialg.max_pieces:
            raise StrategyBudgetExceeded("affine determinant count exceeds max_pieces")
        candidate(sp.Matrix(rows).det())
    for v, polynomials in groups.items():
        for first, second in combinations(polynomials, 2):
            pairs += 1
            if not budget.allow_expensive_exact and pairs > budget.semialg.max_pieces:
                raise StrategyBudgetExceeded("boundary-resultant count exceeds max_pieces")
            candidate(sp.resultant(first, second, v))
    if len(parameters) > 1:
        candidates.sort(key=sp.default_sort_key)
        if not budget.allow_expensive_exact and len(candidates) > budget.semialg.max_boolean_atoms:
            raise StrategyBudgetExceeded("hyperplane arrangement exceeds max_boolean_atoms")
        parts = [sp.true]
        for p in candidates:
            following = []
            for old in parts:
                for sign in (p < 0, sp.Eq(p, 0), p > 0):
                    condition = sp.And(old, sign)
                    if (
                        condition is not sp.false
                        and is_satisfiable(condition, parameters) is not False
                    ):
                        following.append(condition)
                    if (
                        not budget.allow_expensive_exact
                        and len(following) > budget.semialg.max_pieces
                    ):
                        raise StrategyBudgetExceeded("hyperplane arrangement exceeds max_pieces")
            parts = following
        return tuple(ParameterCell(condition) for condition in parts)
    parameter = parameters[0]
    roots = []
    for expression in candidates:
        for root in sp.Poly(expression, parameter).real_roots():
            if not any(sp.simplify(root - old) == 0 for old in roots):
                roots.append(root)

    def compare(a, b):
        difference = sp.simplify(a - b)
        if difference == 0:
            return 0
        if difference.is_positive is True:
            return 1
        if difference.is_negative is True:
            return -1
        raise NotImplementedError("transition-root order was not established")

    roots.sort(key=cmp_to_key(compare))
    if not budget.allow_expensive_exact and 2 * len(roots) + 1 > budget.semialg.max_pieces:
        raise StrategyBudgetExceeded("automatic partition exceeds max_pieces")
    edges = [-sp.oo, *roots, sp.oo]
    cells = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        cells.append(ParameterCell(sp.And(parameter > lo, parameter < hi)))
        if hi != sp.oo:
            cells.append(ParameterCell(sp.Eq(parameter, hi)))
    return tuple(cells)
