"""Cheap exact axis-wise linear/quadratic integration ranges within a parameter cell.

Note this is a native integration recognizer, not a general geometry/CAD backend.
Unknown signs and endpoint orders decline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from itertools import product

import sympy as sp
from sympy.core.relational import Relational

from .assumptions import _ask, _condition_to_predicate, _normalize_assumptions
from .policy import StrategyBudgetExceeded, _normalize_policy, check_semialg_budget


@dataclass(frozen=True)
class CellInterval:
    lower: sp.Expr
    upper: sp.Expr
    left_open: bool = False
    right_open: bool = False


class _Uncertain(Exception):
    pass


def _compare(a, b, context):
    if a == b:
        return 0
    if a == -sp.oo or b == sp.oo:
        return -1
    if a == sp.oo or b == -sp.oo:
        return 1
    from .cell_proofs import nonnegative, polynomial_sign

    delta = sp.simplify(a - b)
    if nonnegative(delta, context, strict=True):
        return 1
    if nonnegative(-delta, context, strict=True):
        return -1
    if delta == 0 or _ask(sp.Q.zero(delta), context) is True:
        return 0
    if _ask(sp.Q.negative(delta), context) is True:
        return -1
    if _ask(sp.Q.positive(delta), context) is True:
        return 1
    reciprocal = sp.radsimp(1 / delta)
    if _ask(sp.Q.positive(reciprocal) & sp.Q.finite(reciprocal), context) is True:
        return 1
    if _ask(sp.Q.negative(reciprocal) & sp.Q.finite(reciprocal), context) is True:
        return -1
    # Weak ordering is sufficient for ambient interval assembly. Intrinsic
    # evaluation separately proves stable dimension and distinct point fibers.
    if _ask(sp.Q.nonnegative(delta), context) is True:
        return 1
    if _ask(sp.Q.nonpositive(delta), context) is True:
        return -1
    sign = polynomial_sign(delta, context)
    if sign is not None:
        return sign
    # Square-root comparison is reduced only after both sides have proved signs.
    radicals = delta.atoms(sp.Pow)
    roots = [r for r in radicals if r.exp == sp.Rational(1, 2)]
    if len(roots) == 1:
        root = roots[0]
        coefficient = sp.expand(delta).coeff(root)
        remainder = sp.simplify(delta - coefficient * root)
        if coefficient in (1, -1) and not remainder.has(root):
            target = -remainder / coefficient
            if nonnegative(root.base, context) and nonnegative(target, context):
                sign = polynomial_sign(sp.expand(root.base - target**2), context)
                if sign is not None:
                    return int(coefficient) * sign
    raise _Uncertain


def _intersection(a, b, context):
    left = _compare(a.lower, b.lower, context)
    right = _compare(a.upper, b.upper, context)
    lower = a.lower if left >= 0 else b.lower
    upper = a.upper if right <= 0 else b.upper
    left_open = (
        (a.left_open or b.left_open) if left == 0 else a.left_open if left > 0 else b.left_open
    )
    right_open = (
        (a.right_open or b.right_open)
        if right == 0
        else a.right_open
        if right < 0
        else b.right_open
    )
    order = _compare(lower, upper, context)
    if order > 0 or (order == 0 and (left_open or right_open)):
        return None
    return CellInterval(lower, upper, left_open, right_open)


def _union(intervals, context):
    intervals = sorted(intervals, key=cmp_to_key(lambda a, b: _compare(a.lower, b.lower, context)))
    result = []
    for interval in intervals:
        if not result:
            result.append(interval)
            continue
        old = result[-1]
        relation = _compare(interval.lower, old.upper, context)
        if relation > 0 or (relation == 0 and interval.left_open and old.right_open):
            result.append(interval)
            continue
        order = _compare(interval.upper, old.upper, context)
        upper = interval.upper if order > 0 else old.upper
        right_open = (
            interval.right_open
            if order > 0
            else old.right_open
            if order < 0
            else interval.right_open and old.right_open
        )
        left_open = (
            old.left_open and interval.left_open
            if _compare(old.lower, interval.lower, context) == 0
            else old.left_open
        )
        result[-1] = CellInterval(old.lower, upper, left_open, right_open)
    return result


_FULL = CellInterval(-sp.oo, sp.oo, True, True)


def _atom_intervals(atom, var, context):
    if atom is sp.true:
        return [_FULL]
    if atom is sp.false:
        return []
    if atom.func in (sp.And, sp.Or):
        groups = [_atom_intervals(a, var, context) for a in atom.args]
        if atom.func == sp.Or:
            return _union([i for group in groups for i in group], context)
        result = [_FULL]
        for group in groups:
            result = [
                i for a in result for b in group if (i := _intersection(a, b, context)) is not None
            ]
        return result
    if not isinstance(atom, Relational):
        raise _Uncertain
    if not atom.has(var):
        truth = _ask(_condition_to_predicate(atom), context)
        if truth is None:
            raise _Uncertain
        return [_FULL] if truth else []
    try:
        poly = sp.Poly(atom.lhs - atom.rhs, var)
    except sp.PolynomialError as exc:
        raise _Uncertain from exc
    if poly.degree() > 2:
        raise _Uncertain
    coefficients = {power: poly.nth(power) for power in range(3)}
    if not all(_ask(sp.Q.real(c) & sp.Q.finite(c), context) is True for c in coefficients.values()):
        raise _Uncertain
    degree = 2
    while degree and _ask(sp.Q.zero(coefficients[degree]), context) is True:
        degree -= 1
    op = atom.rel_op
    if degree == 0:
        condition = type(atom)(coefficients[0], 0)
        truth = _ask(_condition_to_predicate(condition), context)
        if truth is None:
            raise _Uncertain
        return [_FULL] if truth else []
    leading = coefficients[degree]
    from .cell_proofs import nonnegative

    # Weak endpoint ordering is useful for ambient interval assembly, but a
    # polynomial's degree and root formulas require a strictly nonzero leading
    # coefficient on the whole cell, including its equality fibers.
    if nonnegative(leading, context, strict=True):
        sign = 1
    elif nonnegative(-leading, context, strict=True):
        sign = -1
    else:
        raise _Uncertain
    if sign < 0:
        coefficients = {p: -c for p, c in coefficients.items()}
        op = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "==": "==", "!=": "!="}[op]
    if degree == 1:
        root = sp.simplify(-coefficients[0] / coefficients[1])
        if op in ("<", "<="):
            return [CellInterval(-sp.oo, root, True, op == "<")]
        if op in (">", ">="):
            return [CellInterval(root, sp.oo, op == ">", True)]
        if op == "==":
            return [CellInterval(root, root)]
        return [CellInterval(-sp.oo, root, True, True), CellInterval(root, sp.oo, True, True)]
    a, b, c = (coefficients[i] for i in (2, 1, 0))
    discriminant = sp.simplify(b * b - 4 * a * c)
    if op in (">=", "<") and _ask(sp.Q.nonpositive(discriminant), context) is True:
        return [_FULL] if op == ">=" else []
    order = _compare(discriminant, sp.S.Zero, context)
    if order < 0:
        return [_FULL] if op in (">", ">=", "!=") else []
    if order == 0:
        root = sp.simplify(-b / (2 * a))
        if op in ("<=", "=="):
            return [CellInterval(root, root)]
        if op == "<":
            return []
        if op == ">=":
            return [_FULL]
        return [CellInterval(-sp.oo, root, True, True), CellInterval(root, sp.oo, True, True)]
    left = sp.simplify((-b - sp.sqrt(discriminant)) / (2 * a))
    right = sp.simplify((-b + sp.sqrt(discriminant)) / (2 * a))
    if op in ("<", "<="):
        return [CellInterval(left, right, op == "<", op == "<")]
    if op in (">", ">="):
        return [
            CellInterval(-sp.oo, left, True, op == ">"),
            CellInterval(right, sp.oo, op == ">", True),
        ]
    if op == "==":
        return [CellInterval(left, left), CellInterval(right, right)]
    return [
        CellInterval(-sp.oo, left, True, True),
        CellInterval(left, right, True, True),
        CellInterval(right, sp.oo, True, True),
    ]


def symbolic_boxes(formula, variables, *, assumptions=None, bounds=None, policy=None):
    """Return disjoint axis-wise interval products, or None if certification fails."""
    variables = tuple(variables)
    context = _normalize_assumptions(assumptions)
    policy = _normalize_policy(policy)
    allowed, reason = check_semialg_budget(formula, variables, policy)
    if not allowed:
        raise StrategyBudgetExceeded(reason)
    fresh = {v: sp.Dummy(v.name, real=True) for v in variables}
    formula = sp.sympify(formula).xreplace(fresh)
    groups = {v: [] for v in variables}
    atoms = sp.And.make_args(formula)
    for atom in atoms:
        dependencies = [v for v in variables if atom.has(fresh[v])]
        if len(dependencies) > 1:
            return None
        if dependencies:
            groups[dependencies[0]].append(atom)
        else:
            try:
                if not _atom_intervals(atom, sp.Dummy("unused", real=True), context):
                    return ()
            except _Uncertain:
                return None
    if bounds is not None:
        from collections.abc import Mapping

        entries = (
            ((v, *limits) for v, limits in bounds.items())
            if isinstance(bounds, Mapping)
            else bounds
        )
        names = {str(v): v for v in variables}
        for v, lo, hi in entries:
            v = names.get(v, v) if isinstance(v, str) else v
            if v not in groups or sp.sympify(lo).has(*variables) or sp.sympify(hi).has(*variables):
                return None
            groups[v].extend((fresh[v] >= sp.sympify(lo), fresh[v] <= sp.sympify(hi)))
    try:
        intervals = [_atom_intervals(sp.And(*groups[v]), fresh[v], context) for v in variables]
        count = sp.prod(len(group) for group in intervals)
        if not policy.allow_expensive_exact and count > policy.semialg.max_pieces:
            raise StrategyBudgetExceeded("symbolic box count exceeds semialg max_pieces")
        return tuple(
            tuple((v, interval) for v, interval in zip(variables, box, strict=True))
            for box in product(*intervals)
        )
    except _Uncertain:
        return None


def integrate_symbolic_boxes(
    f,
    formula,
    variables,
    *,
    assumptions=None,
    bounds=None,
    policy=None,
    measure_dimension="ambient",
    principal_value=False,
    regularization=None,
    _allow_affine=True,
):
    """Evaluate certified cell ranges and retain coverage of the original formula."""
    from dataclasses import replace

    from .api.integration import integrate_result
    from .conditional import conditional_status
    from .results import ConvergenceEvidence, CoverageEvidence, IntegrationResult, IntegrationStatus

    boxes = symbolic_boxes(
        formula, variables, assumptions=assumptions, bounds=bounds, policy=policy
    )
    if boxes is None:
        if (
            _allow_affine
            and bounds is None
            and principal_value in (False, None)
            and regularization is None
        ):
            from .quadratic_cell_geometry import integrate_quadratic_cell

            quadratic = integrate_quadratic_cell(
                f,
                formula,
                variables,
                assumptions=assumptions,
                policy=policy,
                measure_dimension=measure_dimension,
            )
            if quadratic is not None:
                return quadratic
            from .clipped_quadratic import integrate_clipped_quadratic_cell

            clipped = integrate_clipped_quadratic_cell(
                f,
                formula,
                variables,
                assumptions=assumptions,
                policy=policy,
                measure_dimension=measure_dimension,
            )
            if clipped is not None:
                return clipped
            from .quadratic_sectors import integrate_quadratic_sector

            sector = integrate_quadratic_sector(
                f,
                formula,
                variables,
                assumptions=assumptions,
                policy=policy,
                measure_dimension=measure_dimension,
            )
            if sector is not None:
                return sector
            from .boundary_reduction import integrate_boundary

            boundary = integrate_boundary(
                f,
                formula,
                variables,
                assumptions=assumptions,
                policy=policy,
                measure_dimension=measure_dimension,
            )
            if boundary is not None:
                return boundary
            from .affine_cell_geometry import integrate_affine_cell

            affine = integrate_affine_cell(
                f,
                formula,
                variables,
                assumptions=assumptions,
                policy=policy,
                measure_dimension=measure_dimension,
            )
            if affine is not None:
                return affine
            from .cell_graphs import integrate_cell_graph, integrate_partitioned_cell_graph

            graph = integrate_cell_graph(
                f,
                formula,
                variables,
                assumptions=assumptions,
                policy=policy,
                measure_dimension=measure_dimension,
            )
            if graph is not None:
                return graph
            return integrate_partitioned_cell_graph(
                f,
                formula,
                variables,
                assumptions=assumptions,
                policy=policy,
                measure_dimension=measure_dimension,
            )
        return None
    if measure_dimension in ("intrinsic", "top") or measure_dimension == 0:
        from .cell_proofs import nonnegative

        context = _normalize_assumptions(assumptions)
        if any(
            i.lower != i.upper and not nonnegative(i.upper - i.lower, context, strict=True)
            for box in boxes
            for _, i in box
        ):
            return None
        points = [box for box in boxes if all(i.lower == i.upper for _, i in box)]
        for index, point in enumerate(points):
            for other in points[index + 1 :]:
                if not any(
                    nonnegative(i.lower - j.lower, context, strict=True)
                    or nonnegative(j.lower - i.lower, context, strict=True)
                    for (_, i), (_, j) in zip(point, other, strict=True)
                ):
                    return None
    dimensions = [sum(i.lower != i.upper for _, i in box) for box in boxes]
    target = (
        max(dimensions, default=0)
        if measure_dimension in ("intrinsic", "top")
        else len(variables)
        if measure_dimension == "ambient"
        else measure_dimension
    )
    if type(target) is not int or target < 0 or any(d > target for d in dimensions):
        return None
    prescribed = principal_value not in (False, None) or regularization is not None
    if prescribed and (len(boxes) != 1 or target != len(variables) or dimensions[0] != target):
        return None
    children = []
    for box, dimension in zip(boxes, dimensions, strict=True):
        if dimension < target:
            continue
        fixed = {v: i.lower for v, i in box if i.lower == i.upper}
        density = sp.sympify(f).subs(fixed, simultaneous=True)
        ranges = tuple((v, i.lower, i.upper) for v, i in box if v not in fixed)
        if ranges:
            children.append(
                integrate_result(
                    density,
                    *ranges,
                    assumptions=assumptions,
                    policy=policy,
                    principal_value=principal_value,
                    regularization=regularization,
                )
            )
        else:
            status, cases, convergent, complete = conditional_status(density)
            from .cell_proofs import finite_external_parameters, finite_parameter_domain

            context = _normalize_assumptions(assumptions)

            def point_domain(answer, scope):
                return finite_parameter_domain(answer, scope) and finite_external_parameters(
                    f, variables, scope
                )

            if status is IntegrationStatus.EXACT and not point_domain(density, context):
                status, convergent, complete = IntegrationStatus.UNKNOWN, None, False
            guarded = []
            for case in cases:
                try:
                    branch_context = _normalize_assumptions(
                        sp.And(*context.original, case.condition)
                    )
                except ValueError:
                    guarded.append(case)
                    continue
                if case.status is IntegrationStatus.EXACT and not point_domain(
                    case.value, branch_context
                ):
                    case = replace(case, status=IntegrationStatus.UNKNOWN, convergent=None)
                    complete = False
                guarded.append(case)
            if cases and not complete:
                cases = tuple(guarded)
                status = (
                    IntegrationStatus.CONDITIONAL
                    if any(c.status is IntegrationStatus.EXACT for c in cases)
                    else IntegrationStatus.UNKNOWN
                )
                convergent = None
            finite = point_domain(density, context) and _ask(sp.Q.finite(density), context) is True
            point_evidence = tuple(
                ConvergenceEvidence(kind, True if finite else None, "finite_point_measure")
                for kind in ("ordinary", "absolute")
            )
            for case in cases:
                try:
                    branch_context = _normalize_assumptions(
                        sp.And(*context.original, case.condition)
                    )
                except ValueError:
                    continue
                if (
                    point_domain(case.value, branch_context)
                    and _ask(sp.Q.finite(case.value), branch_context) is True
                ):
                    point_evidence += tuple(
                        ConvergenceEvidence(kind, True, "finite_point_measure", case.condition)
                        for kind in ("ordinary", "absolute")
                    )
            children.append(
                IntegrationResult(
                    density,
                    status,
                    "symbolic_cell_point",
                    cases=cases,
                    convergent=convergent,
                    complete=complete,
                    exact=status is IntegrationStatus.EXACT,
                    assumptions=sp.And(*context.original),
                    convergence_evidence=point_evidence,
                    _convergence_policy=_normalize_policy(policy),
                )
            )
    coverage = CoverageEvidence(
        "integration_domain",
        True,
        "symbolic_cell_geometry",
        (
            ("boxes", len(boxes)),
            ("measure_dimension", target),
            ("basis", "certified axis-wise linear/quadratic ranges under cell assumptions"),
        ),
    )
    if len(children) == 1:
        child = children[0]
        return replace(
            child,
            method="symbolic_cell_geometry",
            coverage=(*child.coverage, coverage),
            diagnostics={**child.diagnostics, "symbolic_boxes": boxes},
        )
    if any(child.cases for child in children):
        # A product of unresolved child partitions needs further orchestration.
        return None
    value = sp.Add(*(child.value for child in children))
    status, cases, convergent, complete = conditional_status(value)
    if any(child.status is IntegrationStatus.DIVERGENT for child in children):
        status, convergent, complete = IntegrationStatus.DIVERGENT, False, True
    absolute = all(child.absolute_convergent is True for child in children)
    if not children:
        absolute = True
    return IntegrationResult(
        value,
        status,
        "symbolic_cell_geometry",
        cases=cases,
        convergent=convergent,
        complete=complete and all(child.complete is True for child in children),
        exact=all(child.exact for child in children),
        coverage=(coverage,),
        convergence_evidence=(
            ConvergenceEvidence(
                "ordinary", True if absolute else None, "finite_domain_decomposition"
            ),
            ConvergenceEvidence(
                "absolute", True if absolute else None, "finite_domain_decomposition"
            ),
        ),
        diagnostics={"symbolic_boxes": boxes, "piece_results": tuple(children)},
    )
