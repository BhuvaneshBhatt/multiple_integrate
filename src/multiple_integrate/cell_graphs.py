"""Certified polynomial graph bounds inside caller-supplied parameter cells."""

from __future__ import annotations

from dataclasses import replace

import sympy as sp
from sympy.core.relational import Relational

from .assumptions import _ask, _normalize_assumptions
from .cell_proofs import interval_nonnegative, nonnegative
from .results import ConvergenceEvidence, CoverageEvidence, IntegrationResult, IntegrationStatus


def integrate_cell_graph(
    f, formula, variables, *, assumptions=None, policy=None, measure_dimension="ambient"
):
    """Lower finite polynomial strips after proving their endpoint order globally."""
    from .api.integration import integrate_result
    from .symbolic_geometry import symbolic_boxes

    if len(variables) != 2 or measure_dimension not in ("ambient", "intrinsic", "top", 2):
        return None
    context = _normalize_assumptions(assumptions)
    for x, y in (tuple(variables), tuple(reversed(variables))):
        axis = []
        lower = []
        upper = []
        valid = True
        for atom in sp.And.make_args(formula):
            if not isinstance(atom, Relational) or atom.rel_op not in ("<", "<=", ">", ">="):
                valid = False
                break
            if not atom.has(y):
                axis.append(atom)
                continue
            expr = (
                sp.expand(atom.lhs - atom.rhs)
                if atom.rel_op in (">", ">=")
                else sp.expand(atom.rhs - atom.lhs)
            )
            try:
                poly = sp.Poly(expr, y)
            except sp.PolynomialError:
                valid = False
                break
            slope = poly.nth(1)
            if poly.degree() != 1 or slope.has(x) or _ask(sp.Q.finite(slope), context) is not True:
                valid = False
                break
            bound = sp.cancel(-poly.nth(0) / slope)
            try:
                bound_poly = sp.Poly(bound, x)
                if not all(
                    _ask(sp.Q.real(c) & sp.Q.finite(c), context) is True
                    for c in bound_poly.coeffs()
                ):
                    valid = False
                    break
            except sp.PolynomialError:
                valid = False
                break
            if nonnegative(slope, context, strict=True):
                lower.append(bound)
            elif nonnegative(-slope, context, strict=True):
                upper.append(bound)
            else:
                valid = False
                break
        if not valid or not lower or not upper:
            continue
        boxes = symbolic_boxes(sp.And(*axis), (x,), assumptions=assumptions, policy=policy)
        if boxes is None or len(boxes) != 1:
            continue
        interval = boxes[0][0][1]
        lo, hi = interval.lower, interval.upper
        if _ask(
            sp.Q.real(lo) & sp.Q.finite(lo) & sp.Q.real(hi) & sp.Q.finite(hi),
            context,
        ) is not True or not nonnegative(hi - lo, context, strict=True):
            continue
        bottom = next(
            (
                b
                for b in lower
                if all(interval_nonnegative(b - c, x, lo, hi, context) for c in lower)
            ),
            None,
        )
        top = next(
            (
                b
                for b in upper
                if all(interval_nonnegative(c - b, x, lo, hi, context) for c in upper)
            ),
            None,
        )
        if (
            bottom is None
            or top is None
            or not interval_nonnegative(top - bottom, x, lo, hi, context)
        ):
            continue
        if (
            measure_dimension != "ambient"
            and measure_dimension != 2
            and not interval_nonnegative(top - bottom, x, lo, hi, context, strict_interior=True)
        ):
            continue
        ranges = ((y, bottom, top), (x, lo, hi))
        try:
            polynomials = [sp.Poly(expr, x) for expr in (bottom, top)]
            density = sp.Poly(f, x, y)
            finite = all(
                _ask(sp.Q.finite(c), context) is True
                for p in (*polynomials, density)
                for c in p.coeffs()
            )
        except sp.PolynomialError:
            finite = False
        if finite:
            from .policy import _normalize_policy

            budget = _normalize_policy(policy)
            degree = max(int(p.degree()) if not p.is_zero else 0 for p in polynomials)
            estimate = max((p + degree * (q + 1) for (p, q), c in density.terms()), default=0)
            if not budget.allow_expensive_exact and (
                estimate > budget.semialg.max_polynomial_degree
                or len(density.terms()) > budget.semialg.max_pieces
            ):
                return None
            inner = sp.Poly(
                sp.expand(
                    sum(
                        c * x**p * (top ** (q + 1) - bottom ** (q + 1)) / sp.Integer(q + 1)
                        for (p, q), c in density.terms()
                    )
                ),
                x,
            )
            value = sp.Add(
                *(
                    c * (hi ** (p + 1) - lo ** (p + 1)) / sp.Integer(p + 1)
                    for (p,), c in inner.terms()
                )
            )
            child = IntegrationResult(
                value,
                IntegrationStatus.EXACT,
                "parameter_cell_graph",
                convergent=True,
                exact=True,
                complete=True,
                assumptions=context.predicate_expr,
            )
        else:
            child = integrate_result(f, *ranges, assumptions=assumptions, policy=policy)
        proof = CoverageEvidence(
            "integration_domain",
            True,
            "parameter_cell_graph",
            (
                ("ranges", ranges),
                ("basis", "polynomial envelopes proved on the complete caller cell"),
            ),
        )
        claims = child.convergence_evidence
        if finite:
            claims = tuple(e for e in claims if e.kind not in ("absolute", "ordinary")) + tuple(
                ConvergenceEvidence(k, True, "compact_polynomial_graph")
                for k in ("absolute", "ordinary")
            )
        return replace(
            child,
            method="parameter_cell_graph",
            coverage=(*child.coverage, proof),
            convergence_evidence=claims,
        )
    return None


def integrate_partitioned_cell_graph(
    f, formula, variables, *, assumptions=None, policy=None, measure_dimension="ambient"
):
    """Exact graph-envelope subdivisions with proved symbolic root order.

    Complete roots and signs are required on the caller cell; no parameter
    samples stand in for a proof of the algebraic fibers.
    """
    from functools import cmp_to_key
    from itertools import combinations

    from .assumptions import _refine_with_assumptions
    from .fast_rules import _polynomial
    from .policy import _normalize_policy
    from .results import UnresolvedIntegration
    from .symbolic_geometry import _compare, _Uncertain, symbolic_boxes

    if len(variables) != 2 or measure_dimension not in ("ambient", 2):
        return None
    density = _polynomial(sp.sympify(f), tuple(variables))
    if density is None or not all(
        _ask(sp.Q.finite(c), _normalize_assumptions(assumptions)) is True for c in density.coeffs()
    ):
        return None
    budget = _normalize_policy(policy)
    context = _normalize_assumptions(assumptions)
    for x, y in (tuple(variables), tuple(reversed(variables))):
        axes = []
        graphs = []
        lower = []
        upper = []
        valid = True
        for atom in sp.And.make_args(formula):
            if not isinstance(atom, Relational) or atom.rel_op not in ("<", "<=", ">", ">="):
                valid = False
                break
            if not atom.has(y):
                axes.append(atom)
                continue
            expr = atom.lhs - atom.rhs if atom.rel_op in (">", ">=") else atom.rhs - atom.lhs
            p = _polynomial(expr, (x, y))
            if p is None or p.degree(y) != 1:
                valid = False
                break
            slope = sp.diff(expr, y)
            if slope.has(x, y):
                valid = False
                break
            graph = sp.cancel(-expr.subs(y, 0) / slope)
            if _ask(sp.Q.positive(slope) & sp.Q.finite(slope), context) is True:
                lower.append(graph)
            elif _ask(sp.Q.negative(slope) & sp.Q.finite(slope), context) is True:
                upper.append(graph)
            else:
                valid = False
                break
            graphs.append(graph)
        if not valid or not lower or not upper:
            continue
        if (
            not budget.allow_expensive_exact
            and len(graphs) * (len(graphs) - 1) // 2 > budget.semialg.max_pieces
        ):
            continue
        boxes = symbolic_boxes(sp.And(*axes), (x,), assumptions=assumptions, policy=policy)
        if boxes is None or len(boxes) != 1:
            continue
        interval = boxes[0][0][1]
        lo, hi = interval.lower, interval.upper
        if (
            _ask(sp.Q.real(lo) & sp.Q.real(hi) & sp.Q.finite(lo) & sp.Q.finite(hi), context)
            is not True
        ):
            continue
        points = [lo, hi]
        for first, second in combinations(graphs, 2):
            p = _polynomial(first - second, (x,))
            if p is None:
                valid = False
                break
            if p.is_zero:
                continue
            if not all(_ask(sp.Q.real(c) & sp.Q.finite(c), context) is True for c in p.coeffs()):
                valid = False
                break
            roots = sp.roots(p.as_expr(), x)
            if sum(roots.values()) != p.degree():
                valid = False
                break
            for root in roots:
                root = _refine_with_assumptions(root, context)
                real = _ask(sp.Q.real(root) & sp.Q.finite(root), context)
                if real is not True:
                    if _ask(sp.Q.real(root), context) is False:
                        continue
                    valid = False
                    break
                if nonnegative(root - lo, context) and nonnegative(hi - root, context):
                    points.append(root)
                elif nonnegative(lo - root, context) or nonnegative(root - hi, context):
                    continue
                else:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        try:
            points = tuple(
                sorted(set(points), key=cmp_to_key(lambda a, b: _compare(a, b, context)))
            )
        except (_Uncertain, UnresolvedIntegration):
            continue
        if not budget.allow_expensive_exact and len(points) > budget.semialg.max_pieces:
            continue
        if len(points) <= 2:
            continue
        children = []
        ranges = []
        for left, right in zip(points, points[1:], strict=False):
            if left == right:
                continue
            if any(
                interval_nonnegative(bottom - top, x, left, right, context, strict_interior=True)
                for bottom in lower
                for top in upper
            ):
                continue
            piece = sp.And(formula, x >= left, x <= right)
            child = integrate_cell_graph(
                f,
                piece,
                variables,
                assumptions=assumptions,
                policy=policy,
                measure_dimension=measure_dimension,
            )
            if child is None or not child.complete:
                valid = False
                break
            children.append(child)
            ranges.append((left, right))
        if not valid:
            continue
        return IntegrationResult(
            sp.Add(*(child.value for child in children)),
            IntegrationStatus.EXACT,
            "partitioned_parameter_graph",
            exact=True,
            complete=True,
            convergent=True,
            assumptions=context.predicate_expr,
            coverage=(
                CoverageEvidence(
                    "integration_domain",
                    True,
                    "partitioned_parameter_graph",
                    (
                        (
                            "basis",
                            "complete graph-intersection roots, proved order, and envelope signs on all intervals",
                        ),
                        ("base_intervals", tuple(ranges)),
                    ),
                ),
            ),
            convergence_evidence=tuple(
                ConvergenceEvidence(k, True, "compact_partitioned_polynomial_graph")
                for k in ("ordinary", "absolute")
            ),
        )
    return None
