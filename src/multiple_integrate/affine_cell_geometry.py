"""Exact full-dimensional affine box images under proved parameter cells."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import sympy as sp
from sympy.core.relational import Relational

from .assumptions import _ask, _normalize_assumptions
from .policy import _normalize_policy
from .results import CoverageEvidence


def integrate_affine_cell(
    f, formula, variables, *, assumptions=None, policy=None, measure_dimension="ambient"
):
    """Lower a bounded family of invertible affine box images to ordinary boxes."""
    from .symbolic_geometry import integrate_symbolic_boxes, symbolic_boxes

    n = len(variables)
    budget = _normalize_policy(policy)
    if not n or measure_dimension not in ("ambient", "intrinsic", "top", n):
        return None
    rows = []
    context = _normalize_assumptions(assumptions)
    try:
        for atom in sp.And.make_args(formula):
            if not isinstance(atom, Relational) or atom.rel_op not in ("<", "<=", ">", ">="):
                return None
            poly = sp.Poly(atom.lhs - atom.rhs, *variables)
            if poly.total_degree() > 1:
                return None
            row = tuple(sp.diff(poly.as_expr(), v) for v in variables)
            if any(row) and row not in rows:
                rows.append(row)
    except sp.PolynomialError:
        return None
    for index, selected in enumerate(combinations(rows, n)):
        if not budget.allow_expensive_exact and index >= budget.semialg.max_pieces:
            return None
        matrix = sp.Matrix(selected)
        det = sp.factor(matrix.det())
        if _ask(sp.Q.real(det) & sp.Q.finite(det) & sp.Q.nonzero(det), context) is not True:
            continue
        if not all(_ask(sp.Q.real(c) & sp.Q.finite(c), context) is True for c in matrix):
            continue
        coords = tuple(sp.Dummy("affine_cell", real=True) for _ in variables)
        inverse = matrix.inv() * sp.Matrix(coords)
        mapping = dict(zip(variables, inverse, strict=True))
        target = (
            sp.sympify(formula)
            .subs(mapping, simultaneous=True)
            .replace(
                lambda a: isinstance(a, Relational),
                lambda a: a.func(sp.expand(a.lhs), sp.expand(a.rhs)),
            )
        )
        boxes = symbolic_boxes(target, coords, assumptions=assumptions, policy=policy)
        if boxes is None or any(any(i.lower == i.upper for _, i in box) for box in boxes):
            continue
        child = integrate_symbolic_boxes(
            sp.sympify(f).subs(mapping, simultaneous=True) / sp.Abs(det),
            target,
            coords,
            assumptions=assumptions,
            policy=policy,
            measure_dimension="ambient",
            _allow_affine=False,
        )
        if child is None:
            continue
        evidence = CoverageEvidence(
            "integration_domain",
            True,
            "affine_parameter_cell",
            (
                ("matrix", sp.ImmutableMatrix(matrix)),
                ("jacobian", 1 / sp.Abs(det)),
                ("source_formula", formula),
                ("basis", "invertible real linear map and certified full-dimensional target boxes"),
            ),
        )
        return replace(
            child,
            method="affine_parameter_cell",
            coverage=(*child.coverage, evidence),
            diagnostics={**child.diagnostics, "affine_cell_matrix": sp.ImmutableMatrix(matrix)},
        )
    return None
