"""Boundary flux and reusable moments on polygonal and polyhedral domains."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations

import sympy as sp
from sympy.core.relational import Relational

from .assumptions import _ask, _normalize_assumptions
from .cell_proofs import native_predicate_true
from .fast_rules import _polynomial, polynomial_function_moment
from .policy import _normalize_policy, _with_policy
from .results import ConvergenceEvidence, CoverageEvidence, IntegrationResult, IntegrationStatus


class _Uncertain(Exception):
    """A whole-cell geometric sign or incidence was not established."""


def _sign(expr, context):
    expr = sp.factor(expr)
    if expr == 0 or native_predicate_true(sp.Eq(expr, 0), context):
        return 0
    if native_predicate_true(expr > 0, context):
        return 1
    if native_predicate_true(expr < 0, context):
        return -1
    raise _Uncertain


def _halfspaces(formula, variables, context):
    """Rational normals, finite real offsets; None means proved empty."""
    rows = []
    for atom in sp.And.make_args(formula):
        if not isinstance(atom, Relational) or atom.rel_op not in ("<", "<=", ">", ">="):
            raise _Uncertain
        expr = atom.lhs - atom.rhs
        if atom.rel_op in (">", ">="):
            expr = -expr
        try:
            poly = sp.Poly(expr, *variables)
        except sp.PolynomialError as exc:
            raise _Uncertain from exc
        if poly.total_degree() > 1:
            raise _Uncertain
        normal = tuple(poly.coeff_monomial(v) for v in variables)
        offset = poly.coeff_monomial(1)
        if (
            not all(v.is_Rational for v in normal)
            or _ask(sp.Q.real(offset) & sp.Q.finite(offset), context) is not True
        ):
            raise _Uncertain
        if not any(normal):
            sign = _sign(offset, context)
            if sign > 0 or (sign == 0 and atom.rel_op in ("<", ">")):
                return None
            continue
        scale = abs(next(v for v in normal if v))
        row = tuple(v / scale for v in (*normal, offset))
        if row not in rows:
            rows.append(row)
    return tuple(rows)


def _polygon(rows, context):
    """Whole supporting-line intersections, without sampling parameter fibers."""
    facets = []
    for a, b, c in rows:
        origin = (-c / a, sp.S.Zero) if a else (sp.S.Zero, -c / b)
        tangent = (-b, a)
        lo, hi = None, None
        feasible = True
        for A, B, C in rows:
            slope = A * tangent[0] + B * tangent[1]
            offset = A * origin[0] + B * origin[1] + C
            if slope:
                endpoint = sp.cancel(-offset / slope)
                if slope > 0:
                    if hi is None or _sign(hi - endpoint, context) >= 0:
                        hi = endpoint
                elif lo is None or _sign(endpoint - lo, context) >= 0:
                    lo = endpoint
            elif _sign(offset, context) > 0:
                feasible = False
                break
        if not feasible:
            continue
        if lo is not None and hi is not None and _sign(hi - lo, context) <= 0:
            continue
        if lo is None or hi is None:
            raise _Uncertain
        facets.append((origin, tangent, lo, hi, (a, b)))
    area = sp.factor(
        sum(_dot(origin, normal) * (hi - lo) / 2 for origin, _, lo, hi, normal in facets)
    )
    sign = _sign(area, context)
    if sign < 0:
        raise _Uncertain
    return tuple(facets) if sign else ()


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


@lru_cache(maxsize=512)
def _convex_facets(rows, conditions):
    return _polygon(rows, _normalize_assumptions(conditions))


def _integral_polynomial(poly, lo, hi):
    return sp.Add(*(c * (hi ** (k + 1) - lo ** (k + 1)) / (k + 1) for (k,), c in poly.terms()))


@lru_cache(maxsize=4096)
def _polygon_moment(facets, powers):
    """Coefficient-independent moment; callers enforce degree budgets before lookup."""
    t = sp.Dummy("boundary_parameter", real=True)
    value = sp.S.Zero
    for origin, tangent, lo, hi, normal in facets:
        weight = _dot(origin, normal)
        if weight == 0:
            continue
        term = sp.prod((p + t * d) ** k for p, d, k in zip(origin, tangent, powers, strict=True))
        value += weight * _integral_polynomial(sp.Poly(term, t), lo, hi) / (sum(powers) + 2)
    return sp.factor(value)


def clear_boundary_cache():
    """Clear geometry and coefficient-independent moment tables together."""
    _convex_facets.cache_clear()
    _polygon_moment.cache_clear()
    _loop_facets.cache_clear()


def _truth(formula, variables, point):
    value = formula.subs(dict(zip(variables, point, strict=True)), simultaneous=True)
    if value not in (sp.true, sp.false):
        raise _Uncertain
    return value == sp.true


@lru_cache(maxsize=128)
def _loop_facets(formula, variables):
    """Exact rational line arrangement; local side signs certify oriented loops."""
    lines = []
    for atom in formula.atoms(Relational):
        if atom.rel_op not in ("<", "<=", ">", ">="):
            raise _Uncertain
        poly = sp.Poly(atom.lhs - atom.rhs, *variables)
        if poly.total_degree() > 1 or not all(c.is_Rational for c in poly.coeffs()):
            raise _Uncertain
        a, b = (poly.coeff_monomial(v) for v in variables)
        c = poly.coeff_monomial(1)
        if not (a or b):
            continue
        scale = a if a else b
        row = (a / scale, b / scale, c / scale)
        if row not in lines:
            lines.append(row)
    vertices = []
    for (a, b, c), (A, B, C) in combinations(lines, 2):
        det = a * B - b * A
        if det:
            vertices.append(((b * C - c * B) / det, (c * A - a * C) / det))
    if not vertices:
        raise _Uncertain
    farx = max(p[0] for p in vertices) + 1
    fary = max(p[1] for p in vertices) + 1
    for k in range(len(lines) + 1):
        point = (farx, fary + k)
        if all(a * point[0] + b * point[1] + c != 0 for a, b, c in lines):
            if _truth(formula, variables, point):
                raise _Uncertain  # Reject the unbounded exterior component.
            break
    facets = []
    starts, ends = Counter(), Counter()
    for a, b, c in lines:
        origin = (-c / a, sp.S.Zero) if a else (sp.S.Zero, -c / b)
        tangent = (-b, a)
        cuts = set()
        for A, B, C in lines:
            slope = A * tangent[0] + B * tangent[1]
            if slope:
                cuts.add(-(A * origin[0] + B * origin[1] + C) / slope)
        cuts = sorted(cuts)
        intervals = list(zip([None, *cuts], [*cuts, None], strict=True))
        for lo, hi in intervals:
            t = (
                (lo + hi) / 2
                if lo is not None and hi is not None
                else (hi - 1 if lo is None else lo + 1)
            )
            point = tuple(p + t * d for p, d in zip(origin, tangent, strict=True))
            delta = sp.S.One
            for A, B, C in lines:
                offset = A * point[0] + B * point[1] + C
                rate = A * a + B * b
                if offset and rate:
                    delta = min(delta, abs(offset / rate) / 2)
            left = _truth(formula, variables, (point[0] - delta * a, point[1] - delta * b))
            right = _truth(formula, variables, (point[0] + delta * a, point[1] + delta * b))
            if left == right:
                continue
            if lo is None or hi is None:
                raise _Uncertain
            sign = 1 if left else -1
            normal = (sign * a, sign * b)
            facets.append((origin, tangent, lo, hi, normal))
            start = tuple(p + lo * d for p, d in zip(origin, tangent, strict=True))
            end = tuple(p + hi * d for p, d in zip(origin, tangent, strict=True))
            if sign < 0:
                start, end = end, start
            starts[start] += 1
            ends[end] += 1
    if starts != ends or not facets or _polygon_moment(tuple(facets), (0, 0)) <= 0:
        raise _Uncertain
    return tuple(facets)


def _polyhedron(poly, rows, variables, context, policy):
    """Recursive rational facet charts; metric normal is n/abs(n_j)."""
    value = sp.S.Zero
    volume = sp.S.Zero
    faces = []
    u, v = sp.symbols("_facet_u _facet_v", real=True, cls=sp.Dummy)
    for row in rows:
        normal, offset = row[:-1], row[-1]
        index = next(i for i, a in enumerate(normal) if a)
        free = [i for i in range(3) if i != index]
        coords = [None] * 3
        coords[free[0]], coords[free[1]] = u, v
        coords[index] = (-offset - normal[free[0]] * u - normal[free[1]] * v) / normal[index]
        mapping = dict(zip(variables, coords, strict=True))
        target = sp.And(
            *(
                sp.Add(*[a * z for a, z in zip(r[:-1], coords, strict=True)], r[-1]) <= 0
                for r in rows
            )
        )
        if target == sp.false:
            continue
        lowered = _halfspaces(target, (u, v), context)
        if not lowered:
            continue
        edges = _convex_facets(lowered, tuple(context.original))
        if not edges:
            continue
        if not all(c.is_Rational for r in lowered for c in r):
            raise _Uncertain
        factor = -offset / abs(normal[index])
        area = _polygon_moment(edges, (0, 0))
        volume += factor * area / 3
        from .adapters.semialg import linear_metric_geometry

        certificate = linear_metric_geometry(
            sp.Matrix(coords).jacobian((u, v)), assumptions=context.original, policy=policy
        )
        faces.append((row, tuple(coords), edges, certificate))
        if (
            not policy.allow_expensive_exact
            and sum(len(face[2]) for face in faces) > policy.semialg.max_pieces
        ):
            raise _Uncertain
        for powers, coefficient in poly.terms():
            transformed = _polynomial(
                sp.prod(mapping[z] ** k for z, k in zip(variables, powers, strict=True)), (u, v)
            )
            if transformed is None:
                raise _Uncertain
            value += (
                factor
                * coefficient
                * sum(c * _polygon_moment(edges, p) for p, c in transformed.terms())
                / (sum(powers) + 3)
            )
    if volume <= 0:
        raise _Uncertain
    return sp.factor(value), tuple(faces)


def _exponential(poly, phase, facets, variables, context):
    rates = tuple(sp.diff(phase, z) for z in variables)
    index = next((i for i, a in enumerate(rates) if _ask(sp.Q.nonzero(a), context) is True), None)
    if index is None:
        raise _Uncertain
    rate = rates[index]
    field = sp.S.Zero
    derivative = poly.as_expr()
    while derivative != 0:
        field += derivative / rate
        derivative = -sp.diff(derivative, variables[index]) / rate
    t = sp.Dummy("boundary_parameter", real=True)
    value = sp.S.Zero
    for origin, tangent, lo, hi, normal in facets:
        if normal[index] == 0:
            continue
        mapping = dict(
            zip(variables, (p + t * d for p, d in zip(origin, tangent, strict=True)), strict=True)
        )
        density = (field * sp.exp(phase)).subs(mapping, simultaneous=True)
        edge_poly = _polynomial(density, (t,))
        moment = (
            _integral_polynomial(edge_poly, lo, hi)
            if edge_poly is not None
            else polynomial_function_moment(density, t, lo, hi, context)
        )
        if moment is None:
            raise _Uncertain
        value += normal[index] * moment
    return sp.factor(value)


@_with_policy
def integrate_boundary(
    f, formula, variables, *, assumptions=None, policy=None, measure_dimension="ambient"
):
    """Prove boundary geometry before exact polynomial or affine-exponential flux."""
    variables = tuple(variables)
    n = len(variables)
    policy = _normalize_policy(policy)
    if n not in (2, 3) or measure_dimension not in ("ambient", "intrinsic", "top", n):
        return None
    formula, f = sp.sympify(formula), sp.sympify(f)
    atoms = formula.atoms(Relational)
    if not policy.allow_expensive_exact and (
        len(atoms) > policy.semialg.max_boolean_atoms
        or len(atoms) > policy.semialg.max_pieces
        or n > policy.semialg.max_variables
    ):
        return None
    context = _normalize_assumptions(assumptions)
    exponentials = tuple(f.atoms(sp.exp))
    phase = None
    if exponentials:
        if n != 2 or len(exponentials) != 1:
            return None
        phase = exponentials[0].args[0]
        phase_poly = _polynomial(phase, variables)
        if (
            phase_poly is None
            or phase_poly.total_degree() > 1
            or not all(
                _ask(sp.Q.real(c) & sp.Q.finite(c), context) is True for c in phase_poly.coeffs()
            )
        ):
            return None
        poly = _polynomial(sp.cancel(f / exponentials[0]), variables)
        if phase_poly.total_degree() == 0:
            poly = _polynomial(f, variables)
            phase = None
    else:
        poly = _polynomial(f, variables)
    if poly is None or not all(_ask(sp.Q.finite(c), context) is True for c in poly.coeffs()):
        return None
    ambient_null = False
    try:
        if isinstance(formula, (sp.Or, sp.Not)) or any(
            isinstance(a, (sp.Or, sp.Not)) for a in sp.And.make_args(formula)
        ):
            if (
                n != 2
                or not policy.allow_expensive_exact
                and len(atoms) * (len(atoms) - 1) // 2 > policy.semialg.max_pieces
            ):
                return None
            facets = _loop_facets(formula, variables)
            rows = ()
        else:
            rows = _halfspaces(formula, variables, context)
            if not rows or len(rows) < n + 1:
                return None
            if all(sum(a != 0 for a in row[:-1]) == 1 for row in rows):
                return None
            if n == 3:
                if not all(c.is_Rational for row in rows for c in row):
                    return None
                value, facets = _polyhedron(poly, rows, variables, context, policy)
            else:
                facets = _convex_facets(rows, tuple(context.original))
                ambient_null = not facets and measure_dimension in ("ambient", n)
        if (
            not facets
            and not ambient_null
            or not policy.allow_expensive_exact
            and len(facets) > policy.semialg.max_pieces
        ):
            return None
        if ambient_null:
            value = sp.S.Zero
        elif n == 2:
            value = (
                _exponential(poly, phase, facets, variables, context)
                if phase is not None
                else sp.Add(*(c * _polygon_moment(facets, p) for p, c in poly.terms()))
            )
    except (_Uncertain, sp.PolynomialError, TypeError, ValueError):
        return None
    method = "affine_exponential_boundary" if phase is not None else "polynomial_boundary"
    details = (
        ("source_region", formula),
        ("facets", facets),
        ("halfspaces", rows),
        (
            "basis",
            "certified zero ambient area"
            if ambient_null
            else "whole-cell facet intersections or certified rational boundary arrangement",
        ),
    )
    return IntegrationResult(
        sp.factor(value),
        IntegrationStatus.EXACT,
        method,
        convergent=True,
        exact=True,
        complete=True,
        assumptions=context.predicate_expr,
        coverage=(
            CoverageEvidence("integration_domain", True, method, details),
            CoverageEvidence(
                "absolute_convergence",
                True,
                method,
                (("basis", "finite continuous density on a certified bounded domain closure"),),
            ),
        ),
        convergence_evidence=tuple(
            ConvergenceEvidence(kind, True, method) for kind in ("ordinary", "absolute")
        ),
        diagnostics={"boundary_facets": len(facets)},
    )
