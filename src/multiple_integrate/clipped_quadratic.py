"""Polynomial ellipsoid moments clipped by proved parallel affine half-spaces."""

from functools import lru_cache

import sympy as sp
from sympy.core.relational import Relational

from .assumptions import _ask, _normalize_assumptions
from .convergence_evidence import _finite
from .fast_rules import _polynomial
from .policy import _normalize_policy, _with_policy, check_semialg_budget
from .quadratic_radial import positive_quadratic
from .results import ConvergenceEvidence, CoverageEvidence, IntegrationResult, IntegrationStatus


@_with_policy
def integrate_clipped_quadratic_cell(
    f, formula, variables, *, assumptions=None, policy=None, measure_dimension="ambient"
):
    """A single SPD ellipsoid and any proved parallel linear slab constraints.

    General intersecting clips and intrinsic slices are  declined.
    """
    variables = tuple(variables)
    n = len(variables)
    context = _normalize_assumptions(assumptions)
    budget = _normalize_policy(policy)
    if (
        n < 2
        or not isinstance(formula, sp.And)
        or not check_semialg_budget(formula, variables, budget)[0]
    ):
        return None
    if measure_dimension not in ("ambient", "intrinsic", "top", n):
        return None

    def proved(predicate):
        return _ask(predicate, context) is True

    polynomial = _polynomial(sp.sympify(f), variables)
    if polynomial is None or not all(_finite(c, context) for c in polynomial.coeffs()):
        return None
    quadratic = None
    clips = []
    for atom in sp.And.make_args(formula):
        if not isinstance(atom, Relational) or atom.rel_op not in ("<", "<=", ">", ">="):
            return None
        q = atom.lhs - atom.rhs if atom.rel_op in ("<", "<=") else atom.rhs - atom.lhs
        poly = _polynomial(q, variables)
        if poly is None or not all(proved(sp.Q.real(c) & sp.Q.finite(c)) for c in poly.coeffs()):
            return None
        if poly.total_degree() == 2 and quadratic is None:
            quadratic = positive_quadratic(q, variables, assumptions)
            if quadratic is None:
                return None
        elif poly.total_degree() <= 1:
            clips.append((q, atom.rel_op in ("<", ">")))
        else:
            return None
    if quadratic is None or not clips:
        return None
    matrix, center, minimum = quadratic
    radius2 = -minimum
    # The point/empty ellipsoid cases stay with the uncut recognizer.
    if not proved(sp.Q.positive(radius2)):
        return None
    direction = None
    lower_bounds, upper_bounds = [], []
    for q, strict in clips:
        row = sp.Matrix([sp.diff(q, v) for v in variables])
        if all(a == 0 for a in row):
            if not proved(sp.Q.negative(q) if strict else sp.Q.nonpositive(q)):
                return None
            continue
        if direction is None:
            direction = row
            pivot = next((i for i, a in enumerate(row) if proved(sp.Q.nonzero(a))), None)
            if pivot is None:
                return None
        ratio = sp.cancel(row[pivot] / direction[pivot])
        if any(sp.simplify(a - ratio * b) != 0 for a, b in zip(row, direction, strict=True)):
            return None
        offset = sp.simplify(q.subs(dict(zip(variables, center, strict=True)), simultaneous=True))
        bound = sp.cancel(-offset / ratio)
        if not _finite(bound, context):
            return None
        if proved(sp.Q.positive(ratio)):
            upper_bounds.append(bound)
        elif proved(sp.Q.negative(ratio)):
            lower_bounds.append(bound)
        else:
            return None
    if direction is None:
        return None
    inverse = matrix.inv()
    variance = sp.simplify((direction.T * inverse * direction)[0])
    height = sp.sqrt(radius2 * variance)

    def extremum(values, maximum):
        best = values[0]
        for value in values[1:]:
            difference = sp.simplify(value - best)
            if proved(sp.Q.nonnegative(difference)):
                if maximum:
                    best = value
            elif proved(sp.Q.nonpositive(difference)):
                if not maximum:
                    best = value
            else:
                return None
        return best

    lower = extremum([-height, *lower_bounds], True)
    upper = extremum([height, *upper_bounds], False)
    if lower is None or upper is None:
        return None
    width = sp.simplify(upper - lower)
    if proved(sp.Q.nonpositive(width)):
        # Equal endpoints may define a lower-dimensional intrinsic slice.
        if (
            not proved(sp.Q.negative(width))
            and measure_dimension != "ambient"
            and measure_dimension != n
        ):
            return None
        value = sp.S.Zero
        dimension = None if proved(sp.Q.zero(width)) else -1
    elif proved(sp.Q.positive(width)):
        u = sp.Dummy("clip_coordinate", real=True)
        residual = tuple(sp.Dummy("clip_residual", real=True) for _ in variables)
        mean = inverse * direction / variance
        shifted = _polynomial(
            sp.sympify(f).subs(
                {
                    v: c + b * u + z
                    for v, c, b, z in zip(variables, center, mean, residual, strict=True)
                },
                simultaneous=True,
            ),
            (u, *residual),
        )
        if shifted is None:
            return None
        covariance = (inverse - (inverse * direction) * (direction.T * inverse) / variance) / 2

        @lru_cache(None)
        def wick(powers):
            if not any(powers):
                return sp.S.One
            if sum(powers) % 2:
                return sp.S.Zero
            i = next(i for i, p in enumerate(powers) if p)
            remaining = list(powers)
            remaining[i] -= 1
            total = sp.S.Zero
            for j, count in enumerate(remaining):
                if count:
                    reduced = remaining.copy()
                    reduced[j] -= 1
                    total += count * covariance[i, j] * wick(tuple(reduced))
            return total

        def primitive(endpoint, power, exponent):
            if proved(sp.Q.zero(endpoint)):
                return sp.S.Zero
            if proved(sp.Q.positive(endpoint)):
                sign = 1
            elif proved(sp.Q.negative(endpoint)):
                sign = (-1) ** (power + 1)
            else:
                return None
            z = sp.simplify(endpoint**2 / height**2)
            angular = (
                sp.gamma(sp.Rational(power + 1, 2))
                * sp.gamma(exponent + 1)
                / sp.gamma(sp.Rational(power + 1, 2) + exponent + 1)
                if z == 1
                else sp.betainc(sp.Rational(power + 1, 2), exponent + 1, 0, z)
            )
            return sign * height ** (power + 1) * radius2**exponent * angular / 2

        value = sp.S.Zero
        for (power, *powers), coefficient in shifted.terms():
            weight = wick(tuple(powers))
            if weight == 0:
                continue
            exponent = sp.Rational(n - 1 + sum(powers), 2)
            left, right = primitive(lower, power, exponent), primitive(upper, power, exponent)
            if left is None or right is None:
                return None
            value += (
                coefficient
                * weight
                * sp.pi ** sp.Rational(n - 1, 2)
                / (sp.sqrt(matrix.det() * variance) * sp.gamma(exponent + 1))
                * (right - left)
            )
        dimension = n
    else:
        return None
    return IntegrationResult(
        sp.expand_func(value),
        IntegrationStatus.EXACT,
        "clipped_quadratic_cell",
        convergent=True,
        complete=True,
        exact=True,
        assumptions=sp.And(*context.original),
        coverage=(
            CoverageEvidence(
                "integration_domain",
                True,
                "clipped_quadratic_cell",
                (
                    ("basis", "SPD conditional slices and proved parallel slab bounds"),
                    ("clip_direction", tuple(direction)),
                    ("clip_interval", (lower, upper)),
                ),
            ),
        ),
        convergence_evidence=tuple(
            ConvergenceEvidence(kind, True, "compact_polynomial_ellipsoid_slab")
            for kind in ("ordinary", "absolute")
        ),
        diagnostics={
            "dimension": dimension,
            "clip_interval": (lower, upper),
            "clip_variance": variance,
        },
    )
