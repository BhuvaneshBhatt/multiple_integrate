"""Polynomial moments on proved quadratic parameter cells."""

from __future__ import annotations

import sympy as sp
from sympy.core.relational import Relational

from .assumptions import _ask, _normalize_assumptions
from .convergence_evidence import _finite
from .fast_rules import _polynomial
from .policy import _normalize_policy, _with_policy, check_semialg_budget
from .quadratic_radial import _evaluate, positive_quadratic
from .results import ConvergenceEvidence, CoverageEvidence, IntegrationResult, IntegrationStatus


@_with_policy
def integrate_quadratic_cell(
    f, formula, variables, *, assumptions=None, policy=None, measure_dimension="ambient"
):
    """A single SPD quadratic sublevel, with proved empty/point/full fibers.

    This does not certify clipped ellipsoids or intrinsic curved surfaces.
    """
    if not isinstance(formula, Relational) or formula.rel_op not in ("<", "<=", ">", ">="):
        return None
    variables = tuple(variables)
    budget = _normalize_policy(policy)
    if len(variables) < 2 or not check_semialg_budget(formula, variables, budget)[0]:
        return None
    q = formula.lhs - formula.rhs if formula.rel_op in ("<", "<=") else formula.rhs - formula.lhs
    data = positive_quadratic(q, variables, assumptions)
    if data is None:
        return None
    matrix, center, minimum = data
    radius_squared = -minimum
    context = _normalize_assumptions(assumptions)
    polynomial = _polynomial(sp.sympify(f), variables)
    if polynomial is None or not all(_finite(c, context) for c in polynomial.coeffs()):
        return None
    point = _ask(sp.Q.zero(radius_squared), context) is True
    empty = _ask(sp.Q.negative(radius_squared), context) is True or (
        point and formula.rel_op in ("<", ">")
    )
    full = _ask(sp.Q.positive(radius_squared), context) is True
    if not (empty or point or full):
        return None
    n = len(variables)
    if measure_dimension not in ("ambient", "intrinsic", "top", n, 0):
        return None
    if full and measure_dimension == 0:
        return None
    if empty or (point and measure_dimension in ("ambient", n)):
        value = sp.S.Zero
        dimension = -1 if empty else 0
    elif point:
        value = polynomial.as_expr().subs(
            dict(zip(variables, center, strict=True)), simultaneous=True
        )
        if not _finite(value, context):
            return None
        dimension = 0
    else:
        local = tuple(sp.Dummy("ellipsoid_coordinate", real=True) for _ in variables)
        shifted = _polynomial(
            polynomial.as_expr().subs(
                dict(
                    zip(variables, (u + c for u, c in zip(local, center, strict=True)), strict=True)
                ),
                simultaneous=True,
            ),
            local,
        )
        if shifted is None:
            return None
        # Reuse the bounded covariance/Wick recurrence of the radial engine.
        value = _evaluate(matrix, shifted, sp.S.One, None, None, sp.S.Zero, sp.sqrt(radius_squared))
        dimension = n
    return IntegrationResult(
        sp.simplify(value),
        IntegrationStatus.EXACT,
        "quadratic_parameter_cell",
        convergent=True,
        complete=True,
        exact=True,
        assumptions=sp.And(*context.original),
        coverage=(
            CoverageEvidence(
                "integration_domain",
                True,
                "quadratic_parameter_cell",
                (
                    ("basis", "proved SPD quadratic and critical-level sign"),
                    ("dimension", dimension),
                ),
            ),
        ),
        convergence_evidence=tuple(
            ConvergenceEvidence(kind, True, "compact_polynomial_ellipsoid")
            for kind in ("ordinary", "absolute")
        ),
        diagnostics={"dimension": dimension, "center": center, "radius_squared": radius_squared},
    )
