"""One-dimensional singularity decomposition for exact improper integration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key

import sympy as sp

from multiple_integrate.adapters import function_singularities
from multiple_integrate.adapters.semialg import is_satisfiable as semialg_is_satisfiable
from multiple_integrate.adapters.semialg import region_dimension as semialg_region_dimension
from multiple_integrate.assumptions import (
    _ask,
    _condition_to_predicate,
    _integrate,
    _normalize_assumptions,
)
from multiple_integrate.conditional import conditional_status
from multiple_integrate.planner import record_coverage, record_strategy
from multiple_integrate.results import IntegrationStatus


def _point_from_condition(condition: object, var: sp.Symbol) -> sp.Expr | None:
    cond = sp.sympify(condition)
    if isinstance(cond, sp.Equality):
        if cond.lhs == var and not cond.rhs.has(var):
            return sp.sympify(cond.rhs)
        if cond.rhs == var and not cond.lhs.has(var):
            return sp.sympify(cond.lhs)
        try:
            solutions = sp.solve(cond, var)
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            return None
        if len(solutions) == 1 and not sp.sympify(solutions[0]).has(var):
            return sp.sympify(solutions[0])
    return None


def _records(result: object):
    records = getattr(result, "records", None)
    if records is not None:
        for record in records:
            kind = str(getattr(getattr(record, "kind", None), "value", "")).lower()
            if "remov" in kind or "cut" in kind:
                continue
            yield record
        return
    if isinstance(result, dict):
        for key, values in result.items():
            name = str(key).lower()
            if "remov" in name or "cut" in name:
                continue
            if isinstance(values, (tuple, list, set, sp.Tuple)):
                yield from values
            elif values is not None:
                yield values


def _needs_backend_query(expr: sp.Expr, var: sp.Symbol) -> bool:
    """Whether funcprops can add singularity information for this 1-D integrand.

    Polynomials are entire, so querying a global singularity engine for them is
    pure overhead. Other expression families remain cautious and continue
    through funcprops unless a future local proof establishes the same fact.
    """
    try:
        return not sp.sympify(expr).is_polynomial(var)
    except (TypeError, ValueError, sp.PolynomialError):
        return True


def singular_points(
    expr: sp.Expr,
    var: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    *,
    assumptions=None,
) -> tuple[sp.Expr, ...]:
    """Return certified finite singular points strictly inside an interval."""
    if not _needs_backend_query(expr, var):
        return ()
    context = None if assumptions is None else _normalize_assumptions(assumptions)
    domain = sp.Interval(lower, upper)
    result = function_singularities(expr, var, domain=domain, assumptions=assumptions)
    points: list[sp.Expr] = []
    for record in _records(result):
        guard = getattr(record, "guard", sp.true)
        if (
            guard not in (True, sp.true)
            and _ask(_condition_to_predicate(guard), context) is not True
        ):
            continue
        condition = getattr(record, "condition", record)
        point = _point_from_condition(condition, var)
        if point is None or point in (-sp.oo, sp.oo) or point.is_real is False:
            continue
        left = _ask(sp.Q.positive(point - lower), context) if lower != -sp.oo else True
        right = _ask(sp.Q.positive(upper - point), context) if upper != sp.oo else True
        if left is True and right is True and point not in points:
            points.append(point)

    def compare(left, right):
        difference = sp.simplify(left - right)
        if difference.is_zero is True:
            return 0
        if difference.is_negative is True:
            return -1
        if difference.is_positive is True:
            return 1
        if _ask(sp.Q.negative(difference), context) is True:
            return -1
        if _ask(sp.Q.positive(difference), context) is True:
            return 1
        raise ValueError("singular point order could not be certified")

    try:
        return tuple(sorted(points, key=cmp_to_key(compare)))
    except ValueError:
        return ()


def split_integral(
    expr: sp.Expr,
    var: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    opts: dict,
    *,
    assumptions=None,
    return_evidence=False,
) -> sp.Expr | None:
    """Integrate interval pieces separated by funcprops-certified singularites."""
    points = singular_points(expr, var, lower, upper, assumptions=assumptions)
    if not points:
        return None
    endpoints = (lower, *points, upper)
    values = [
        _integrate(expr, (var, lo, hi), **opts)
        for lo, hi in zip(endpoints, endpoints[1:], strict=False)
    ]
    divergent = any(conditional_status(value)[0] is IntegrationStatus.DIVERGENT for value in values)
    if divergent:
        record_strategy(
            "divergent_interval_piece",
            reason="a one-sided integral at a certified interior singularity diverges",
        )
    record_coverage(
        "interval_partition",
        True,
        "singularity_split",
        pieces=len(values),
        basis="certified ordered interior points; all interval pieces consumed",
    )
    value = sp.simplify(sp.Add(*values))
    return (value, divergent) if return_evidence else value


__all__ = ["SingularLocusStratum", "singular_points", "split_integral", "stratify_singular_locus"]


@dataclass(frozen=True)
class SingularLocusStratum:
    """One semialgebraic stratum of an algebraic singular locus."""

    condition: sp.Expr
    dimension: int | None
    codimension: int | None
    components: tuple[sp.Expr, ...]
    regular: bool | None
    method: str


def _denominator_components(expr: sp.Expr, variables: tuple[sp.Symbol, ...]) -> tuple[sp.Expr, ...]:
    denominator = sp.cancel(sp.sympify(expr)).as_numer_denom()[1]
    try:
        _coefficient, factors = sp.factor_list(denominator, *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return ()
    components: list[sp.Expr] = []
    for factor, _multiplicity in factors:
        if any(factor.has(variable) for variable in variables):
            components.append(sp.expand(factor))
    return tuple(components)


@dataclass(frozen=True)
class SingularDiagnostics:
    """Budgeted auxiliary geometry, never a prerequisite for evaluation."""

    strata: tuple[SingularLocusStratum, ...]
    complete: bool
    queries: int
    limit: int
    pruned: int
    factors_certified: bool = True

    @property
    def geometry_certified(self):
        """Whether all algebraic denominator geometry fields were certified."""
        return (
            self.complete
            and self.factors_certified
            and all(s.dimension is not None and s.regular is not None for s in self.strata)
        )


def stratify_singular_locus(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...] | list[sp.Symbol],
    *,
    domain: sp.Expr = sp.true,
    max_queries: int | None = None,
    return_details: bool = False,
) -> tuple[SingularLocusStratum, ...] | SingularDiagnostics:
    """Inspect pole intersections with empty-superset pruning and a query budget.

    These closed loci may overlap; they are not a disjoint integration cover.
    Unknown geometry is retained. Exhausting this auxiliary budget does not
    invalidate an integral value; the returned diagnostics are marked partial.
    """
    from itertools import combinations
    from math import comb

    from .policy import _CURRENT_POLICY

    limit = _CURRENT_POLICY.get().max_singular_queries if max_queries is None else max_queries
    if type(limit) is not int or limit < 0:
        raise ValueError("max_queries must be a nonnegative integer")
    vars_ = tuple(variables)
    components = _denominator_components(sp.sympify(expr), vars_)
    try:
        sp.Poly(sp.cancel(expr).as_numer_denom()[1], *vars_)
        factors_certified = True
    except sp.PolynomialError:
        factors_certified = False
    strata: list[SingularLocusStratum] = []
    queries = pruned = 0
    complete = True
    candidates = {(i,) for i in range(len(components))}
    while candidates:
        survivors = set()
        for indices in sorted(candidates):
            if queries >= limit:
                complete = False
                break
            selected = tuple(components[i] for i in indices)
            original_condition = sp.And(sp.sympify(domain), *(sp.Eq(item, 0) for item in selected))
            real_forms = []
            real_geometry = True
            for item in selected:
                polynomial = sp.Poly(item, *vars_)
                if all(_ask(sp.Q.real(c) & sp.Q.finite(c)) is True for c in polynomial.coeffs()):
                    real_forms.append(item)
                    continue
                # Coordinates range over R regardless of their Symbol assumptions.
                fresh = {v: sp.Dummy(v.name, real=True) for v in vars_}
                real, imaginary = polynomial.as_expr().xreplace(fresh).as_real_imag()
                reverse = {u: v for v, u in fresh.items()}
                for part in (real, imaginary):
                    part = sp.expand(part.xreplace(reverse))
                    try:
                        coefficients = sp.Poly(part, *vars_).coeffs()
                    except sp.PolynomialError:
                        real_geometry = False
                        break
                    if not all(_ask(sp.Q.real(c) & sp.Q.finite(c)) is True for c in coefficients):
                        real_geometry = False
                        break
                    if part != 0 and part not in real_forms:
                        real_forms.append(part)
                if not real_geometry:
                    break
            if not real_geometry:
                factors_certified = False
                survivors.add(indices)
                strata.append(
                    SingularLocusStratum(
                        original_condition, None, None, selected, None, "algebraic_factorization"
                    )
                )
                continue
            condition = sp.And(sp.sympify(domain), *(sp.Eq(item, 0) for item in real_forms))
            queries += 1
            try:
                dimension = semialg_region_dimension(condition, vars_)
            except (
                ValueError,
                TypeError,
                NotImplementedError,
                ArithmeticError,
                sp.polys.polyerrors.BasePolynomialError,
            ):
                dimension = None
            if dimension is not None and dimension < 0:
                pruned += 1
                continue
            survivors.add(indices)
            codimension = None if dimension is None else len(vars_) - dimension
            regular: bool | None = None
            rank = codimension if codimension is not None else (1 if len(real_forms) == 1 else None)
            if (
                rank is not None
                and 0 < rank <= min(len(real_forms), len(vars_))
                and queries < limit
            ):
                # Full rank on the zero set certifies a smooth implicit locus,
                # including transverse intersections and complex pole equations.
                matrix = sp.Matrix([[sp.diff(item, v) for v in vars_] for item in real_forms])
                minors = []
                budget = _CURRENT_POLICY.get()
                count = comb(len(real_forms), rank) * comb(len(vars_), rank)
                if budget.allow_expensive_exact or count <= budget.hypersurface.max_pieces:
                    for rows in combinations(range(len(real_forms)), rank):
                        for columns in combinations(range(len(vars_)), rank):
                            minors.append(matrix.extract(rows, columns).det())
                    # Keep separate equations so linear minors remain cheap for the backend.
                    critical = sp.And(condition, *(sp.Eq(m, 0) for m in minors))
                    queries += 1
                    try:
                        certified = semialg_is_satisfiable(critical, vars_)
                    except (
                        ValueError,
                        TypeError,
                        NotImplementedError,
                        ArithmeticError,
                        sp.polys.polyerrors.BasePolynomialError,
                    ):
                        certified = None
                    regular = None if certified is None else certified is False
                else:
                    complete = False
            elif rank is not None and queries >= limit:
                complete = False
            strata.append(
                SingularLocusStratum(
                    condition,
                    dimension,
                    codimension,
                    selected,
                    regular,
                    "semialg_dimension" if dimension is not None else "algebraic_factorization",
                )
            )
        if not complete:
            break
        # Only extend nonempty/unknown intersections whose proper subsets survive.
        # Certified empty subsets eliminate entire branches before construction.
        candidates = {
            (*parent, j)
            for parent in survivors
            for j in range(parent[-1] + 1, len(components))
            if all(subset in survivors for subset in combinations((*parent, j), len(parent)))
        }
    strata.sort(
        key=lambda item: (
            -(item.codimension if item.codimension is not None else -1),
            sp.default_sort_key(item.condition),
        )
    )
    details = SingularDiagnostics(
        tuple(strata), complete, queries, limit, pruned, factors_certified
    )
    return details if return_details else details.strata
