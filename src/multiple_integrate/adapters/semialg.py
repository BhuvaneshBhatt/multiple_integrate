"""Typed lazy adapter for :mod:`semialg` geometry services."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.polys.polyerrors import BasePolynomialError

from multiple_integrate.semialg_geometry import domain_geometry

_QUERY_ERRORS = (
    ImportError,
    NotImplementedError,
    ArithmeticError,
    BasePolynomialError,
)


class SemialgContractError(RuntimeError):
    """Raised when semialg returns an object that violates the adapter contract."""


def _required(obj: object, name: str, *, context: str):
    try:
        return getattr(obj, name)
    except AttributeError as exc:
        raise SemialgContractError(f"{context} is missing required attribute {name!r}") from exc


def _required_callable(obj: object, name: str, *, context: str):
    value = _required(obj, name, context=context)
    if not callable(value):
        raise SemialgContractError(f"{context}.{name} must be callable")
    return value


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SemialgContractError(f"{context} must be a dict")
    return dict(value)


@dataclass(frozen=True)
class DomainPiece:
    integrand: sp.Expr
    limits: tuple[tuple[sp.Symbol, sp.Expr, sp.Expr], ...]
    method: str
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class IntrinsicPiece:
    integrand: sp.Expr
    limits: tuple[tuple[sp.Symbol, sp.Expr, sp.Expr], ...]
    dimension: int
    ambient_dimension: int
    metric_factor: sp.Expr
    cell_index: tuple[int, ...]
    method: str
    diagnostics: dict[str, object]
    direct_value: sp.Expr | None = None
    chart_mapping: tuple[tuple[sp.Symbol, sp.Expr], ...] = ()
    chart_variables: tuple[sp.Symbol, ...] = ()
    jacobian: sp.MatrixBase | None = None
    gram_matrix: sp.MatrixBase | None = None


@dataclass(frozen=True)
class SingularStratum:
    """Uncertified target-dimensional CAD stratum reported by semialg."""

    cell_index: tuple[int, ...]
    dimension: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntrinsicReduction:
    condition: sp.Expr
    variables: tuple[sp.Symbol, ...]
    dimension: int
    pieces: tuple[IntrinsicPiece, ...]
    singular_strata: tuple[SingularStratum, ...]
    method: str
    diagnostics: dict[str, object]

    @property
    def certified(self) -> bool:
        return not self.singular_strata


@dataclass(frozen=True)
class DomainReduction:
    condition: sp.Expr
    variables: tuple[sp.Symbol, ...]
    pieces: tuple[DomainPiece, ...]
    method: str
    diagnostics: dict[str, object]


def _module():
    try:
        import semialg
    except ImportError:
        return None
    return semialg


def semialg_available() -> bool:
    """Return whether the semialg backend can be imported."""
    return _module() is not None


def _intrinsic_parametric_cover(*args, **kwargs):
    """Call semialg's public specialist intrinsic-cover API lazily."""
    try:
        from semialg.parametric_geometry import intrinsic_parametric_cover
    except ImportError:
        return None
    from ..isolated_queries import isolated_query
    from ..policy import _CURRENT_POLICY

    seconds = _CURRENT_POLICY.get().backend_timeout_seconds
    if seconds is not None:
        return isolated_query(
            "semialg.parametric_geometry", "intrinsic_parametric_cover", args, kwargs, seconds
        )
    return intrinsic_parametric_cover(*args, **kwargs)


def _call(name: str, *args, **kwargs):
    module = _module()
    if module is None:
        return None
    query = getattr(module, name, None)
    if query is None:
        return None
    try:
        from ..isolated_queries import isolated_query
        from ..policy import _CURRENT_POLICY

        seconds = _CURRENT_POLICY.get().backend_timeout_seconds
        if seconds is not None:
            return isolated_query("semialg", name, args, kwargs, seconds)
        return query(*args, **kwargs)
    except _QUERY_ERRORS:
        return None


def region_dimension(condition: object, variables: tuple[sp.Symbol, ...]) -> int | None:
    """Return the exact semialgebraic dimension when the backend can certify it."""
    try:
        measured = _call("region_dimension", sp.sympify(condition), tuple(variables))
    except ValueError:
        return None
    return None if measured is None else int(measured)


def is_satisfiable(condition: object, variables: tuple[sp.Symbol, ...]) -> bool | None:
    """Return a certified satisfiability result when available."""
    module = _module()
    if module is None:
        return None
    query = getattr(module, "is_satisfiable", None)
    if query is None:
        return None
    try:
        result = _call("is_satisfiable", sp.sympify(condition), tuple(variables))
    except _QUERY_ERRORS + (TypeError, ValueError):
        return None
    return result if result in (True, False) else None


def implies(premise: object, conclusion: object, variables: tuple[sp.Symbol, ...]) -> bool | None:
    """Return whether a semialgebraic implication can be certified."""
    module = _module()
    if module is None:
        return None
    query = getattr(module, "implies", None)
    if query is None:
        return None
    try:
        result = _call("implies", sp.sympify(premise), sp.sympify(conclusion), tuple(variables))
    except _QUERY_ERRORS + (TypeError, ValueError):
        return None
    return result if result in (True, False) else None


def semialgebraic_domain(
    region: object, variables: tuple[sp.Symbol, ...] | list[sp.Symbol] | None = None
) -> tuple[sp.Expr, tuple[sp.Symbol, ...]]:
    """Normalize a Boolean formula or semialg ``SemialgebraicRegion``.

    Raw formulas require explicit integration variables so free parameters are
    never accidentally promoted to integration coordinates.
    """
    if variables is not None:
        variables = _domain_variables(variables)
    formula = getattr(region, "formula", None)
    owned_variables = getattr(region, "variables", None)
    if formula is not None and owned_variables is not None:
        owned_variables = _domain_variables(owned_variables)
        if variables is not None and variables != owned_variables:
            raise ValueError("variables= must match the semialgebraic region's coordinate order")
        qf = getattr(region, "quantifier_free_formula", None)
        formula = qf() if callable(qf) else formula
        return sp.sympify(formula), owned_variables

    geometry = domain_geometry(region)
    if geometry is not None:
        owned = tuple(getattr(region, "variables", ()))
        vars_ = tuple(variables) if variables is not None else owned
        if not vars_:
            raise ValueError("variables= is required to lower this integration domain")
        symbolic = geometry.as_semialgebraic_region(vars_)
        qf = getattr(symbolic, "quantifier_free_formula", None)
        formula = qf() if callable(qf) else symbolic.formula
        return sp.sympify(formula), tuple(map(sp.sympify, vars_))

    # Canonical semialg Geometry objects do not own coordinate
    # symbols; caller supplies them when lowering to a formula.
    as_symbolic = getattr(region, "as_semialgebraic_region", None)
    if callable(as_symbolic):
        if variables is None:
            owned_variables = getattr(region, "variables", None)
            if owned_variables is not None:
                variables = tuple(owned_variables)
        if variables is not None:
            symbolic = (
                as_symbolic(tuple(variables)) if not hasattr(region, "ranges") else as_symbolic()
            )
            if symbolic is not None:
                qf = getattr(symbolic, "quantifier_free_formula", None)
                lowered = qf() if callable(qf) else getattr(symbolic, "formula", None)
                if lowered is not None:
                    return sp.sympify(lowered), tuple(map(sp.sympify, variables))
    if variables is None:
        raise ValueError("variables= is required when region is a raw semialgebraic condition")
    return sp.sympify(region), _domain_variables(variables)


def _domain_variables(variables) -> tuple[sp.Symbol, ...]:
    """Validate coordinates consistently for both owned and raw domains."""
    vars_ = tuple(map(sp.sympify, variables))
    if not vars_ or any(not isinstance(var, sp.Symbol) for var in vars_):
        raise ValueError("variables must contain one or more SymPy Symbol objects")
    if len(set(vars_)) != len(vars_):
        raise ValueError("semialgebraic integration variables must be distinct")
    return vars_


def reduce_domain(
    integrand: object,
    region: object,
    *,
    variables: tuple[sp.Symbol, ...] | list[sp.Symbol] | None = None,
    bounds=None,
    parameters=None,
) -> DomainReduction | None:
    """Ask semialg for certified cylindrical pieces without evaluating them."""
    formula, vars_ = semialgebraic_domain(region, variables)
    kwargs = {"bounds": bounds}
    if parameters is None:
        # Ambient coordinates are explicit; other symbols are parameters, not
        # extra integration dimensions for Semialg's variable normalizer.
        inferred = (formula.free_symbols | sp.sympify(integrand).free_symbols) - set(vars_)
        parameters = tuple(sorted(inferred, key=sp.default_sort_key)) or None
    if parameters is not None:
        kwargs["parameters"] = parameters
    reduced = _call(
        "reduce_region_integral",
        sp.sympify(integrand),
        formula,
        vars_,
        **kwargs,
    )
    if reduced is None:
        return None
    raw_pieces = _required(reduced, "pieces", context="semialg domain reduction")
    pieces: list[DomainPiece] = []
    for index, piece in enumerate(raw_pieces):
        context = f"semialg domain piece {index}"
        limits = tuple(
            (sp.sympify(var), sp.sympify(lo), sp.sympify(hi))
            for var, lo, hi in _required(piece, "limits", context=context)
        )
        pieces.append(
            DomainPiece(
                integrand=sp.sympify(_required(piece, "integrand", context=context)),
                limits=limits,
                method=str(_required(piece, "method", context=context)),
                diagnostics=_mapping(
                    _required(piece, "diagnostics", context=context),
                    context=f"{context}.diagnostics",
                ),
            )
        )
    return DomainReduction(
        condition=sp.sympify(_required(reduced, "condition", context="semialg domain reduction")),
        variables=tuple(_required(reduced, "variables", context="semialg domain reduction")),
        pieces=tuple(pieces),
        method=str(_required(reduced, "method", context="semialg domain reduction")),
        diagnostics=_mapping(
            _required(reduced, "diagnostics", context="semialg domain reduction"),
            context="semialg domain reduction.diagnostics",
        ),
    )


def reduce_intrinsic_domain(
    integrand: object,
    region: object,
    *,
    variables: tuple[sp.Symbol, ...] | list[sp.Symbol] | None = None,
    dimension: int | None = None,
) -> IntrinsicReduction | None:
    """Return exact semialg-owned intrinsic chart pieces without evaluating them.

    This boundary consumes :func:`semialg.parametric_geometry.intrinsic_parametric_cover` rather
    than reaching into semialg's CAD internals. Canonical geometry and CAD-
    derived strata therefore share one public chart contract.
    """
    module = _module()
    if module is None:
        return None

    vars_hint = variables
    if vars_hint is None:
        owned = getattr(region, "variables", None)
        if owned is not None:
            vars_hint = tuple(owned)

    geometry = domain_geometry(region) or region

    try:
        if vars_hint is None:
            symbolic = getattr(region, "formula", None)
            if symbolic is not None:
                vars_hint = tuple(getattr(region, "variables", ()))
        cover = _intrinsic_parametric_cover(
            geometry, vars_hint, dimension=dimension, require_verified=True
        )
        if cover is None:
            return None
    except _QUERY_ERRORS + (ValueError,):
        return None

    vars_ = tuple(map(sp.sympify, _required(cover, "variables", context="semialg intrinsic cover")))
    target = dimension
    if target is None:
        certified = _required_callable(
            cover, "certified_dimension", context="semialg intrinsic cover"
        )
        measured = certified()
        if measured is None:
            dimension_method = getattr(geometry, "dimension", None)
            measured = dimension_method() if callable(dimension_method) else None
        if measured is None:
            return None
        target = int(measured)

    expr = sp.sympify(integrand)
    pieces: list[IntrinsicPiece] = []
    for index, chart in enumerate(_required(cover, "charts", context="semialg intrinsic cover")):
        raw_chart_vars = tuple(
            map(sp.sympify, _required(chart, "parameters", context=f"semialg chart {index}"))
        )
        # semialg may encode geometric facts such as r>=0 in Symbol assumptions.
        # MultipleIntegrate's singularity/convergence machinery treats integration
        # variables as neutral real coordinates and gets those facts from bounds,
        # so normalize assumed chart symbols at this package boundary.
        replacements = {
            var: sp.Dummy(f"mi_chart_{index}_{pos}", real=True)
            for pos, var in enumerate(raw_chart_vars)
            if isinstance(var, sp.Symbol)
            and var.assumptions0 != sp.Symbol("_", real=True).assumptions0
        }
        chart_vars = tuple(replacements.get(var, var) for var in raw_chart_vars)
        limits = tuple(
            (
                replacements.get(sp.sympify(var), sp.sympify(var)),
                sp.sympify(lo).xreplace(replacements),
                sp.sympify(hi).xreplace(replacements),
            )
            for var, lo, hi in _required(chart, "bounds", context=f"semialg chart {index}")
        )
        mapping = tuple(
            sp.sympify(value).xreplace(replacements)
            for value in _required(chart, "mapping", context=f"semialg chart {index}")
        )
        chart_integrand = _required_callable(
            chart, "intrinsic_integrand", context=f"semialg chart {index}"
        )(expr, vars_).xreplace(replacements)
        chart_condition = sp.sympify(
            _required(chart, "condition", context=f"semialg chart {index}")
        ).xreplace(replacements)
        if chart_condition != sp.true:
            chart_integrand = sp.Piecewise(
                (chart_integrand, chart_condition),
                (sp.Integer(0), True),
            )
        jacobian = sp.ImmutableMatrix(
            _required_callable(chart, "jacobian_matrix", context=f"semialg chart {index}")()
        ).xreplace(replacements)
        gram_matrix = sp.ImmutableMatrix(
            _required_callable(chart, "gram_matrix", context=f"semialg chart {index}")()
        ).xreplace(replacements)
        metric_factor = sp.sympify(
            _required_callable(chart, "metric_factor", context=f"semialg chart {index}")()
        ).xreplace(replacements)
        pieces.append(
            IntrinsicPiece(
                integrand=chart_integrand,
                limits=limits,
                dimension=int(
                    _required(chart, "domain_dimension", context=f"semialg chart {index}")
                    if _required(chart, "domain_dimension", context=f"semialg chart {index}")
                    is not None
                    else target
                ),
                ambient_dimension=int(
                    _required(chart, "ambient_dimension", context=f"semialg chart {index}")
                ),
                metric_factor=metric_factor,
                chart_mapping=tuple(zip(vars_, mapping, strict=True)),
                chart_variables=chart_vars,
                jacobian=jacobian,
                gram_matrix=gram_matrix,
                cell_index=(index,),
                method=f"semialg_parametric_{_required(chart, 'source', context=f'semialg chart {index}')}",
                diagnostics={
                    "regular": True,
                    "chart_source": _required(chart, "source", context=f"semialg chart {index}"),
                },
            )
        )
    formula = sp.sympify(_required(cover, "formula", context="semialg intrinsic cover"))
    return IntrinsicReduction(
        condition=formula,
        variables=vars_,
        dimension=int(target),
        pieces=tuple(pieces),
        singular_strata=(),
        method="semialg_intrinsic_parametric_cover",
        diagnostics={"chart_count": len(pieces), "target_dimension": int(target)},
    )


def region_condition(region: object) -> sp.Expr | None:
    """Return semialg-owned geometry formulas before range-based fallbacks."""
    lower = getattr(region, "as_semialgebraic_region", None)
    if callable(lower):
        try:
            symbolic = lower()
        except (TypeError, ValueError, NotImplementedError):
            symbolic = None
        if symbolic is not None:
            qf = getattr(symbolic, "quantifier_free_formula", None)
            formula = qf() if callable(qf) else getattr(symbolic, "formula", None)
            if formula is not None:
                return sp.sympify(formula)
    pieces = getattr(region, "pieces", None)
    if pieces:
        conditions = [region_condition(piece) for piece in pieces]
        if any(condition is None for condition in conditions):
            return None
        return sp.Or(*conditions)
    ranges = getattr(region, "ranges", None)
    if not ranges:
        return None
    atoms = []
    for var, lo, hi in ranges:
        var, lo, hi = sp.sympify(var), sp.sympify(lo), sp.sympify(hi)
        if lo != -sp.oo:
            atoms.append(sp.Ge(var, lo))
        if hi != sp.oo:
            atoms.append(sp.Le(var, hi))
    return sp.And(*atoms) if atoms else sp.true


def _has_positive_orientation(region: object, assumptions=None) -> bool:
    predicate = getattr(assumptions, "predicate_expr", sp.true)
    for _, lo, hi in tuple(getattr(region, "ranges", ())):
        lo = sp.sympify(lo)
        hi = sp.sympify(hi)
        if lo == -sp.oo or hi == sp.oo:
            continue
        diff = sp.simplify(hi - lo)
        if diff.is_nonnegative is True:
            continue
        try:
            if sp.ask(sp.Q.nonnegative(diff), predicate) is True:
                continue
        except (TypeError, ValueError):
            pass
        return False
    return True


def _coordinate_bounds_survive_assumptions(variables) -> bool:
    # Sign assumptions can simplify explicit bounds away before delegation.
    # Keep those coordinates on local iterated/formula paths instead.
    return all(
        variable.is_nonnegative is not True and variable.is_nonpositive is not True
        for variable in variables
    )


def semialgebraic_measure(region: object, *, assumptions=None) -> sp.Expr | None:
    condition = region_condition(region)
    variables = tuple(getattr(region, "variables", ()))
    if (
        condition is None
        or not variables
        or bool(condition.free_symbols - set(variables))
        or not _coordinate_bounds_survive_assumptions(variables)
        or not _has_positive_orientation(region, assumptions)
    ):
        return None
    result = _call("semialgebraic_measure", condition, variables)
    if result is None:
        return None
    if isinstance(result, (sp.Basic, int, float)):
        return sp.sympify(result)
    value = _required(result, "measure", context="semialg measure result")
    return sp.sympify(value)


def region_moment(region: object, expr: object, *, assumptions=None) -> sp.Expr | None:
    condition = region_condition(region)
    variables = tuple(getattr(region, "variables", ()))
    expr = sp.sympify(expr)
    if (
        condition is None
        or not variables
        or bool(condition.free_symbols - set(variables))
        or not _coordinate_bounds_survive_assumptions(variables)
        or not _has_positive_orientation(region, assumptions)
    ):
        return None
    try:
        sp.Poly(expr, *variables)
    except sp.PolynomialError:
        return None
    result = _call("region_moment", condition, variables, integrand=expr)
    if result is None:
        return None
    if isinstance(result, (sp.Basic, int, float)):
        return sp.sympify(result)
    return sp.sympify(_required(result, "value", context="semialg moment result"))


__all__ = [
    "SemialgContractError",
    "SingularStratum",
    "IntrinsicPiece",
    "IntrinsicReduction",
    "DomainPiece",
    "DomainReduction",
    "reduce_intrinsic_domain",
    "reduce_domain",
    "region_condition",
    "region_moment",
    "semialg_available",
    "semialgebraic_domain",
    "semialgebraic_measure",
]


def chart_geometry(chart, level_map, variables, *, assumptions=(), policy=None):
    """Consume Semialg's public, replayable chart geometry certificate."""
    from multiple_integrate.policy import _normalize_policy
    from multiple_integrate.results import UnresolvedIntegration

    try:
        from semialg.chart_certificates import (
            Chart,
            GeometryBudget,
            GeometryProofError,
            certify_chart,
        )
    except ImportError as exc:
        raise UnresolvedIntegration("Semialg chart-certification API is unavailable") from exc
    policy = _normalize_policy(policy)
    b = policy.semialg
    budget = GeometryBudget(
        b.max_variables, b.max_polynomial_degree, b.max_boolean_atoms, b.max_pieces
    )
    candidate = Chart(
        chart.coordinates,
        chart.forward,
        chart.inverse,
        chart.ranges,
        chart.chart_exclusions,
        chart.source_exclusions,
        chart.source_domain,
        chart.kind,
    )
    try:
        return certify_chart(
            candidate, level_map, variables, assumptions=assumptions, budget=budget
        )
    except GeometryProofError as exc:
        raise UnresolvedIntegration(str(exc)) from exc


def chart_cover(region, certificates, *, assumptions=(), policy=None):
    """Request source coverage and disjoint interiors from Semialg."""
    from semialg.chart_certificates import GeometryBudget, GeometryProofError, certify_cover

    from multiple_integrate.policy import _normalize_policy
    from multiple_integrate.results import UnresolvedIntegration

    b = _normalize_policy(policy).semialg
    try:
        return certify_cover(
            region,
            certificates,
            assumptions=assumptions,
            budget=GeometryBudget(
                b.max_variables, b.max_polynomial_degree, b.max_boolean_atoms, b.max_pieces
            ),
        )
    except GeometryProofError as exc:
        raise UnresolvedIntegration(str(exc)) from exc


def linear_metric_geometry(matrix, *, assumptions=(), policy=None):
    """Shared full-rank metric certificate for affine boundary/intrinsic charts."""
    try:
        from semialg.chart_certificates import (
            GeometryBudget,
            GeometryProofError,
            certify_linear_metric,
        )
    except ImportError:
        return None
    from multiple_integrate.policy import _normalize_policy
    from multiple_integrate.results import UnresolvedIntegration

    b = _normalize_policy(policy).semialg
    try:
        return certify_linear_metric(
            matrix,
            assumptions=assumptions,
            budget=GeometryBudget(
                b.max_variables, b.max_polynomial_degree, b.max_boolean_atoms, b.max_pieces
            ),
        )
    except GeometryProofError as exc:
        raise UnresolvedIntegration(str(exc)) from exc


def null_geometry(region, variables, *, assumptions=(), policy=None):
    """Certify an ambient null set with the caller's geometry budget."""
    from multiple_integrate.policy import _normalize_policy
    from multiple_integrate.results import UnresolvedIntegration

    try:
        from semialg.chart_certificates import GeometryBudget, GeometryProofError, certify_null_set
    except ImportError as exc:
        raise UnresolvedIntegration("Semialg null-set certification API is unavailable") from exc
    b = _normalize_policy(policy).semialg
    try:
        return certify_null_set(
            region,
            variables,
            assumptions=assumptions,
            budget=GeometryBudget(
                b.max_variables, b.max_polynomial_degree, b.max_boolean_atoms, b.max_pieces
            ),
        )
    except GeometryProofError as exc:
        raise UnresolvedIntegration(str(exc)) from exc


def geometry_chart_type():
    """Return the public Semialg chart candidate class when availble."""
    try:
        from semialg.chart_certificates import Chart
    except ImportError:
        return None
    return Chart


def clear_geometry_cache():
    """Clear the shared Semialg geometry caches when the service is available."""
    try:
        from semialg.chart_certificates import clear_chart_cache
    except ImportError:
        return
    clear_chart_cache()


def chart_parameter_geometry_events(chart, level_map, variables, parameters, *, policy=None):
    """Parameter-only geometry event candidates from Semialg's specialized API."""
    from semialg.chart_certificates import (
        Chart,
        GeometryBudget,
        GeometryProofError,
        chart_parameter_events,
    )

    from ..policy import _normalize_policy
    from ..results import UnresolvedIntegration

    b = _normalize_policy(policy).semialg
    candidate = Chart(
        chart.coordinates,
        chart.forward,
        chart.inverse,
        chart.ranges,
        chart.chart_exclusions,
        chart.source_exclusions,
        chart.source_domain,
        chart.kind,
    )
    try:
        return chart_parameter_events(
            candidate,
            level_map,
            variables,
            parameters,
            budget=GeometryBudget(
                b.max_variables, b.max_polynomial_degree, b.max_boolean_atoms, b.max_pieces
            ),
        )
    except GeometryProofError as exc:
        raise UnresolvedIntegration(str(exc)) from exc


def parameter_geometry_event_candidates(expressions, parameters, *, policy=None):
    """Bounded parameter endpoint and coefficient events owned by Semialg."""
    try:
        from semialg.chart_certificates import (
            GeometryBudget,
            GeometryProofError,
            parameter_geometry_events,
        )
    except ImportError as exc:
        from ..results import UnresolvedIntegration

        raise UnresolvedIntegration("Semialg parameter-geometry API is unavailable") from exc
    from ..policy import _normalize_policy
    from ..results import UnresolvedIntegration

    b = _normalize_policy(policy).semialg
    try:
        return parameter_geometry_events(
            expressions,
            parameters,
            budget=GeometryBudget(
                b.max_variables, b.max_polynomial_degree, b.max_boolean_atoms, b.max_pieces
            ),
        )
    except GeometryProofError as exc:
        raise UnresolvedIntegration(str(exc)) from exc
