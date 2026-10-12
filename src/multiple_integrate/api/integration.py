"""Canonical public integration API."""

from __future__ import annotations

import sympy as sp

from multiple_integrate.assumptions import _ask, _condition_to_predicate, _normalize_assumptions
from multiple_integrate.conditional import complete_cases, conditional_status, parameter_strata
from multiple_integrate.convergence import conditional_cases
from multiple_integrate.convergence_evidence import convergence_evidence
from multiple_integrate.core import integrate
from multiple_integrate.planner import (
    STRATEGY_OWNERS,
    convergence_trace,
    coverage_trace,
    fresh_strategy_trace,
    strategy_trace,
)
from multiple_integrate.policy import StrategyBudgetExceeded, _normalize_policy, _with_policy
from multiple_integrate.regularization import (
    RegularizationProvenance,
    RegularizationResult,
    _regularization_trace,
    _regularization_trace_context,
)
from multiple_integrate.results import (
    CoverageEvidence,
    DivergentIntegration,
    IntegrationOutcome,
    IntegrationResult,
    IntegrationStatus,
    UnevaluatedRegionIntegral,
    UnresolvedIntegration,
)
from multiple_integrate.singularities import stratify_singular_locus


def _unevaluated_request(f, ranges, region, variables, measure_dimension, bounds=None):
    from multiple_integrate.region_types import BoxRegion, Region

    if region is None and len(ranges) == 1 and isinstance(ranges[0], Region):
        region, ranges = ranges[0], ()
    if isinstance(region, BoxRegion):
        domain = sp.And(
            *(
                sp.Or(
                    sp.And(sp.Ge(v, lo, evaluate=False), sp.Le(v, hi, evaluate=False)),
                    sp.And(sp.Ge(v, hi, evaluate=False), sp.Le(v, lo, evaluate=False)),
                )
                for v, lo, hi in region.ranges
            )
        )
        return UnevaluatedRegionIntegral(
            f, domain, region.variables, measure_dimension, region=region
        )
    if region is not None:
        from multiple_integrate.adapters.semialg import semialgebraic_domain
        from multiple_integrate.region_types import Region

        if isinstance(region, Region) and not region.variables:
            return sp.Integral(sp.sympify(f), *region.ranges)
        if isinstance(region, Region):
            from multiple_integrate.semialg_geometry import domain_geometry

            if domain_geometry(region) is None:
                return sp.Integral(sp.sympify(f), *region.ranges)
        coords = tuple(variables or getattr(region, "variables", ()))
        domain, coords = semialgebraic_domain(region, coords or None)
        if bounds is not None:
            from collections.abc import Mapping

            entries = (
                ((v, *limits) for v, limits in bounds.items())
                if isinstance(bounds, Mapping)
                else bounds
            )
            names = {str(v): v for v in coords}
            atoms = []
            for v, lower, upper in entries:
                v = names.get(v, sp.sympify(v)) if isinstance(v, str) else sp.sympify(v)
                atoms.extend(
                    (
                        sp.Ge(v, sp.sympify(lower), evaluate=False),
                        sp.Le(v, sp.sympify(upper), evaluate=False),
                    )
                )
            domain = sp.And(domain, *atoms)
        return UnevaluatedRegionIntegral(f, domain, coords, measure_dimension, region=region)
    if ranges:
        return sp.Integral(sp.sympify(f), *ranges)
    if variables:
        return sp.Integral(sp.sympify(f), *tuple(variables))
    return sp.sympify(f)


@_with_policy
def _evaluate_result(
    f: sp.Expr,
    *ranges,
    assumptions=None,
    principal_value=False,
    regularization=None,
    region=None,
    variables=None,
    bounds=None,
    measure_dimension: int | str = "ambient",
    policy=None,
    _include_singular_strata=True,
) -> IntegrationResult:
    """Evaluate an integral and return method, conditions, cases, and status.

    The flat ``value`` is exactly the result of :func:`integrate`.
    Parameter-dependent improper integrals are additionally exposed as explicit
    convergence/divergence strata when the theorem family is known.
    """
    context = _normalize_assumptions(assumptions)
    # Match the flat API's positional-region normalization before assembling
    # budget placeholders or identifying integration coordinates.
    if region is None and len(ranges) == 1:
        candidate = ranges[0]
        if (hasattr(candidate, "formula") and hasattr(candidate, "variables")) or (
            variables is not None
            and not (isinstance(candidate, (tuple, list, sp.Tuple)) and len(candidate) == 3)
        ):
            region, ranges = candidate, ()
    budget_error = None
    unsupported_error = None
    with fresh_strategy_trace():
        try:
            value = integrate(
                f,
                *ranges,
                assumptions=assumptions,
                principal_value=principal_value,
                regularization=regularization,
                region=region,
                variables=variables,
                bounds=bounds,
                measure_dimension=measure_dimension,
                policy=policy,
            )
        except DivergentIntegration as exc:
            value = exc.value
        except StrategyBudgetExceeded as exc:
            budget_error = exc
            value = _unevaluated_request(f, ranges, region, variables, measure_dimension, bounds)
        except NotImplementedError as exc:
            unsupported_error = exc
            value = _unevaluated_request(f, ranges, region, variables, measure_dimension, bounds)
        attempts = strategy_trace()
        coverage = coverage_trace()
        branch_evidence = convergence_trace()

    from collections.abc import Mapping

    from multiple_integrate.cell_proofs import finite_external_parameters, finite_parameter_domain

    native = (
        region
        if region is not None
        else ranges[0]
        if len(ranges) == 1 and hasattr(ranges[0], "ranges")
        else None
    )
    if region is None:
        from multiple_integrate.core import _normalize_evaluation_domain

        try:
            _, input_ranges, _ = _normalize_evaluation_domain(ranges, assumptions)
        except UnresolvedIntegration:
            if unsupported_error is None:
                raise
            input_ranges = getattr(native, "ranges", ())
    else:
        input_ranges = getattr(native, "ranges", ())
    coordinates = tuple(
        variables or getattr(native, "variables", ()) or tuple(r[0] for r in input_ranges)
    )
    guarded_ranges = input_ranges
    parameter_inputs = [sp.sympify(f)]
    parameter_inputs.extend(b for r in guarded_ranges for b in r[1:])
    if bounds is not None:
        bound_values = bounds.values() if isinstance(bounds, Mapping) else (r[1:] for r in bounds)
        parameter_inputs.extend(b for limits in bound_values for b in limits)

    def parameter_domain_established(answer, scope):
        return finite_parameter_domain(answer, scope) and all(
            finite_external_parameters(expr, coordinates, scope) for expr in parameter_inputs
        )

    parameter_divergence_missing = False
    divergence_amplitude_missing = False
    status, cases, convergent, complete = conditional_status(value)
    if (
        status is IntegrationStatus.UNKNOWN
        and value.has(sp.Sum)
        and not value.has(sp.Integral, sp.Limit, UnevaluatedRegionIntegral, sp.nan, sp.zoo)
        and any(
            p.scope == "absolute_convergence"
            and p.complete is True
            and p.method in ("weighted_radial", "entire_box_series")
            for p in coverage
        )
    ):
        # This native family supplies a termwise finite-radius convergence proof.
        status, convergent, complete = IntegrationStatus.EXACT, True, True
    if status is IntegrationStatus.EXACT and not parameter_domain_established(value, context):
        status, convergent, complete = IntegrationStatus.UNKNOWN, None, False
    if value is sp.nan and any(
        attempt.method == "divergent_interval_piece" for attempt in attempts
    ):
        status, convergent, complete = IntegrationStatus.DIVERGENT, False, True
    if budget_error is not None:
        status = IntegrationStatus.UNEVALUATED
        convergent = None
        complete = False
    if unsupported_error is not None:
        status = (
            IntegrationStatus.UNKNOWN
            if isinstance(unsupported_error, UnresolvedIntegration)
            else IntegrationStatus.UNSUPPORTED
        )
        convergent = None
        complete = False
    explicit_ranges = bool(ranges) and all(isinstance(r, (tuple, list, sp.Tuple)) for r in ranges)
    known_cases = (
        conditional_cases(sp.sympify(f), ranges)
        if budget_error is None
        and unsupported_error is None
        and region is None
        and explicit_ranges
        and principal_value in (False, None)
        and regularization is None
        else ()
    )
    if known_cases and context.predicates:
        from dataclasses import replace

        from multiple_integrate.cell_proofs import native_predicate_true

        scoped = []
        for case in known_cases:
            predicate = _condition_to_predicate(case.condition)
            truth = _ask(predicate, context)
            if native_predicate_true(predicate, context):
                truth = True
            elif native_predicate_true(sp.Not(predicate), context):
                truth = False
            if truth is not False:
                scoped.append(replace(case, condition=sp.true if truth is True else case.condition))
        known_cases = tuple(scoped)
    if known_cases:
        cases = complete_cases(known_cases, sp.Integral(sp.sympify(f), *ranges))
        status = (
            cases[0].status
            if len(cases) == 1 and cases[0].condition is sp.true
            else IntegrationStatus.CONDITIONAL
        )
        convergent = cases[0].convergent if status is not IntegrationStatus.CONDITIONAL else None
        complete = all(
            case.status not in (IntegrationStatus.UNKNOWN, IntegrationStatus.UNEVALUATED)
            for case in cases
        )

    if cases:
        from dataclasses import replace

        guarded = []
        downgraded = False
        for case in cases:
            if case.status in (IntegrationStatus.EXACT, IntegrationStatus.DIVERGENT):
                try:
                    branch_context = _normalize_assumptions(
                        sp.And(*context.original, case.condition)
                    )
                except ValueError:
                    # An unreachable branch is vacuous under the request assumptions.
                    guarded.append(case)
                    continue
                domain_valid = (
                    parameter_domain_established(case.value, branch_context)
                    if case.status is IntegrationStatus.EXACT
                    else all(
                        finite_external_parameters(expr, coordinates, branch_context)
                        for expr in parameter_inputs
                    )
                )
                if not domain_valid:
                    parameter_divergence_missing |= case.status is IntegrationStatus.DIVERGENT
                    case = replace(case, status=IntegrationStatus.UNKNOWN, convergent=None)
                    downgraded = True
            guarded.append(case)
        if downgraded:
            cases = tuple(guarded)
            complete, convergent = False, None
            status = (
                IntegrationStatus.CONDITIONAL
                if any(c.status is IntegrationStatus.EXACT for c in cases)
                else IntegrationStatus.UNKNOWN
            )
            if known_cases:
                known_cases = cases

    if known_cases and (
        status not in (IntegrationStatus.EXACT, IntegrationStatus.DIVERGENT)
        or (status is IntegrationStatus.EXACT and cases[0].value == 0)
    ):
        # Preserve the evaluator's refined scalar form on a proved single
        # nonzero case; reconstruct parameter branches and zero fibers.
        value = sp.Piecewise(*((case.value, case.condition) for case in cases), evaluate=False)

    if status is IntegrationStatus.DIVERGENT and not all(
        finite_external_parameters(expr, coordinates, context) for expr in parameter_inputs
    ):
        parameter_divergence_missing = True
        status, convergent, complete = IntegrationStatus.UNKNOWN, None, False
        value = _unevaluated_request(f, ranges, region, variables, measure_dimension, bounds)

    if status is IntegrationStatus.DIVERGENT:
        from multiple_integrate.cell_proofs import nonzero

        coefficient, _ = sp.factor_terms(sp.sympify(f)).as_independent(*coordinates, as_Add=False)
        if not nonzero(coefficient, context):
            divergence_amplitude_missing = True
            status, convergent, complete = IntegrationStatus.UNKNOWN, None, False
            value = _unevaluated_request(f, ranges, region, variables, measure_dimension, bounds)

    if principal_value and status is IntegrationStatus.EXACT:
        status = IntegrationStatus.PRINCIPAL_VALUE
    if regularization is not None and status is IntegrationStatus.EXACT:
        status = IntegrationStatus.FINITE_PART

    if cases:
        coverage += (
            CoverageEvidence(
                "parameter_space",
                True,
                "explicit_cases",
                (("cases", len(cases)), ("basis", "disjoint cases retain unresolved complement")),
            ),
        )
    method = (
        attempts[-1].method
        if attempts
        else ("unsupported" if unsupported_error is not None else "iterated_fallback")
    )
    conditions = tuple(case.condition for case in cases if case.status is IntegrationStatus.EXACT)
    exact = status in (
        IntegrationStatus.EXACT,
        IntegrationStatus.PRINCIPAL_VALUE,
        IntegrationStatus.FINITE_PART,
    )
    limitations = ()
    if budget_error is not None:
        limitations = (str(budget_error),)
    elif unsupported_error is not None:
        limitations = (str(unsupported_error),)
    elif status is IntegrationStatus.UNKNOWN:
        limitations = (
            "symbolic evaluation or convergence could not be resolved under the supplied assumptions",
        )
    if parameter_divergence_missing:
        limitations += ("divergence requires a defined external parameter domain",)
    if divergence_amplitude_missing:
        limitations += ("divergence requires a proved nonzero coordinate-independent amplitude",)
    provenance = (
        f"strategy:{method}",
        f"owner:{STRATEGY_OWNERS.get(method, 'integrate')}",
    )
    if variables is not None:
        integration_variables = tuple(variables)
    elif region is not None and hasattr(region, "variables"):
        integration_variables = tuple(region.variables)
    else:
        integration_variables = tuple(
            entry[0]
            for entry in ranges
            if isinstance(entry, (tuple, list, sp.Tuple)) and len(entry) == 3
        )
    strata = parameter_strata(cases, integration_variables=integration_variables)
    diagnostic_details = None
    if integration_variables and _include_singular_strata:
        diagnostic_domain = sp.true
        try:
            request = _unevaluated_request(f, ranges, region, variables, measure_dimension, bounds)
            if isinstance(request, UnevaluatedRegionIntegral):
                diagnostic_domain = request.domain
            elif explicit_ranges:
                diagnostic_domain = sp.And(
                    *(
                        atom
                        for v, lo, hi in ranges
                        for atom in (sp.Ge(v, sp.Min(lo, hi)), sp.Le(v, sp.Max(lo, hi)))
                    )
                )
        except (TypeError, ValueError, NotImplementedError):
            pass
        diagnostic_details = stratify_singular_locus(
            sp.sympify(f),
            integration_variables,
            domain=diagnostic_domain,
            max_queries=_normalize_policy(policy).max_singular_queries,
            return_details=True,
        )
    singular_strata = () if diagnostic_details is None else diagnostic_details.strata
    if diagnostic_details is not None and not diagnostic_details.complete:
        limitations += ("singular-locus diagnostics are partial: auxiliary query budget exhausted",)
    measure_matches = (
        measure_dimension in (None, "ambient")
        or (type(measure_dimension) is int and measure_dimension == len(integration_variables))
        or (
            measure_dimension in ("intrinsic", "top")
            and callable(getattr(region, "dimension", None))
            and region.dimension() == len(integration_variables)
        )
    )
    evidence_ranges = ranges if measure_matches else ()
    if measure_matches and not ranges and bounds is None:
        from multiple_integrate.region_types import BoxRegion

        if isinstance(region, BoxRegion):
            evidence_ranges = region.ranges
    evidence = convergence_evidence(
        f,
        evidence_ranges,
        status=status,
        method=method,
        prescribed=bool(principal_value) or regularization is not None,
        theorem_cases=known_cases,
        assumptions=assumptions,
        policy=policy,
    )
    from multiple_integrate.convergence_evidence import region_absolute_rule
    from multiple_integrate.results import ConvergenceEvidence

    domain_proved = (
        measure_matches
        and region is not None
        and bounds is None
        and region_absolute_rule(
            f, region, assumptions=assumptions, policy=policy, variables=integration_variables
        )
        is True
    )
    piece_proof = next(
        (c for c in coverage if c.scope == "absolute_convergence" and c.complete is True), None
    )
    null_proof = None
    if measure_matches and region is not None and parameter_domain_established(value, context):
        for claim in coverage:
            if claim.scope != "ambient_null_set" or claim.complete is not True:
                continue
            details = dict(claim.details)
            if details.get("source_density") != sp.sympify(f):
                continue
            matches = details.get("source_region") == region
            if not matches and "source_formula" in details:
                from multiple_integrate.adapters.semialg import semialgebraic_domain

                source_formula, source_variables = semialgebraic_domain(
                    region, integration_variables
                )
                matches = details["source_formula"] == source_formula and tuple(
                    details["source_variables"]
                ) == tuple(source_variables)
            if matches:
                null_proof = claim
                break
    if piece_proof is None and null_proof is not None:
        piece_proof = null_proof
    if domain_proved or piece_proof is not None:
        proof_method = piece_proof.method if piece_proof is not None else "native_domain_comparison"
        evidence = tuple(e for e in evidence if e.kind not in ("ordinary", "absolute")) + (
            ConvergenceEvidence("ordinary", True, proof_method),
            ConvergenceEvidence("absolute", True, proof_method),
        )
    if branch_evidence:
        evidence = (
            tuple(
                e
                for e in evidence
                if not (e.kind in ("ordinary", "absolute") and e.established is None)
            )
            + branch_evidence
        )
    if (
        not cases
        and not principal_value
        and regularization is None
        and any(
            e.kind == "ordinary" and e.established is False and e.condition is sp.true
            for e in evidence
        )
    ):
        status, convergent, complete, exact = IntegrationStatus.DIVERGENT, False, True, False
    return IntegrationResult(
        value=sp.sympify(value),
        status=status,
        method=method,
        conditions=conditions,
        cases=cases,
        parameter_strata=strata,
        singular_strata=singular_strata,
        convergent=convergent,
        exact=exact,
        complete=complete,
        assumptions=context.predicate_expr,
        attempts=attempts,
        provenance=provenance,
        limitations=limitations,
        coverage=coverage,
        convergence_evidence=evidence,
        _convergence_policy=_normalize_policy(policy),
        diagnostics={
            "singular_diagnostics_complete": None
            if diagnostic_details is None
            else diagnostic_details.complete,
            "singular_denominator_geometry_certified": None
            if diagnostic_details is None
            else diagnostic_details.geometry_certified,
            "singular_diagnostic_queries": 0
            if diagnostic_details is None
            else diagnostic_details.queries,
            "strategy_owner": STRATEGY_OWNERS.get(method, "integrate"),
            "measure_dimension": measure_dimension,
            "principal_value": type(principal_value).__name__
            if principal_value not in (False, None)
            else None,
            "regularization": type(regularization).__name__ if regularization is not None else None,
        },
    )


def integrate_result(
    f: sp.Expr,
    *ranges,
    assumptions=None,
    principal_value=False,
    regularization=None,
    region=None,
    variables=None,
    bounds=None,
    measure_dimension: int | str = "ambient",
    policy=None,
) -> IntegrationResult:
    """Evaluate an integral with uniform outcomes and scoped coverage evidence."""
    return _evaluate_result(
        f,
        *ranges,
        assumptions=assumptions,
        principal_value=principal_value,
        regularization=regularization,
        region=region,
        variables=variables,
        bounds=bounds,
        measure_dimension=measure_dimension,
        policy=policy,
    )


def regularized_result(
    f: sp.Expr,
    *ranges,
    assumptions=None,
    principal_value=False,
    regularization=None,
    region=None,
    variables=None,
    bounds=None,
    measure_dimension: int | str = "ambient",
    policy=None,
) -> RegularizationResult:
    """Evaluate a PV or finite part and return its regulator-level provenance."""
    if principal_value in (False, None) and regularization is None:
        raise ValueError("a principal-value or regularization prescription is required")
    context = _normalize_assumptions(assumptions)
    with _regularization_trace_context():
        result = _evaluate_result(
            f,
            *ranges,
            assumptions=assumptions,
            principal_value=principal_value,
            regularization=regularization,
            region=region,
            variables=variables,
            bounds=bounds,
            measure_dimension=measure_dimension,
            policy=policy,
            _include_singular_strata=False,
        )
        value = result.value
        trace = _regularization_trace()
    if trace:
        provenance = trace[-1]
    else:
        prescription = regularization if regularization is not None else principal_value
        provenance = RegularizationProvenance(
            method="regularized_integration",
            prescription=type(prescription).__name__,
            constant_term=sp.sympify(value)
            if result.outcome is IntegrationOutcome.FINITE
            else None,
            shared_cutoff=True,
            details={"provenance_detail": "direct principal-value evaluation"},
        )
    prescription = regularization if regularization is not None else principal_value
    status = result.status.value
    return RegularizationResult(
        value=sp.sympify(value),
        status=status,
        method=provenance.method,
        prescription=prescription,
        provenance=provenance,
        exact=result.exact,
        complete=result.complete,
        cases=result.cases,
        parameter_strata=result.parameter_strata,
        convergent=result.convergent,
        coverage=result.coverage,
        convergence_evidence=result.convergence_evidence,
        _convergence_policy=result._convergence_policy,
        limitations=result.limitations,
        assumptions=context.predicate_expr,
        diagnostics={"measure_dimension": measure_dimension},
    )
