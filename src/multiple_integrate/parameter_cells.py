"""Integration over caller-supplied, certified disjoint parameter cells."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import sympy as sp
from sympy.logic.boolalg import Boolean

from .adapters.semialg import is_satisfiable, semialgebraic_domain
from .assumptions import _ask, _condition_to_predicate, _normalize_assumptions
from .policy import StrategyBudgetExceeded, _normalize_policy, _with_policy, check_semialg_budget
from .results import (
    CoverageEvidence,
    IntegralCase,
    IntegrationResult,
    IntegrationStatus,
    ParameterStratum,
)


@dataclass(frozen=True)
class ParameterCell:
    """A condition and optional certified substitutions on an exceptional fiber.

    Substitutions are simultaneous and must follow from the condition; generic
    cells must retain their parameters rather than choosing sample points.
    """

    condition: sp.Expr
    substitutions: tuple[tuple[sp.Symbol, sp.Expr], ...] = ()

    def __post_init__(self):
        condition = sp.sympify(self.condition)
        if not isinstance(condition, Boolean):
            raise TypeError("parameter-cell condition must be Boolean")
        entries = (
            self.substitutions.items()
            if isinstance(self.substitutions, Mapping)
            else self.substitutions
        )
        entries = tuple((sp.sympify(k), sp.sympify(v)) for k, v in entries)
        if any(not isinstance(k, sp.Symbol) for k, _ in entries) or len(
            {k for k, _ in entries}
        ) != len(entries):
            raise ValueError("cell substitutions require distinct Symbol keys")
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "substitutions", entries)


def _equality_substitutions(condition, parameters, budget):
    """Exact RREF specialization of conjunctive parameter-only linear equalities."""
    atoms = sp.And.make_args(condition)
    equalities = [a.lhs - a.rhs for a in atoms if isinstance(a, sp.Equality)]
    if not equalities:
        return {}
    if not budget.allow_expensive_exact and len(parameters) > budget.semialg.max_variables:
        return {}
    try:
        linear = [e for e in equalities if sp.Poly(e, *parameters).total_degree() <= 1]
        if not linear:
            return {}
        solutions = sp.linsolve(linear, parameters)
        if len(solutions) != 1:
            return {}
        values = next(iter(solutions))
        return {
            p: v
            for p, v in zip(parameters, values, strict=True)
            if p != v and v.free_symbols <= set(parameters)
        }
    except (sp.PolynomialError, ValueError, TypeError):
        return {}


def _validate_parameter_cells(cells, parameters, parent, budget, *, automatic=False):
    """Shared exact partition and equality-fiber validation for integration routes."""
    if not budget.allow_expensive_exact and len(cells) > budget.semialg.max_pieces:
        raise StrategyBudgetExceeded("parameter cell count exceeds semialg max_pieces")
    if any(not isinstance(c, ParameterCell) for c in cells):
        raise TypeError("cells must contain ParameterCell objects")

    def empty(formula):
        allowed, reason = check_semialg_budget(formula, parameters, budget)
        if not allowed:
            raise StrategyBudgetExceeded(reason)
        if formula is sp.false:
            return True
        if formula is sp.true:
            return False
        return is_satisfiable(formula, parameters) is False

    # Validate the entire partition before evaluating any child.
    active = []
    for cell in cells:
        if not cell.condition.free_symbols <= set(parameters):
            raise ValueError("cell conditions must involve only declared parameters")
        condition = sp.And(parent, cell.condition)
        if empty(condition):
            continue
        # Automatic root intervals and sign-tree leaves are disjoint by
        # construction. Supplied partitions still need independent validation.
        if not automatic:
            for old in active:
                if not empty(sp.And(condition, old.condition)):
                    raise ValueError(
                        "parameter cells overlap or disjointness could not be certified"
                    )
        inferred = _equality_substitutions(condition, parameters, budget)
        combined = {
            **inferred,
            **{k: v.subs(inferred, simultaneous=True) for k, v in cell.substitutions},
        }
        for key, value in combined.items():
            if key not in parameters or not value.free_symbols <= set(parameters):
                raise ValueError("substitutions must involve only declared parameters")
            if not empty(sp.And(condition, sp.Ne(key, value))):
                raise ValueError("parameter-cell substitution does not follow from its condition")
        active.append(replace(cell, substitutions=tuple(combined.items())))

    return tuple(active)


@_with_policy
def integrate_cells(
    f,
    *ranges,
    cells=None,
    parameters,
    assumptions=None,
    region=None,
    variables=None,
    bounds=None,
    measure_dimension="ambient",
    policy=None,
    principal_value=False,
    regularization=None,
):
    """Evaluate disjoint supplied cells; preserve the uncovered complement as unknown.

    Omit ``cells`` for budgeted polynomial parameter-event sign conditions;
    broader families require a supplied partition. Coverage means coverage of
    parameter space under ``assumptions``;
    child results retain their own geometric and convergence evidence.
    """
    from .api.integration import _unevaluated_request, integrate_result

    parameters = tuple(parameters)
    if len(set(parameters)) != len(parameters) or any(
        not isinstance(p, sp.Symbol) or p.is_real is not True for p in parameters
    ):
        raise ValueError("parameters must be distinct real Symbols")
    automatic = cells is None
    if automatic:
        from .transition_cells import linear_quadratic_cells

        cells = linear_quadratic_cells(
            *ranges,
            parameters=parameters,
            region=region,
            variables=variables,
            bounds=bounds,
            policy=policy,
        )
    cells = tuple(cells)
    if any(not isinstance(c, ParameterCell) for c in cells):
        raise TypeError("cells must contain ParameterCell objects")
    budget = _normalize_policy(policy)
    if not budget.allow_expensive_exact and len(cells) > budget.semialg.max_pieces:
        raise StrategyBudgetExceeded("parameter cell count exceeds semialg max_pieces")
    coords = tuple(variables or getattr(region, "variables", ()) or tuple(r[0] for r in ranges))
    if set(coords) & set(parameters):
        raise ValueError("parameters cannot be integration coordinates")
    parent = sp.And(*_normalize_assumptions(assumptions).original)
    if not parent.free_symbols <= set(parameters):
        raise ValueError("cell orchestration assumptions must involve only declared parameters")

    active = _validate_parameter_cells(cells, parameters, parent, budget, automatic=automatic)

    def empty(formula):
        allowed, reason = check_semialg_budget(formula, parameters, budget)
        if not allowed:
            raise StrategyBudgetExceeded(reason)
        return (
            formula is sp.false
            or formula is not sp.true
            and is_satisfiable(formula, parameters) is False
        )

    request = _unevaluated_request(f, ranges, region, variables, measure_dimension, bounds)
    geometry_expressions = [sp.sympify(b) for r in ranges for b in r[1:]]
    if region is not None:
        from sympy.core.relational import Relational

        original_formula, _ = semialgebraic_domain(region, coords)
        geometry_expressions.extend(
            side for atom in original_formula.atoms(Relational) for side in (atom.lhs, atom.rhs)
        )
    if bounds is not None:
        entries = bounds.values() if isinstance(bounds, Mapping) else (r[1:] for r in bounds)
        geometry_expressions.extend(sp.sympify(b) for limits in entries for b in limits)
    geometry_denominators = {
        sp.denom(sp.together(expr)) for expr in geometry_expressions if expr.has(*parameters)
    }
    geometry_denominators = {
        d for d in geometry_denominators if d.has(*parameters) and not d.has(*coords)
    }
    cases, evidence, children = [], [], []
    for cell in active:
        substitutions = dict(cell.substitutions)
        cell_context = _normalize_assumptions(sp.And(parent, cell.condition))
        if any(
            d.subs(substitutions, simultaneous=True) == 0
            or _ask(sp.Q.zero(d), cell_context) is True
            for d in geometry_denominators
        ):
            child = IntegrationResult(
                request,
                IntegrationStatus.UNKNOWN,
                "parameter_geometry_pole",
                complete=False,
                limitations=("geometry is undefined on a parameter-coefficient pole fiber",),
            )
            children.append(child)
            cases.append(IntegralCase(request, cell.condition, child.status, None))
            continue
        local_f = sp.sympify(f).subs(substitutions, simultaneous=True)
        if substitutions and any(
            local_f.has(bad) and not sp.sympify(f).has(bad)
            for bad in (sp.nan, sp.zoo, sp.oo, -sp.oo)
        ):
            child = IntegrationResult(
                request,
                IntegrationStatus.UNKNOWN,
                "parameter_density_pole",
                complete=False,
                limitations=(
                    "density is undefined or nonfinite on this parameter specialization; no divergence theorem follows",
                ),
            )
            children.append(child)
            cases.append(IntegralCase(request, cell.condition, child.status, None))
            continue
        local_ranges = tuple(
            (
                v,
                sp.sympify(lo).subs(substitutions, simultaneous=True),
                sp.sympify(hi).subs(substitutions, simultaneous=True),
            )
            for v, lo, hi in ranges
        )
        local_region = region
        if region is not None:
            from sympy.core.relational import Relational

            formula, _ = semialgebraic_domain(region, coords)
            formula = formula.subs(substitutions, simultaneous=True)
            context = _normalize_assumptions(sp.And(parent, cell.condition))
            replacements = {}
            for atom in formula.atoms(Relational):
                if atom.free_symbols <= set(parameters):
                    truth = _ask(_condition_to_predicate(atom), context)
                    if truth is not None:
                        replacements[atom] = sp.true if truth else sp.false
            local_region = formula.xreplace(replacements)
        local_bounds = bounds
        if bounds is not None and substitutions:
            entries = (
                ((v, *bs) for v, bs in bounds.items()) if isinstance(bounds, Mapping) else bounds
            )
            local_bounds = tuple(
                (
                    v,
                    sp.sympify(lo).subs(substitutions, simultaneous=True),
                    sp.sympify(hi).subs(substitutions, simultaneous=True),
                )
                for v, lo, hi in entries
            )
        # A fiber substitution changes the requested regulator as well as the density.
        local_pv, local_fp = principal_value, regularization
        collapsed_regulator = False
        regulator_reason = "parameter specialization makes the requested hypersurface identically zero; a proper cutoff hypersurface is required"
        if substitutions:
            for name, prescription in (("pv", local_pv), ("fp", local_fp)):
                if hasattr(prescription, "hypersurface") and prescription.hypersurface is not None:
                    specialized_hypersurface = prescription.hypersurface.subs(
                        substitutions, simultaneous=True
                    )
                    if specialized_hypersurface.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                        collapsed_regulator = True
                        regulator_reason = "parameter specialization makes the requested hypersurface undefined or nonfinite"
                        continue
                    if sp.expand(specialized_hypersurface) == 0:
                        collapsed_regulator = True
                        continue
                    updated = replace(prescription, hypersurface=specialized_hypersurface)
                    if name == "pv":
                        local_pv = updated
                    else:
                        local_fp = updated
                if hasattr(prescription, "singularities"):
                    updated = replace(
                        local_pv if name == "pv" else local_fp,
                        singularities=tuple(
                            p.subs(substitutions, simultaneous=True)
                            for p in prescription.singularities
                        ),
                    )
                    if name == "pv":
                        local_pv = updated
                    else:
                        local_fp = updated
        collapsed_regulator = collapsed_regulator or any(
            getattr(prescription, "hypersurface", None) is not None
            and sp.expand(prescription.hypersurface) == 0
            for prescription in (local_pv, local_fp)
        )
        symbolic_child = (
            IntegrationResult(
                _unevaluated_request(
                    local_f, local_ranges, local_region, coords, measure_dimension, local_bounds
                ),
                IntegrationStatus.UNSUPPORTED,
                "parameter_cells",
                complete=False,
                limitations=(regulator_reason,),
            )
            if collapsed_regulator
            else None
        )
        if symbolic_child is None and local_region is not None:
            from .symbolic_geometry import integrate_symbolic_boxes

            symbolic_child = integrate_symbolic_boxes(
                local_f,
                local_region,
                coords,
                assumptions=sp.And(parent, cell.condition),
                bounds=local_bounds,
                policy=policy,
                measure_dimension=measure_dimension,
                principal_value=local_pv,
                regularization=local_fp,
            )
        if symbolic_child is not None:
            child = symbolic_child
        elif local_region is not None and local_region.free_symbols - set(coords):
            child = IntegrationResult(
                _unevaluated_request(
                    local_f, local_ranges, local_region, coords, measure_dimension, local_bounds
                ),
                IntegrationStatus.UNKNOWN,
                "parameter_cells",
                complete=False,
                limitations=(
                    "generic parameter-dependent geometry requires a certified symbolic reduction; supply symbolic ranges or specialize an exceptional fiber",
                ),
            )
        else:
            child = integrate_result(
                local_f,
                *local_ranges,
                region=local_region,
                variables=coords if region is not None else variables,
                bounds=local_bounds,
                assumptions=sp.And(parent, cell.condition),
                policy=policy,
                measure_dimension=measure_dimension,
                principal_value=local_pv,
                regularization=local_fp,
            )
        children.append(child)
        for branch in child.cases or (
            IntegralCase(child.value, sp.true, child.status, child.convergent),
        ):
            # Child conditions describe the specialized request; pull them back to the cell.
            condition = sp.And(
                cell.condition, branch.condition.subs(substitutions, simultaneous=True)
            )
            cases.append(IntegralCase(branch.value, condition, branch.status, branch.convergent))
        evidence.extend(
            replace(
                e,
                condition=sp.And(
                    cell.condition, e.condition.subs(substitutions, simultaneous=True)
                ),
            )
            for e in child.convergence_evidence
        )
    union = sp.Or(*(c.condition for c in active))
    covered = empty(sp.And(parent, sp.Not(union)))
    if not covered:
        cases.append(IntegralCase(request, sp.Not(union), IntegrationStatus.UNKNOWN, None))
    value = sp.Piecewise(*((c.value, c.condition) for c in cases), (request, True), evaluate=False)
    return IntegrationResult(
        value,
        IntegrationStatus.CONDITIONAL,
        "parameter_cells",
        cases=tuple(cases),
        parameter_strata=tuple(
            ParameterStratum(c.condition, c.value, c.status, c.convergent, parameters)
            for c in cases
        ),
        complete=covered and all(c.complete is True for c in children),
        assumptions=parent,
        exact=covered and all(c.exact for c in children),
        coverage=(
            CoverageEvidence(
                "parameter_space",
                covered,
                "automatic_transition_cells" if automatic else "caller_supplied_cells",
                (("cells", len(active)),),
            ),
        ),
        convergence_evidence=tuple(evidence),
        _convergence_policy=budget,
        diagnostics={"cell_results": tuple(children)},
        limitations=(("uncovered parameter values retained as unknown",) if not covered else ())
        + (
            ("one or more supplied cells remains unresolved; inspect cell_results",)
            if any(child.complete is not True for child in children)
            else ()
        ),
    )


__all__ = ["ParameterCell", "integrate_cells"]
