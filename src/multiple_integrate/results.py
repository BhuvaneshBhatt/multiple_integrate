"""Structured result types for symbolic multiple integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Any

import sympy as sp


class IntegrationStatus(str, Enum):
    """Semantic status of an integration result."""

    EXACT = "exact"
    CONDITIONAL = "conditional"
    UNEVALUATED = "unevaluated"
    DIVERGENT = "divergent"
    PRINCIPAL_VALUE = "principal_value"
    FINITE_PART = "finite_part"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class DivergentIntegration(ValueError):
    """A prescribed cutoff has a certified nonfinite limit."""

    def __init__(self, message, value):
        super().__init__(message)
        self.value = value


class UnresolvedIntegration(NotImplementedError):
    """A supported route could not certify the requested geometry or limit."""


class IntegrationOutcome(str, Enum):
    """Uniform mathematical outcome, independent of evaluation prescription."""

    FINITE = "finite"
    DIVERGENT = "divergent"
    CONDITIONAL = "conditional"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


def outcome_for_status(status) -> IntegrationOutcome:
    status = IntegrationStatus(status)
    if status in (
        IntegrationStatus.EXACT,
        IntegrationStatus.PRINCIPAL_VALUE,
        IntegrationStatus.FINITE_PART,
    ):
        return IntegrationOutcome.FINITE
    if status is IntegrationStatus.UNEVALUATED:
        return IntegrationOutcome.UNKNOWN
    return IntegrationOutcome(status.value)


def _finite_outcome(outcome):
    return (
        True
        if outcome is IntegrationOutcome.FINITE
        else False
        if outcome is IntegrationOutcome.DIVERGENT
        else None
    )


def _unconditional_claim(evidence, kind):
    """Combine consistent proved facts; unknown claims supply no contradiction."""
    result = None
    for claim in evidence:
        if claim.kind == kind and sp.sympify(claim.condition) is sp.true:
            if claim.established is None:
                continue
            if result is not None and result != claim.established:
                return None
            result = claim.established
    return result


def _scoped_claim(evidence, kind, assumptions, policy=None):
    """Summarize only proved, exhaustive, consistent parameter conditions."""
    known = [e for e in evidence if e.kind == kind and e.established is not None]
    if not known:
        return None
    if len({e.established for e in known}) == 1 and any(
        sp.sympify(e.condition) is sp.true for e in known
    ):
        return known[0].established
    if all(sp.sympify(e.condition) is sp.true for e in known):
        return _unconditional_claim(known, kind)
    from .adapters.semialg import is_satisfiable
    from .policy import _normalize_policy, check_semialg_budget

    predicates = {
        sp.Q.positive: lambda a: a > 0,
        sp.Q.nonnegative: lambda a: a >= 0,
        sp.Q.negative: lambda a: a < 0,
        sp.Q.nonpositive: lambda a: a <= 0,
        sp.Q.zero: lambda a: sp.Eq(a, 0),
        sp.Q.nonzero: lambda a: sp.Ne(a, 0),
    }
    parent = sp.sympify(assumptions)
    replacements = {}
    for atom in parent.atoms(sp.AppliedPredicate):
        if atom.function not in predicates or len(atom.arguments) != 1:
            return None
        replacements[atom] = predicates[atom.function](atom.arguments[0])
    parent = parent.xreplace(replacements)
    conditions = [sp.sympify(e.condition) for e in known]
    variables = tuple(
        sorted(
            set().union(parent.free_symbols, *(c.free_symbols for c in conditions)),
            key=sp.default_sort_key,
        )
    )
    if any(not isinstance(v, sp.Symbol) or v.is_real is not True for v in variables):
        return None
    policy = _normalize_policy(policy)

    def empty(condition):
        if condition is sp.false:
            return True
        allowed, _ = check_semialg_budget(condition, variables, policy)
        return allowed and is_satisfiable(condition, variables) is False

    active = [e for e in known if not empty(sp.And(parent, e.condition))]
    if not active or len({e.established for e in active}) != 1:
        return None
    covered = sp.Or(*(e.condition for e in active))
    return active[0].established if empty(sp.And(parent, sp.Not(covered))) else None


@dataclass(frozen=True)
class ConvergenceEvidence:
    """A scoped convergence claim, distinct from symbolic evaluation."""

    kind: str
    established: bool | None
    method: str
    condition: sp.Expr = sp.true
    details: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class CoverageEvidence:
    """Scoped evidence of coverage; not a convergence or finiteness proof."""

    scope: str
    complete: bool | None
    method: str
    details: tuple[tuple[str, object], ...] = ()


class UnevaluatedRegionIntegral(sp.Expr):
    """Symbolic placeholder for an unevaluated geometric-region integral.

    The object retains a semialgebraic domain formula, bound variables, and
    measure dimension, so a strategy-budget refusal does not erase the domain
    semantics.
    """

    def __new__(
        cls,
        integrand: object,
        domain: object,
        variables: tuple[sp.Symbol, ...] = (),
        measure_dimension: object = "ambient",
        region_token=None,
        *,
        region: object = None,
    ):
        from sympy.core.symbol import Str

        integrand_expr = sp.sympify(integrand)
        domain_expr = sp.sympify(domain)
        variables_expr = sp.Tuple(*map(sp.sympify, variables))
        measure_expr = Str(str(measure_dimension))
        region_repr = (
            Str(repr(region))
            if region is not None
            else region_token
            if isinstance(region_token, Str)
            else Str("")
        )
        obj = sp.Expr.__new__(
            cls, integrand_expr, domain_expr, variables_expr, measure_expr, region_repr
        )
        obj._region = region
        return obj

    @property
    def integrand(self) -> sp.Expr:
        return self.args[0]

    @property
    def domain(self) -> sp.Expr:
        return self.args[1]

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return tuple(self.args[2])

    @property
    def measure_dimension(self) -> str:
        return str(self.args[3])

    @property
    def region(self) -> object:
        return self._region

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        bound = set(self.variables)
        return (self.integrand.free_symbols | self.domain.free_symbols) - bound

    def _eval_subs(self, old, new):
        if old in self.variables or (old.is_Symbol and old not in self.free_symbols):
            return self
        integrand, domain = self.integrand, self.domain
        variables = self.variables
        # Alpha-rename bound coordinates before introducing a colliding parameter.
        # SymPy implements simultaneous subs using temporary Dummies and a
        # final xreplace. Protect all bound coordinates before that final pass.
        collision = set(variables) if new.has(sp.Dummy) else set(variables) & new.free_symbols
        if collision:
            mapping = {v: sp.Dummy(v.name, **v.assumptions0) for v in collision}
            integrand = integrand.xreplace(mapping)
            domain = domain.xreplace(mapping)
            variables = tuple(mapping.get(v, v) for v in variables)
        updated_integrand = integrand.subs(old, new)
        updated_domain = domain.subs(old, new)
        if (updated_integrand, updated_domain, variables) == (
            self.integrand,
            self.domain,
            self.variables,
        ):
            return self
        return self.func(updated_integrand, updated_domain, variables, self.args[3])

    def _sympystr(self, printer) -> str:
        variables = ", ".join(printer.doprint(var) for var in self.variables)
        return (
            f"RegionIntegral({printer.doprint(self.integrand)}, "
            f"domain={printer.doprint(self.domain)}, variables=({variables}), "
            f"measure_dimension={self.measure_dimension!r})"
        )


@dataclass(frozen=True)
class IntegralCase:
    """One parameter-conditioned branch of an integration theorem."""

    value: sp.Expr
    condition: sp.Expr
    status: IntegrationStatus
    convergent: bool | None

    @property
    def outcome(self) -> IntegrationOutcome:
        return outcome_for_status(self.status)


@dataclass(frozen=True)
class ParameterStratum:
    """One mutually exclusive parameter cell with its symbolic outcome."""

    condition: sp.Expr
    value: sp.Expr
    status: IntegrationStatus
    convergent: bool | None
    parameters: tuple[sp.Symbol, ...] = ()


@dataclass(frozen=True)
class StrategyAttempt:
    """A lightweight record of a planner/strategy decision."""

    method: str
    succeeded: bool = True
    reason: str | None = None


@dataclass(frozen=True)
class IntegrationResult:
    """Structured form of a multiple-integration result.

    ``value`` matches :func:`integrate` when flat evaluation succeeds. If a
    strategy budget declines the request, it retains a faithful unevaluated
    representation while the flat API raises ``StrategyBudgetExceeded``.
    """

    value: sp.Expr
    status: IntegrationStatus
    method: str
    conditions: tuple[sp.Expr, ...] = ()
    cases: tuple[IntegralCase, ...] = ()
    parameter_strata: tuple[ParameterStratum, ...] = ()
    singular_strata: tuple[Any, ...] = ()
    convergent: bool | None = None
    exact: bool = False
    complete: bool | None = None
    assumptions: sp.Expr = sp.true
    attempts: tuple[StrategyAttempt, ...] = ()
    provenance: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    coverage: tuple[CoverageEvidence, ...] = ()
    convergence_evidence: tuple[ConvergenceEvidence, ...] = ()
    _convergence_policy: object = field(default=None, repr=False, compare=False)

    @property
    def outcome(self) -> IntegrationOutcome:
        return outcome_for_status(self.status)

    @property
    def finite(self) -> bool | None:
        return _finite_outcome(self.outcome)

    @property
    def conditional(self) -> bool:
        return self.status is IntegrationStatus.CONDITIONAL

    @cached_property
    def absolute_convergent(self) -> bool | None:
        return _scoped_claim(
            self.convergence_evidence, "absolute", self.assumptions, self._convergence_policy
        )

    @cached_property
    def ordinary_convergent(self) -> bool | None:
        return _scoped_claim(
            self.convergence_evidence, "ordinary", self.assumptions, self._convergence_policy
        )

    def explain(self) -> str:
        """Return a compact human-readable explanation of the result."""
        pieces = [f"status={self.status.value}", f"method={self.method}"]
        if self.conditions:
            pieces.append("conditions=" + ", ".join(map(sp.sstr, self.conditions)))
        if self.limitations:
            pieces.append("limitations=" + "; ".join(self.limitations))
        return "; ".join(pieces)
