"""Planner budgets and deterministic symbolic-complexity guards."""

from __future__ import annotations

import contextvars
import math
from dataclasses import dataclass
from functools import wraps

import sympy as sp
from sympy.core.relational import Relational


class StrategyBudgetExceeded(NotImplementedError):
    """Raised before dispatch when an exact strategy exceeds its configured budget."""


@dataclass(frozen=True)
class StrategyBudget:
    """Upper bounds for one family of exact symbolic strategies."""

    max_variables: int = 4
    max_polynomial_degree: int = 8
    max_boolean_atoms: int = 24
    max_pieces: int = 64
    max_algebraic_branches: int = 16

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class IntegrationPolicy:
    """Explicit policy controlling potentially expensive exact strategies.

    ``allow_expensive_exact`` disables pre-dispatch complexity rejection while
    retaining the numerical limits as provenance/diagnostic information.
    """

    semialg: StrategyBudget = StrategyBudget()
    hypersurface: StrategyBudget = StrategyBudget(max_variables=5, max_polynomial_degree=10)
    asymptotic: StrategyBudget = StrategyBudget(max_variables=3, max_polynomial_degree=12)
    allow_expensive_exact: bool = False
    max_singular_queries: int = 128
    backend_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        timeout = self.backend_timeout_seconds
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("backend_timeout_seconds must be finite and positive or None")
        for name in ("semialg", "hypersurface", "asymptotic"):
            if not isinstance(getattr(self, name), StrategyBudget):
                raise TypeError(f"{name} must be a StrategyBudget")
        if not isinstance(self.allow_expensive_exact, bool):
            raise TypeError("allow_expensive_exact must be Boolean")
        if type(self.max_singular_queries) is not int or self.max_singular_queries < 0:
            raise ValueError("max_singular_queries must be a nonnegative integer")


DEFAULT_POLICY = IntegrationPolicy()
_CURRENT_POLICY: contextvars.ContextVar[IntegrationPolicy] = contextvars.ContextVar(
    "multiple_integrate_policy", default=DEFAULT_POLICY
)


def _normalize_policy(policy: IntegrationPolicy | None) -> IntegrationPolicy:
    if policy is None:
        return _CURRENT_POLICY.get()
    if not isinstance(policy, IntegrationPolicy):
        raise TypeError("policy must be an IntegrationPolicy")
    return policy


def polynomial_degree(expr: object, variables: tuple[sp.Symbol, ...]) -> int | None:
    """Return total polynomial degree, or ``None`` for a non-polynomial expression."""
    try:
        polynomial = sp.Poly(sp.sympify(expr), *variables)
        return 0 if polynomial.is_zero else int(polynomial.total_degree())
    except (sp.PolynomialError, TypeError, ValueError):
        return None


def boolean_atom_count(formula: object) -> int:
    """Count relational atoms in a Boolean formula without invoking QE."""
    expr = sp.sympify(formula)
    return len(expr.atoms(Relational))


def check_semialg_budget(
    formula: object,
    variables: tuple[sp.Symbol, ...],
    policy: IntegrationPolicy,
) -> tuple[bool, str | None]:
    """Return whether semialgebraic reduction is within the configured budget."""
    if policy.allow_expensive_exact:
        return True, None
    budget = policy.semialg
    if len(variables) > budget.max_variables:
        return False, f"semialg variable count {len(variables)} exceeds {budget.max_variables}"
    atoms = boolean_atom_count(formula)
    if atoms > budget.max_boolean_atoms:
        return False, f"semialg Boolean atom count {atoms} exceeds {budget.max_boolean_atoms}"
    polys: list[sp.Expr] = []
    for atom in sp.sympify(formula).atoms(Relational):
        polys.append(sp.expand(atom.lhs - atom.rhs))
    degrees = [
        degree for poly in polys if (degree := polynomial_degree(poly, variables)) is not None
    ]
    if degrees and max(degrees) > budget.max_polynomial_degree:
        return (
            False,
            f"semialg polynomial degree {max(degrees)} exceeds {budget.max_polynomial_degree}",
        )
    return True, None


__all__ = [
    "DEFAULT_POLICY",
    "IntegrationPolicy",
    "StrategyBudget",
    "StrategyBudgetExceeded",
    "boolean_atom_count",
    "check_semialg_budget",
    "polynomial_degree",
]


def _reject_semialg_budget(reason: str) -> None:
    from .planner import record_strategy

    record_strategy("semialg_budget_guard", succeeded=False, reason=reason)
    raise StrategyBudgetExceeded(f"strategy budget rejected semialg reduction: {reason}")


def enforce_semialg_budget(formula, variables, policy: IntegrationPolicy) -> None:
    """Enforce the shared ambient/intrinsic pre-dispatch geometry budget."""
    allowed, reason = check_semialg_budget(formula, variables, policy)
    if not allowed:
        _reject_semialg_budget(reason)


def enforce_semialg_piece_budget(piece_count: int, policy: IntegrationPolicy) -> None:
    """Enforce the shared post-reduction piece budget."""
    if not policy.allow_expensive_exact and piece_count > policy.semialg.max_pieces:
        _reject_semialg_budget(
            f"semialg piece count {piece_count} exceeds {policy.semialg.max_pieces}"
        )


def _with_policy(function):
    """Retain an explicit policy through orchestration and auxiliary queries."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _CURRENT_POLICY.set(_normalize_policy(kwargs.get("policy")))
        try:
            return function(*args, **kwargs)
        finally:
            _CURRENT_POLICY.reset(token)

    return wrapped
