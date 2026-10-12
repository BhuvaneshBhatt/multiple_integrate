"""Certified local charts for regular implicit polynomial hypersurfaces."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .adapters.semialg import is_satisfiable as semialg_is_satisfiable
from .policy import _CURRENT_POLICY, StrategyBudgetExceeded, polynomial_degree
from .results import UnresolvedIntegration


@dataclass(frozen=True)
class HypersurfaceChart:
    """One disjoint transverse chart in an implicit-hypersurface atlas."""

    transverse: sp.Symbol
    level: sp.Symbol
    defining_equation: sp.Expr
    derivative: sp.Expr
    inverse: sp.Expr | None = None
    jacobian: sp.Expr | None = None
    patch_condition: sp.Expr = sp.true
    derivative_sign: int = 1
    chart_index: int = 0

    @property
    def explicit(self) -> bool:
        """Whether this chart has been realized as an explicit inverse branch."""
        return self.inverse is not None and self.jacobian is not None

    @property
    def implicit_relation(self) -> sp.Expr:
        """The chart relation retained without solving for the transverse coordinate."""
        return sp.Eq(self.defining_equation, self.level)


@dataclass(frozen=True)
class HypersurfaceAtlas:
    """A semialgebraically certified atlas covering a regular level set."""

    polynomial: sp.Expr
    variables: tuple[sp.Symbol, ...]
    charts: tuple[HypersurfaceChart, ...]
    regularity_certified: bool
    partition_rule: str = "gradient_dominance"


def _is_regular_level_set(formula: sp.Expr, g: sp.Expr, variables: tuple[sp.Symbol, ...]) -> bool:
    """Certify that ``g=0`` has no critical point in the integration domain."""
    critical = sp.And(
        formula,
        sp.Eq(g, 0),
        *(sp.Eq(sp.diff(g, variable), 0) for variable in variables),
    )
    try:
        satisfiable = semialg_is_satisfiable(critical, variables)
        return None if satisfiable is None else not satisfiable
    except (ArithmeticError, TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        return None


def _gradient_patch(derivatives: tuple[sp.Expr, ...], index: int, sign: int) -> sp.Expr:
    """Disjoint full-dimensional patch where one gradient component dominates.

    Strict inequalities against earlier coordinates provide a deterministic
    tie break.  On a regular level set, the selected derivative cannot vanish.
    """
    chosen_sq = sp.expand(derivatives[index] ** 2)
    atoms: list[sp.Expr] = []
    for other_index, derivative in enumerate(derivatives):
        if other_index == index:
            continue
        other_sq = sp.expand(derivative**2)
        relation = sp.Gt(chosen_sq, other_sq) if other_index < index else sp.Ge(chosen_sq, other_sq)
        atoms.append(relation)
    atoms.append(sp.Gt(derivatives[index], 0) if sign > 0 else sp.Lt(derivatives[index], 0))
    return sp.And(*atoms)


def _explicit_inverse(
    g: sp.Expr,
    transverse: sp.Symbol,
    level: sp.Symbol,
    derivative: sp.Expr,
    sign: int,
) -> tuple[sp.Expr, sp.Expr] | None:
    """Select the explicit inverse branch with the requested derivative sign."""
    try:
        roots = sp.solve(sp.Eq(g, level), transverse)
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        return None
    if len(roots) == 1:
        root = sp.simplify(roots[0])
        if root.has(transverse):
            return None
        derivative_on_branch = sp.simplify(derivative.subs(transverse, root))
        return root, sp.simplify(1 / sp.Abs(derivative_on_branch))
    candidates: list[tuple[sp.Expr, sp.Expr]] = []
    for root in roots:
        root = sp.simplify(root)
        if root.has(transverse):
            continue
        derivative_on_branch = sp.simplify(derivative.subs(transverse, root))
        radical_replacements = {}
        for power in derivative_on_branch.atoms(sp.Pow):
            if power.exp.is_Rational and power.exp.q % 2 == 0:
                radical_replacements[power] = sp.Dummy("positive_radical", positive=True, real=True)
        sign_probe = derivative_on_branch.xreplace(radical_replacements)
        sign_ok = sign_probe.is_positive is True if sign > 0 else sign_probe.is_negative is True
        if sign_ok:
            candidates.append((root, sp.simplify(1 / sp.Abs(derivative_on_branch))))
    if len(candidates) == 1:
        return candidates[0]
    return None


def build_hypersurface_atlas(
    polynomial: sp.Expr,
    formula: sp.Expr,
    variables: tuple[sp.Symbol, ...],
) -> HypersurfaceAtlas:
    """Build a disjoint executable atlas for a regular implicit level set.

    The atlas partitions a neighborhood by the dominant gradient component and
    derivative sign.  Every certified patch is retained as an implicit chart.
    Explicit inverse branches are optional realizations, not part of the chart
    semantics.
    """
    g = sp.expand(sp.sympify(polynomial))
    policy = _CURRENT_POLICY.get()
    budget = policy.hypersurface
    if not policy.allow_expensive_exact and len(variables) > budget.max_variables:
        raise StrategyBudgetExceeded(
            f"hypersurface variable count {len(variables)} exceeds {budget.max_variables}"
        )
    degree = polynomial_degree(g, variables)
    if (
        not policy.allow_expensive_exact
        and degree is not None
        and degree > budget.max_polynomial_degree
    ):
        raise StrategyBudgetExceeded(
            f"hypersurface polynomial degree {degree} exceeds {budget.max_polynomial_degree}"
        )
    try:
        sp.Poly(g, *variables)
    except sp.PolynomialError as exc:
        raise ValueError("hypersurface must be polynomial in the integration variables") from exc
    regular = _is_regular_level_set(sp.sympify(formula), g, variables)
    if regular is not True:
        error = NotImplementedError if regular is False else UnresolvedIntegration
        raise error(
            "semialg could not certify that the implicit hypersurface is regular on the domain"
        )

    derivatives = tuple(sp.simplify(sp.diff(g, variable)) for variable in variables)
    level = sp.Dummy("pv_level", real=True)
    charts: list[HypersurfaceChart] = []
    chart_index = 0
    for index, (variable, derivative) in enumerate(zip(variables, derivatives, strict=True)):
        if derivative == 0:
            continue
        for sign in (1, -1):
            patch = _gradient_patch(derivatives, index, sign)
            # Skip empty patches with a decision query rather than asking the
            # integration reducer to construct chart bounds prematurely.
            if semialg_is_satisfiable(sp.And(formula, patch), variables) is False:
                continue
            inverse_data = _explicit_inverse(g, variable, level, derivative, sign)
            inverse = None if inverse_data is None else inverse_data[0]
            jacobian = None if inverse_data is None else inverse_data[1]
            charts.append(
                HypersurfaceChart(
                    transverse=variable,
                    level=level,
                    defining_equation=g,
                    derivative=derivative,
                    inverse=inverse,
                    jacobian=jacobian,
                    patch_condition=patch,
                    derivative_sign=sign,
                    chart_index=chart_index,
                )
            )
            chart_index += 1
    if not charts:
        raise NotImplementedError(
            "regular implicit hypersurface has no certified transverse charts"
        )
    if not policy.allow_expensive_exact and len(charts) > budget.max_algebraic_branches:
        raise StrategyBudgetExceeded(
            f"hypersurface chart count {len(charts)} exceeds {budget.max_algebraic_branches}"
        )
    return HypersurfaceAtlas(g, variables, tuple(charts), True)


__all__ = ["HypersurfaceAtlas", "HypersurfaceChart", "build_hypersurface_atlas"]
