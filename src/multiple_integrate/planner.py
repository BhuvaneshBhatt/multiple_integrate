"""Planner instrumentation and architectural ownership for integration strategies."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from functools import wraps

from .results import ConvergenceEvidence, CoverageEvidence, StrategyAttempt

# Stable owner labels keep result provenance independent of callable identity.
STRATEGY_OWNERS = {
    "coarea_affine": "multiple_integrate.coarea",
    "coarea_radial": "multiple_integrate.coarea",
    "coarea_charts": "multiple_integrate.coarea",
    "affine_exponential_boundary": "multiple_integrate.boundary_reduction",
    "polynomial_boundary": "multiple_integrate.boundary_reduction",
    "quadratic_sector": "multiple_integrate.quadratic_sectors",
    "endpoint_beta_moments": "multiple_integrate.selected_families",
    "generalized_laplace_moments": "multiple_integrate.selected_families",
    "rational_mellin_moments": "multiple_integrate.selected_families",
    "periodic_trigonometric_tail": "multiple_integrate.fast_rules",
    "entire_box_series": "multiple_integrate.entire_box_series",
    "polynomial_function_moments": "multiple_integrate.fast_rules",
    "box_polynomial_moments": "multiple_integrate.fast_rules",
    "root_free_rational": "multiple_integrate.fast_rules",
    "quartic_rational_moments": "multiple_integrate.fast_rules",
    "reciprocal_phase": "multiple_integrate.fast_rules",
    "backend_deadline": "multiple_integrate.isolated_queries",
    "clipped_quadratic_cell": "multiple_integrate.clipped_quadratic",
    "cusp_horn": "multiple_integrate.cusp_horns",
    "nonseparable_improper": "multiple_integrate.nonseparable_improper",
    "anisotropic_moments": "multiple_integrate.anisotropic",
    "weighted_quadratic": "multiple_integrate.quadratic_radial",
    "weighted_radial": "multiple_integrate.quadratic_radial",
    "regular_density_cutoff": "multiple_integrate.regular_density",
    "principal_value": "multiple_integrate.principal_value",
    "iterated_principal_value": "multiple_integrate.principal_value",
    "geometric_principal_value": "multiple_integrate.principal_value",
    "hypersurface_principal_value": "multiple_integrate.principal_value",
    "hadamard_finite_part": "multiple_integrate.hadamard",
    "conditional_convergence": "multiple_integrate.convergence",
    "piecewise": "integrate",
    "indicator": "integrate",
    "constant_volume": "multiple_integrate.regions",
    "semialg_measure": "semialg",
    "ambient_null_set": "semialg",
    "divergent_interval_piece": "multiple_integrate.singularities",
    "semialgebraic_domain_moment": "semialg",
    "semialgebraic_domain_reduction": "semialg",
    "semialg_cad_reduction": "semialg",
    "semialg_intrinsic_reduction": "semialg",
    "region_shortcut": "multiple_integrate.regions",
    "symmetry": "integrate",
    "region_transform": "integrate",
    "gaussian_linear_map": "integrate",
    "graph_reversal": "multiple_integrate.regions",
    "additive_split": "integrate",
    "step_function": "integrate",
    "product_separable": "integrate",
    "inactive_dimension_factor": "integrate",
    "superellipse": "integrate",
    "decomposition": "integrate",
    "iterated_fallback": "sympy",
    "cache": "integrate",
}

_TRACE: contextvars.ContextVar[tuple[StrategyAttempt, ...]] = contextvars.ContextVar(
    "multiple_integrate_strategy_trace", default=()
)


_COVERAGE: contextvars.ContextVar[tuple[CoverageEvidence, ...] | None] = contextvars.ContextVar(
    "multiple_integrate_coverage", default=None
)


_CONVERGENCE = contextvars.ContextVar("multiple_integrate_convergence", default=None)


def record_convergence(kind, established, method, condition=None):
    """Record a proof on its own parameter branch, separate from domain coverage."""
    import sympy as sp

    current = _CONVERGENCE.get()
    if current is not None:
        _CONVERGENCE.set(
            (
                *current,
                ConvergenceEvidence(
                    kind, established, method, sp.true if condition is None else condition
                ),
            )
        )


def convergence_trace():
    return _CONVERGENCE.get() or ()


_INTEGRATION_DEPTH = contextvars.ContextVar("multiple_integrate_depth", default=0)


@contextmanager
def without_coverage():
    """Prevent a subproblem's certificate from describing its parent domain."""
    token = _COVERAGE.set(None)
    convergence_token = _CONVERGENCE.set(None)
    try:
        yield
    finally:
        _COVERAGE.reset(token)
        _CONVERGENCE.reset(convergence_token)


def isolate_nested_coverage(function):
    """Keep recursive flat integration evidence local to its subproblem."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        depth = _INTEGRATION_DEPTH.get()
        token = _INTEGRATION_DEPTH.set(depth + 1)
        convergence_token = (
            _CONVERGENCE.set(()) if not depth and _CONVERGENCE.get() is None else None
        )
        try:
            if depth:
                with without_coverage():
                    return function(*args, **kwargs)
            return function(*args, **kwargs)
        finally:
            _INTEGRATION_DEPTH.reset(token)
            if convergence_token is not None:
                _CONVERGENCE.reset(convergence_token)

    return wrapped


def record_coverage(scope: str, complete: bool | None, method: str, **details) -> None:
    current = _COVERAGE.get()
    if current is not None:
        evidence = CoverageEvidence(scope, complete, method, tuple(details.items()))
        _COVERAGE.set(current + (evidence,))


def coverage_trace() -> tuple[CoverageEvidence, ...]:
    return _COVERAGE.get() or ()


def record_strategy(method: str, *, succeeded: bool = True, reason: str | None = None) -> None:
    current = _TRACE.get()
    _TRACE.set(current + (StrategyAttempt(method, succeeded, reason),))


def strategy_trace() -> tuple[StrategyAttempt, ...]:
    return _TRACE.get()


@contextmanager
def fresh_strategy_trace():
    token = _TRACE.set(())
    coverage_token = _COVERAGE.set(())
    convergence_token = _CONVERGENCE.set(())
    try:
        yield
    finally:
        _TRACE.reset(token)
        _COVERAGE.reset(coverage_token)
        _CONVERGENCE.reset(convergence_token)


def record_piece_convergence(pieces, method, *, assumptions=None):
    """Propagate absolute convergence across a complete disjoint domain reduction."""
    if _COVERAGE.get() is None:
        return
    import sympy as sp

    from .assumptions import _ask, _normalize_assumptions
    from .convergence_evidence import absolute_rule

    def proved(piece):
        value = getattr(piece, "direct_value", None)
        if value is not None or not piece.limits:
            return (
                _ask(
                    sp.Q.finite(piece.integrand if value is None else value),
                    _normalize_assumptions(assumptions),
                )
                is True
            )
        return absolute_rule(piece.integrand, piece.limits, assumptions=assumptions)[0] is True

    if all(proved(p) for p in pieces):
        record_coverage(
            "absolute_convergence",
            True,
            method,
            pieces=len(pieces),
            basis="all pieces of a complete domain reduction are absolutely integrable",
        )
