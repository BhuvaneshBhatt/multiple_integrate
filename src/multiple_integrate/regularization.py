"""Regularization prescriptions, provenance, and finite-part extraction."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import sympy as sp

from .adapters.asymptotic import cutoff_expansion
from .policy import _CURRENT_POLICY, StrategyBudgetExceeded, polynomial_degree
from .results import (
    ConvergenceEvidence,
    CoverageEvidence,
    IntegralCase,
    IntegrationOutcome,
    ParameterStratum,
    UnevaluatedRegionIntegral,
    UnresolvedIntegration,
    _finite_outcome,
    _scoped_claim,
    outcome_for_status,
)


@dataclass(frozen=True)
class HadamardFinitePart:
    """Hadamard finite-part prescription from a shared cutoff integral.

    ``hypersurface`` selects a polynomial singular locus ``g(x)=0`` for a
    multiple integral.  For one explicit range it may be omitted; rational
    poles and singular endpoints are then detected from the integrand.
    """

    hypersurface: sp.Expr | None = None
    singularities: tuple[sp.Expr, ...] = ()

    def __post_init__(self) -> None:
        hypersurface = None if self.hypersurface is None else sp.sympify(self.hypersurface)
        singularities = tuple(map(sp.sympify, self.singularities))
        object.__setattr__(self, "hypersurface", hypersurface)
        object.__setattr__(self, "singularities", singularities)


@dataclass(frozen=True)
class RegularizationCertificate:
    """Replayable claims supporting a regularized value."""

    claims: tuple[tuple[str, object], ...] = ()
    verifier_version: int = 1


@dataclass(frozen=True)
class RegularizationVerification:
    """Result of independently replaying a regularization certificate."""

    valid: bool
    checks: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...] = ()
    strength: str = "none"
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegularizationProvenance:
    """Auditable description of how a finite regularized value was assigned."""

    method: str
    prescription: str
    cutoff_parameter: sp.Symbol | None = None
    regulator: sp.Expr | None = None
    singular_locus: sp.Expr | None = None
    cutoff_domain: sp.Expr | None = None
    cutoff_integral: sp.Expr | None = None
    asymptotic_expansion: sp.Expr | None = None
    divergent_terms: tuple[sp.Expr, ...] = ()
    constant_term: sp.Expr | None = None
    shared_cutoff: bool = True
    backend: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    certificate: RegularizationCertificate = RegularizationCertificate()


@dataclass(frozen=True)
class RegularizationResult:
    """Structured result for principal-value and finite-part evaluation."""

    value: sp.Expr
    status: str
    method: str
    prescription: object
    provenance: RegularizationProvenance
    exact: bool = True
    complete: bool | None = True
    assumptions: sp.Expr = sp.true
    limitations: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    cases: tuple[IntegralCase, ...] = ()
    parameter_strata: tuple[ParameterStratum, ...] = ()
    convergent: bool | None = None
    coverage: tuple[CoverageEvidence, ...] = ()
    convergence_evidence: tuple[ConvergenceEvidence, ...] = ()
    _convergence_policy: object = field(default=None, repr=False, compare=False)

    @property
    def outcome(self) -> IntegrationOutcome:
        return outcome_for_status(self.status)

    @property
    def finite(self) -> bool | None:
        return _finite_outcome(self.outcome)

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
        pieces = [f"status={self.status}", f"method={self.method}"]
        if self.limitations:
            pieces.append("limitations=" + "; ".join(self.limitations))
        return "; ".join(pieces)


_REGULARIZATION_TRACE: ContextVar[tuple[RegularizationProvenance, ...] | None] = ContextVar(
    "multiple_integrate_regularization_trace", default=None
)


@contextmanager
def _regularization_trace_context() -> Iterator[None]:
    token = _REGULARIZATION_TRACE.set(())
    try:
        yield
    finally:
        _REGULARIZATION_TRACE.reset(token)


def _record_regularization(provenance: RegularizationProvenance) -> None:
    current = _REGULARIZATION_TRACE.get()
    if current is not None:
        _REGULARIZATION_TRACE.set((*current, provenance))


def _regularization_trace() -> tuple[RegularizationProvenance, ...]:
    return _REGULARIZATION_TRACE.get() or ()


def _divergent_terms(expansion: sp.Expr, epsilon: sp.Symbol) -> tuple[sp.Expr, ...]:
    terms = []
    for term in sp.Add.make_args(sp.expand(expansion)):
        if term.has(sp.Order):
            continue
        power = term.as_powers_dict().get(epsilon, sp.Integer(0))
        has_negative_power = power.is_negative is True
        has_log = term.has(sp.log(epsilon))
        if has_negative_power or (has_log and power.is_nonpositive is True):
            terms.append(term)
    return tuple(sorted(terms, key=sp.default_sort_key))


def finite_part_from_cutoff(
    cutoff_integral: sp.Expr,
    epsilon: sp.Symbol,
    *,
    singular_locus: sp.Expr | None = None,
    cutoff_domain: sp.Expr | None = None,
    backend: str | None = None,
    details: dict[str, Any] | None = None,
) -> tuple[sp.Expr, RegularizationProvenance]:
    """Extract the Hadamard finite part as the epsilon^0 asymptotic coefficient."""
    expr = sp.sympify(cutoff_integral)
    # The cutoff must already have been evaluated on its *truncated* domain.
    # Passing a RegionIntegral or unevaluated inner integral through asymptotic
    # extraction can otherwise produce a spurious finite certificate whose
    # constant term still contains an unregularized singular integral.
    unresolved = (sp.Integral, sp.Limit, UnevaluatedRegionIntegral)
    if expr.has(*unresolved):
        raise UnresolvedIntegration(
            "the shared-cutoff integral is unevaluated; no finite part is certified"
        )
    policy = _CURRENT_POLICY.get()
    free_parameters = tuple(sorted(expr.free_symbols - {epsilon}, key=sp.default_sort_key))
    budget = policy.asymptotic
    if not policy.allow_expensive_exact and len(free_parameters) > budget.max_variables:
        raise StrategyBudgetExceeded(
            f"asymptotic parameter count {len(free_parameters)} exceeds {budget.max_variables}"
        )
    numerator, denominator = sp.cancel(expr).as_numer_denom()
    degrees = [
        degree
        for candidate in (numerator, denominator)
        if (degree := polynomial_degree(candidate, (epsilon,))) is not None
    ]
    if not policy.allow_expensive_exact and degrees and max(degrees) > budget.max_polynomial_degree:
        raise StrategyBudgetExceeded(
            f"asymptotic cutoff degree {max(degrees)} exceeds {budget.max_polynomial_degree}"
        )
    expansion, asymptotic_backend = cutoff_expansion(expr, epsilon)
    divergent = _divergent_terms(expansion, epsilon)
    finite_expr = sp.expand(expansion - sp.Add(*divergent))
    constant = sp.limit(finite_expr, epsilon, 0, dir="+")
    if constant.has(sp.oo, -sp.oo, sp.zoo, sp.nan, *unresolved):
        raise UnresolvedIntegration(
            "cutoff asymptotics did not expose a fully evaluated finite Hadamard constant"
        )
    provenance = RegularizationProvenance(
        method="hadamard_finite_part",
        prescription="HadamardFinitePart",
        cutoff_parameter=epsilon,
        regulator=epsilon,
        singular_locus=singular_locus,
        cutoff_domain=cutoff_domain,
        cutoff_integral=expr,
        asymptotic_expansion=expansion,
        divergent_terms=divergent,
        constant_term=sp.simplify(constant),
        shared_cutoff=True,
        backend=backend or asymptotic_backend,
        details={**dict(details or {}), "asymptotic_backend": asymptotic_backend},
        certificate=RegularizationCertificate(
            claims=(
                ("cutoff_parameter_positive", epsilon),
                ("cutoff_integral", expr),
                ("asymptotic_expansion", expansion),
                ("divergent_terms", divergent),
                ("constant_term", sp.simplify(constant)),
                ("shared_cutoff", True),
            )
        ),
    )
    return sp.simplify(constant), provenance


def verify_regularization_result(result: RegularizationResult) -> RegularizationVerification:
    """Replay the algebraic claims behind a structured regularization result.

    Verification is independent of planner provenance: it checks the cutoff
    expansion, divergent-term classification, constant extraction, and any
    supplied algebraic geometry claims that can be replayed locally.
    """
    if not isinstance(result, RegularizationResult):
        raise TypeError("result must be a RegularizationResult")
    if result.outcome is not IntegrationOutcome.FINITE:
        return RegularizationVerification(
            False,
            (("finite_outcome", False),),
            ("result does not establish a finite regularized value",),
        )
    provenance = result.provenance
    if (
        provenance.singular_locus is not None or provenance.cutoff_domain is not None
    ) and not isinstance(provenance.regulator, sp.Basic):
        return RegularizationVerification(
            False, (("regulator", False),), ("missing or malformed regulator",)
        )
    checks: list[tuple[str, bool]] = []
    failures: list[str] = []
    value_checked = False

    if provenance.backend == "dominated_convergence":
        from .regular_density import regular_density_prerequisites

        details = provenance.details
        required = {"density", "ranges", "polynomial", "assumptions"}
        if not required <= details.keys():
            return RegularizationVerification(
                False, (("prerequisites", False),), ("missing dominated-convergence prerequisites",)
            )
        density, ranges, polynomial = details["density"], details["ranges"], details["polynomial"]
        proved = regular_density_prerequisites(
            density, ranges, polynomial, assumptions=details["assumptions"]
        )
        epsilon = provenance.cutoff_parameter
        domain = sp.And(*(a for v, lo, hi in ranges for a in (v >= lo, v <= hi)))
        geometry = (
            epsilon is not None
            and epsilon.is_positive is True
            and (
                provenance.regulator == sp.Abs(polynomial)
                and provenance.singular_locus == sp.Eq(polynomial, 0)
                and provenance.cutoff_domain == sp.And(domain, polynomial**2 > epsilon**2)
            )
        )
        ordinary = sp.integrate(density, *ranges) if proved else sp.nan
        same = (
            provenance.constant_term is not None
            and sp.simplify(ordinary - result.value) == 0
            and sp.simplify(provenance.constant_term - result.value) == 0
        )
        checks = (
            ("prerequisites", bool(proved)),
            ("cutoff_geometry", bool(geometry)),
            ("ordinary_value", bool(same)),
        )
        valid = all(ok for _, ok in checks)
        return RegularizationVerification(
            valid,
            checks,
            tuple(name for name, ok in checks if not ok),
            "dominated_convergence" if valid else "none",
        )

    epsilon = provenance.cutoff_parameter
    if epsilon is not None and provenance.cutoff_integral is not None:
        expansion, _backend = cutoff_expansion(provenance.cutoff_integral, epsilon)
        stored_expansion = (
            expansion
            if provenance.asymptotic_expansion is None
            else provenance.asymptotic_expansion
        )
        expanded = sp.expand(stored_expansion)
        expansion_ok = sp.simplify(sp.expand(expansion) - expanded) == 0
        checks.append(("asymptotic_expansion", bool(expansion_ok)))
        if not expansion_ok:
            failures.append("stored asymptotic expansion does not replay")
        divergent = _divergent_terms(expanded, epsilon)
        divergence_ok = set(divergent) == set(provenance.divergent_terms)
        checks.append(("divergent_terms", divergence_ok))
        if not divergence_ok:
            failures.append("stored divergent terms do not match the expansion")
        finite_expr = sp.expand(expanded - sp.Add(*divergent))
        try:
            constant = sp.limit(finite_expr, epsilon, 0, dir="+")
            constant_ok = sp.simplify(constant - result.value) == 0
        except (TypeError, ValueError, NotImplementedError, sp.PoleError):
            constant_ok = False
        checks.append(("constant_term", constant_ok))
        value_checked = True
        if not constant_ok:
            failures.append("stored regularized value is not the replayed constant term")
    if provenance.constant_term is not None:
        constant_ok = sp.simplify(provenance.constant_term - result.value) == 0
        name = "provenance_constant_term" if value_checked else "constant_term"
        checks.append((name, constant_ok))
        value_checked = True
        if not constant_ok:
            failures.append("provenance constant term differs from result value")

    product_realization = provenance.details.get("realization")
    if product_realization in (
        "coordinate_product_cutoff",
        "affine_product_cutoff",
        "polynomial_graph_cutoff",
        "triangular_polynomial_cutoff",
    ):
        from .region_types import BoxRegion
        from .singular_geometry import product_level_cutoff

        details = provenance.details
        try:
            affine = product_realization == "affine_product_cutoff"
            mapped = affine or product_realization in (
                "polynomial_graph_cutoff",
                "triangular_polynomial_cutoff",
            )
            source = details["source_formula"] if mapped else BoxRegion(details["source_ranges"])
            replay = product_level_cutoff(
                details["source_density"],
                source,
                details["source_level"],
                epsilon,
                variables=details.get("source_variables"),
                assumptions=details["assumptions"],
            )
            geometry_ok = (
                replay is not None and sp.simplify(replay[0] - provenance.cutoff_integral) == 0
            )
            if geometry_ok and affine:
                geometry_ok = all(
                    details[key] == replay[1][key]
                    for key in ("affine_matrix", "inverse_matrix", "affine_jacobian")
                )
            if geometry_ok and product_realization == "polynomial_graph_cutoff":
                geometry_ok = details["graph_jacobian"] == replay[1]["graph_jacobian"]
            if geometry_ok and product_realization == "triangular_polynomial_cutoff":
                rebound = dict(
                    zip(replay[1]["target_variables"], details["target_variables"], strict=True)
                )
                geometry_ok = (
                    details["triangular_jacobian"] == replay[1]["triangular_jacobian"]
                    and details["defining_forms"] == replay[1]["defining_forms"]
                    and all(
                        sp.simplify(a - b.xreplace(rebound)) == 0
                        for a, b in zip(
                            details["forward_map"], replay[1]["forward_map"], strict=True
                        )
                    )
                )
            domain = (
                details["source_formula"]
                if mapped
                else sp.And(
                    *(atom for v, lo, hi in details["source_ranges"] for atom in (v >= lo, v <= hi))
                )
            )
            level = details["source_level"]
            geometry_ok = geometry_ok and (
                (
                    provenance.regulator == epsilon
                    or (
                        isinstance(provenance.regulator, sp.Basic)
                        and provenance.regulator.func is sp.Abs
                        and sp.simplify(provenance.regulator.args[0] - level) == 0
                    )
                )
                and isinstance(provenance.singular_locus, sp.Equality)
                and sp.simplify(
                    provenance.singular_locus.lhs - provenance.singular_locus.rhs - level
                )
                == 0
                and provenance.cutoff_domain == sp.And(domain, level**2 > epsilon**2)
            )
        except (KeyError, TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            geometry_ok = False
        checks.append(("product_cutoff_geometry", bool(geometry_ok)))
        if not geometry_ok:
            failures.append("product cutoff or affine Jacobian does not replay")

    shape_matrix = provenance.details.get("shape_matrix")
    if shape_matrix is not None:
        matrix = sp.ImmutableMatrix(shape_matrix)
        symmetric = matrix == matrix.T
        positive = matrix.is_positive_definite is True
        checks.extend((("shape_matrix_symmetric", symmetric), ("shape_matrix_positive", positive)))
        if not symmetric or not positive:
            failures.append(
                "ellipsoidal cutoff matrix is not certified symmetric positive definite"
            )

    atlas_regular = provenance.details.get("regularity_certified")
    if atlas_regular is not None:
        ok = atlas_regular is True
        checks.append(("hypersurface_regularity", ok))
        if not ok:
            failures.append("hypersurface atlas lacks a regularity certificate")

    if not value_checked:
        failures.append("no replayable regularized-value evidence was supplied")
    strength = (
        "cutoff_replay"
        if epsilon is not None and provenance.cutoff_integral is not None
        else ("constant_consistency" if value_checked else "none")
    )
    limitations = (
        (
            "cutoff replay reuses the supported family algorithm; it is not an independent integration proof",
        )
        if product_realization
        in (
            "coordinate_product_cutoff",
            "affine_product_cutoff",
            "polynomial_graph_cutoff",
            "triangular_polynomial_cutoff",
        )
        else (
            "verification does not independently derive the cutoff integral or certify its geometry",
        )
        if strength == "cutoff_replay"
        else ("constant equality checks internal consistency, not the regulator or integration",)
    )
    return RegularizationVerification(
        not failures, tuple(checks), tuple(failures), strength, limitations
    )


__all__ = [
    "HadamardFinitePart",
    "RegularizationCertificate",
    "RegularizationProvenance",
    "RegularizationVerification",
    "RegularizationResult",
    "finite_part_from_cutoff",
    "verify_regularization_result",
]
