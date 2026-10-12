"""Scoped sufficient convergence theorems, separate from evaluated values."""

from __future__ import annotations

import sympy as sp

from .adapters.semialg import is_satisfiable
from .assumptions import _ask, _normalize_assumptions
from .policy import _normalize_policy, check_semialg_budget
from .results import ConvergenceEvidence, IntegrationStatus


def _true(predicate, assumptions):
    return _ask(predicate, _normalize_assumptions(assumptions)) is True


def _finite(expr, assumptions):
    from .cell_proofs import finite_parameter_domain

    context = _normalize_assumptions(assumptions)
    if not finite_parameter_domain(expr, context):
        return False
    if expr.func in (sp.gamma, sp.loggamma):
        return _finite(expr.args[0], context)
    if expr.func is sp.polygamma:
        order, argument = expr.args
        return (
            order.is_integer is True and order.is_nonnegative is True and _finite(argument, context)
        )
    return expr.is_finite is True or _ask(sp.Q.finite(expr), context) is True


def compact_continuity(f, ranges, *, assumptions=None, policy=None):
    """Prove continuity on a finite closed box for selected expression families."""
    if not ranges or not all(len(r) == 3 for r in ranges):
        return False
    coords = tuple(r[0] for r in ranges)
    if not all(
        _true(sp.Q.real(sp.sympify(b)) & sp.Q.finite(sp.sympify(b)), assumptions)
        and not sp.sympify(b).has(*coords)
        for r in ranges
        for b in r[1:]
    ):
        return False

    def positive(base, strict):
        # Coordinate Symbol assumptions do not restrict the requested interval.
        # Rebind them as unconstrained real variables before a domain proof.
        if not base.has(*coords):
            return _true(sp.Q.positive(base) if strict else sp.Q.nonnegative(base), assumptions)
        fresh = {v: sp.Dummy(v.name, real=True) for v in coords}
        rebound = base.xreplace(fresh)
        bad = sp.Le(rebound, 0) if strict else sp.Lt(rebound, 0)
        parameter_assumptions = tuple(
            a for a in _normalize_assumptions(assumptions).original if not a.has(*coords)
        )
        domain = sp.And(
            *(
                atom
                for v, lo, hi in ranges
                for atom in (fresh[v] >= sp.Min(lo, hi), fresh[v] <= sp.Max(lo, hi))
            )
        )
        formula = sp.And(domain, *parameter_assumptions, bad)
        variables = tuple(fresh.values())
        allowed, _ = check_semialg_budget(formula, variables, _normalize_policy(policy))
        return allowed and is_satisfiable(formula, variables) is False

    def continuous(expr):
        expr = sp.sympify(expr)
        if not expr.has(*coords):
            return _finite(expr, assumptions)
        try:
            poly = sp.Poly(expr, *coords)
            return all(_finite(c, assumptions) for c in poly.coeffs())
        except sp.PolynomialError:
            pass
        if expr.is_Add or expr.is_Mul:
            return all(continuous(arg) for arg in expr.args)
        if expr.func in (sp.exp, sp.sin, sp.cos, sp.Abs):
            return continuous(expr.args[0])
        if expr.func is sp.log:
            return continuous(expr.args[0]) and positive(expr.args[0], True)
        if expr.is_Pow:
            base, power = expr.args
            if (
                power.has(*coords)
                or not continuous(base)
                or not continuous(power)
                or power.is_finite is not True
            ):
                return False
            if power.is_integer is True and power.is_nonnegative is True:
                return True
            return positive(base, power.is_nonnegative is not True)
        return False

    return continuous(sp.sympify(f))


def _one_dimensional_absolute(f, range_, assumptions, policy=None):
    v, lo, hi = map(sp.sympify, range_)
    coefficient, power = f.as_coeff_exponent(v)
    if not coefficient.has(v) and power.is_real is True and _finite(coefficient, assumptions):
        if lo == 0 and hi == 1 and _true(sp.Q.positive(power + 1), assumptions):
            return True
        if lo == 1 and hi == sp.oo and _true(sp.Q.negative(power + 1), assumptions):
            return True
    from .selected_families import selected_interval_rule

    if selected_interval_rule(f, v, lo, hi, assumptions, evaluate=False, policy=policy) is not None:
        return True
    # A polynomial times a strictly decaying Gaussian has all finite moments.
    exponentials = list(f.atoms(sp.exp))
    if len(exponentials) == 1 and lo == -sp.oo and hi == sp.oo:
        e = exponentials[0]
        try:
            q = sp.Poly(e.args[0], v)
            p = sp.Poly(sp.cancel(f / e), v)
            return (
                q.degree() == 2
                and _true(sp.Q.negative(q.nth(2)), assumptions)
                and all(c.is_real is True and _finite(c, assumptions) for c in q.all_coeffs())
                and all(_finite(c, assumptions) for c in p.all_coeffs())
            )
        except sp.PolynomialError:
            pass
    return None


def _base_absolute_rule(f, ranges, *, assumptions=None, policy=None):
    """Return (three-valued claim, method) for a small explicit theorem corpus."""
    f = sp.sympify(f)
    if not ranges or not all(
        isinstance(r, (tuple, list, sp.Tuple)) and len(r) == 3 for r in ranges
    ):
        return None, "not_established"
    if compact_continuity(f, ranges, assumptions=assumptions, policy=policy):
        return True, "compact_continuous"
    coords = tuple(r[0] for r in ranges)
    if len(set(coords)) != len(coords) or not all(isinstance(v, sp.Symbol) for v in coords):
        return None, "not_established"
    if any(sp.sympify(b).has(*coords) for r in ranges for b in r[1:]):
        return None, "not_established"
    from .cell_proofs import finite_external_parameters

    coefficient, _ = f.as_independent(*coords, as_Add=False)
    context = _normalize_assumptions(assumptions)
    if (
        _true(sp.Q.zero(coefficient), assumptions)
        and _finite(coefficient, assumptions)
        and finite_external_parameters(f, coords, context)
        and all(
            _true(sp.Q.extended_real(sp.sympify(b)), assumptions) for r in ranges for b in r[1:]
        )
    ):
        return True, "zero_amplitude"
    # Radial power comparison: local order versus dimension, or full-space tail.
    squared = sp.Add(*(v**2 for v in coords))
    if f.is_Pow and f.exp.is_real is True:
        base, exponent = f.args
        local = sp.expand(base - squared) == 0
        tail = sp.expand(base - squared - 1) == 0
        centered = all(
            _true(sp.Q.negative(sp.sympify(lo)), assumptions)
            and _true(sp.Q.positive(sp.sympify(hi)), assumptions)
            and sp.sympify(lo).is_finite is True
            and sp.sympify(hi).is_finite is True
            for _, lo, hi in ranges
        )
        full = all(lo == -sp.oo and hi == sp.oo for _, lo, hi in ranges)
        threshold = 2 * exponent + len(coords)
        if (local and centered) or (tail and full):
            good = sp.Q.positive(threshold) if local else sp.Q.negative(threshold)
            bad = sp.Q.nonpositive(threshold) if local else sp.Q.nonnegative(threshold)
            if _true(good, assumptions):
                return True, "radial_power_comparison"
            if _true(bad, assumptions):
                return False, "radial_power_comparison"
    # A real quadratic exponent with a positive definite negative Hessian
    # dominates every polynomial moment, including coupled Gaussian factors.
    exponentials = list(f.atoms(sp.exp))
    if len(exponentials) == 1 and all(lo == -sp.oo and hi == sp.oo for _, lo, hi in ranges):
        e = exponentials[0]
        try:
            q = sp.Poly(e.args[0], *coords)
            p = sp.Poly(sp.cancel(f / e), *coords)
            matrix = -sp.hessian(e.args[0], coords)
            real_finite = all(_true(sp.Q.real(c) & sp.Q.finite(c), assumptions) for c in q.coeffs())
            finite_density = all(_true(sp.Q.finite(c), assumptions) for c in p.coeffs())
            if (
                q.total_degree() == 2
                and real_finite
                and finite_density
                and all(
                    _true(sp.Q.positive(matrix[:i, :i].det()), assumptions)
                    for i in range(1, len(coords) + 1)
                )
            ):
                return True, "positive_definite_gaussian_moments"
        except sp.PolynomialError:
            pass
    factors = sp.separatevars(f, symbols=coords, dict=True)
    if factors is not None and _finite(factors.get("coeff", sp.S.One), assumptions):
        claims = [
            _one_dimensional_absolute(factors.get(v, sp.S.One), r, assumptions, policy)
            for v, r in zip(coords, ranges, strict=True)
        ]
        if all(c is True for c in claims):
            return True, "absolute_product_theorem"
    return None, "not_established"


def absolute_rule(f, ranges, *, assumptions=None, policy=None, _depth=0):
    """Sufficient absolute comparison proofs, with bounded recursive work."""
    claim, method = _base_absolute_rule(f, ranges, assumptions=assumptions, policy=policy)
    if claim is not None:
        return claim, method
    if not ranges or any(not isinstance(r, (tuple, list, sp.Tuple)) or len(r) != 3 for r in ranges):
        return None, "not_established"
    coords = tuple(r[0] for r in ranges)
    if any(sp.sympify(b).has(*coords) for r in ranges for b in r[1:]):
        return None, "not_established"
    f = sp.sympify(f)
    from .anisotropic import anisotropic_absolute
    from .convergence_comparison import (
        affine_laurent_log_absolute,
        affine_power_absolute,
        bounded_multiplier,
        local_polynomial_comparison,
        polynomial_majorant,
        power_log_absolute,
        scaled_quadratic_power,
    )
    from .quadratic_radial import weighted_quadratic_data
    from .selected_families import (
        coercive_exponential_absolute,
        positive_even_tail_absolute,
        selected_interval_rule,
    )

    if len(ranges) == 1:
        selected = selected_interval_rule(f, *ranges[0], assumptions, evaluate=False, policy=policy)
        if selected is not None:
            return True, selected[0]
    from .nonseparable_improper import improper_data

    if improper_data(f, ranges, assumptions, policy=policy) is not None:
        return True, "nonseparable_improper_comparison"
    if coercive_exponential_absolute(f, ranges, assumptions, policy=policy) is True:
        return True, "coercive_even_exponential_comparison"
    if positive_even_tail_absolute(f, ranges, assumptions, policy=policy) is True:
        return True, "positive_even_tail_comparison"
    anisotropic = anisotropic_absolute(f, ranges, assumptions, policy)
    if anisotropic is not None:
        return anisotropic, "weighted_even_power_comparison"
    if len(ranges) == 1 and (
        affine_power_absolute(f, ranges[0], assumptions) is True
        or power_log_absolute(f, ranges[0], assumptions) is True
        or affine_laurent_log_absolute(f, ranges[0], assumptions) is True
    ):
        return True, "endpoint_power_log_comparison"
    if local_polynomial_comparison(f, ranges, assumptions) is True:
        return True, "local_polynomial_comparison"
    scaled = scaled_quadratic_power(f, ranges, assumptions)
    if scaled is not None:
        return scaled, "affine_quadratic_comparison"
    if (
        all(lo == -sp.oo and hi == sp.oo for _, lo, hi in ranges)
        and weighted_quadratic_data(f, coords, assumptions=assumptions, policy=policy) is not None
    ):
        return True, "weighted_quadratic_comparison"
    if _depth >= 3:
        return None, "not_established"
    budget = _normalize_policy(policy)

    def proved(expr):
        return (
            absolute_rule(expr, ranges, assumptions=assumptions, policy=policy, _depth=_depth + 1)[
                0
            ]
            is True
        )

    if (
        f.is_Add
        and len(f.args) <= budget.semialg.max_pieces
        and all(proved(term) for term in f.args)
    ):
        return True, "finite_sum_comparison"
    if f.is_Mul and len(f.args) <= budget.semialg.max_pieces:
        for factor in f.args:
            for cosine in factor.atoms(sp.cos):
                if factor == 1 - cosine or factor == cosine - 1:
                    fresh = {v: sp.Dummy(v.name, real=True) for v in coords}
                    if _true(sp.Q.real(cosine.args[0].xreplace(fresh)), assumptions) and proved(
                        (f / factor) * cosine.args[0] ** 2
                    ):
                        return True, "quadratic_vanishing_multiplier_comparison"
            if factor.func == sp.sin:
                fresh = {v: sp.Dummy(v.name, real=True) for v in coords}
                if _true(sp.Q.real(factor.args[0].xreplace(fresh)), assumptions) and proved(
                    (f / factor) * factor.args[0]
                ):
                    return True, "vanishing_multiplier_comparison"
            if bounded_multiplier(
                factor,
                coords,
                ranges,
                assumptions,
                lambda e, rs, **kw: compact_continuity(e, rs, policy=policy, **kw),
            ) and proved(f / factor):
                return True, "bounded_multiplier_comparison"
    majorant = polynomial_majorant(f, coords, assumptions)
    if majorant is not None and proved(majorant):
        return True, "positive_polynomial_majorant"
    return None, "not_established"


def region_absolute_rule(f, region, *, assumptions=None, policy=None, variables=None):
    """Propagate existence over a finite union of supported native domains."""
    from .quadratic_radial import radial_region_data
    from .region_types import BoxRegion, UnionRegion

    if isinstance(region, BoxRegion):
        return absolute_rule(f, region.ranges, assumptions=assumptions, policy=policy)[0]
    if isinstance(region, UnionRegion):
        budget = _normalize_policy(policy)
        if len(region.pieces) <= budget.semialg.max_pieces and all(
            region_absolute_rule(f, p, assumptions=assumptions, policy=policy) is True
            for p in region.pieces
        ):
            return True
        return None
    if (
        radial_region_data(f, region, assumptions=assumptions, policy=policy, variables=variables)
        is not None
    ):
        return True
    return None


def convergence_evidence(
    f, ranges, *, status, method, prescribed=False, theorem_cases=(), assumptions=None, policy=None
):
    established = (
        True
        if status
        in (
            IntegrationStatus.EXACT,
            IntegrationStatus.PRINCIPAL_VALUE,
            IntegrationStatus.FINITE_PART,
        )
        else False
        if status is IntegrationStatus.DIVERGENT
        else None
    )
    evidence = [
        ConvergenceEvidence(
            "prescribed" if prescribed else "iterated",
            established,
            method,
            details=(("basis", "evaluator outcome; not an independent convergence certificate"),),
        )
    ]
    evidence.extend(
        ConvergenceEvidence("ordinary", c.convergent, "convergence_theorem", c.condition)
        for c in theorem_cases
    )
    absolute, proof = absolute_rule(f, ranges, assumptions=assumptions, policy=policy)
    # Radial powers are positive off a null set. Absolute divergence therefore
    # establishes ordinary divergence too; this implication is not general.
    if absolute is not None:
        evidence = [e for e in evidence if e.kind != "ordinary"]
        evidence.append(ConvergenceEvidence("ordinary", absolute, proof))
    elif not theorem_cases:
        evidence.append(ConvergenceEvidence("ordinary", None, "not_established"))
    evidence.append(ConvergenceEvidence("absolute", absolute, proof))
    return tuple(evidence)
