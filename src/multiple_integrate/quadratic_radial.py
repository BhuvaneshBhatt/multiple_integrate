"""Certified polynomial-weighted Gaussian and quadratic radial moments."""

from __future__ import annotations

from functools import lru_cache

import sympy as sp

from .assumptions import _ask, _normalize_assumptions, _refine_with_assumptions
from .planner import record_convergence, record_coverage, record_strategy
from .policy import _normalize_policy
from .recognition import _simplify_preserving_sums
from .region_types import (
    AnnulusRegion,
    BallRegion,
    DiskRegion,
    EllipsoidRegion,
    SphericalShellRegion,
)


def _proved(predicate, assumptions):
    return _ask(predicate, _normalize_assumptions(assumptions)) is True


def positive_quadratic(q, variables, assumptions=None):
    """Return SPD matrix, center and minimum for a certified real quadratic."""
    try:
        poly = sp.Poly(q, *variables)
        if poly.total_degree() != 2 or not all(
            _proved(sp.Q.real(c) & sp.Q.finite(c), assumptions) for c in poly.coeffs()
        ):
            return None
        matrix = sp.hessian(q, variables) / 2
        if not all(
            _proved(sp.Q.positive(matrix[:i, :i].det()), assumptions)
            for i in range(1, len(variables) + 1)
        ):
            return None
        linear = sp.Matrix([sp.diff(q, v).subs(dict.fromkeys(variables, 0)) for v in variables])
        center = -matrix.inv() * linear / 2
        minimum = sp.simplify(q.subs(dict(zip(variables, center, strict=True)), simultaneous=True))
        return sp.ImmutableMatrix(matrix), tuple(center), minimum
    except (sp.PolynomialError, ValueError, ZeroDivisionError):
        return None


def _profile(expr, variables, q, assumptions):
    """Split P(x)*C*exp(-a*q)*q**s or P(x)*C*(b+q)**s."""
    polynomial = sp.S.One
    scale = sp.S.One
    gaussian = None
    power = None
    for factor in sp.Mul.make_args(expr):
        try:
            sp.Poly(factor, *variables)
            polynomial *= factor
            continue
        except sp.PolynomialError:
            pass
        argument = (
            factor.args[0] if factor.func == sp.exp else factor.base if factor.is_Pow else None
        )
        if argument is None:
            return None
        try:
            argument_poly = sp.Poly(argument, *variables)
            if argument_poly.total_degree() != 2:
                return None
            qpoly = sp.Poly(q, *variables)
            monomial = next(m for m, c in qpoly.terms() if sum(m) == 2 and c != 0)
            ratio = sp.simplify(
                argument_poly.coeff_monomial(monomial) / qpoly.coeff_monomial(monomial)
            )
            offset = sp.simplify(argument - ratio * q)
            if offset.has(*variables) or not _proved(
                sp.Q.real(offset) & sp.Q.finite(offset), assumptions
            ):
                return None
            if factor.func == sp.exp:
                if gaussian is not None or not _proved(sp.Q.negative(ratio), assumptions):
                    return None
                gaussian = -ratio
                scale *= sp.exp(offset)
            else:
                if (
                    power is not None
                    or factor.exp.has(*variables)
                    or not _proved(sp.Q.positive(ratio), assumptions)
                    or not _proved(sp.Q.real(factor.exp) & sp.Q.finite(factor.exp), assumptions)
                ):
                    return None
                shift = sp.simplify(offset / ratio)
                if _proved(sp.Q.zero(shift), assumptions):
                    shift = sp.S.Zero
                power = (shift, factor.exp)
                scale *= ratio**factor.exp
        except sp.PolynomialError:
            return None
    # A mixed Gaussian and shifted power needs a richer special-function route.
    if (
        gaussian is not None
        and power is not None
        and power[0] != 0
        and not _proved(sp.Q.positive(power[0]) & sp.Q.negative(power[1]), assumptions)
    ):
        return None
    if gaussian is None and power is None:
        return None
    try:
        poly = sp.Poly(sp.expand(polynomial), *variables)
    except sp.PolynomialError:
        return None
    if not all(_proved(sp.Q.finite(c), assumptions) for c in poly.coeffs()):
        return None
    return poly, sp.simplify(scale), gaussian, power


def _candidates(expr, variables):
    seen = set()
    for factor in sp.Mul.make_args(expr):
        arg = factor.args[0] if factor.func == sp.exp else factor.base if factor.is_Pow else None
        if arg is not None:
            for q in (arg, -arg):
                q = sp.expand(q)
                if q not in seen:
                    seen.add(q)
                    yield q


def weighted_quadratic_data(expr, variables, *, assumptions=None, policy=None):
    """Recognize a full-space weighted SPD quadratic profile, without evaluating."""
    variables = tuple(variables)
    if not variables:
        return None
    budget = _normalize_policy(policy)
    if not budget.allow_expensive_exact and len(variables) > budget.semialg.max_variables:
        return None
    for candidate in _candidates(sp.sympify(expr), variables):
        data = positive_quadratic(candidate, variables, assumptions)
        if data is None:
            continue
        matrix, center, minimum = data
        q = sp.expand(candidate - minimum)
        split = _profile(sp.sympify(expr), variables, q, assumptions)
        if split is None:
            continue
        poly, scale, gaussian, power = split
        local = tuple(sp.Dummy("quadratic_local", real=True) for _ in variables)
        translated = sp.Poly(
            sp.expand(
                poly.as_expr().subs(
                    {v: u + c for v, u, c in zip(variables, local, center, strict=True)},
                    simultaneous=True,
                )
            ),
            *local,
        )
        if not budget.allow_expensive_exact and (
            translated.total_degree() > budget.semialg.max_polynomial_degree
            or len(translated.terms()) > budget.semialg.max_pieces
        ):
            continue
        if _moments_converge(
            translated, len(variables), gaussian, power, sp.S.Zero, sp.oo, assumptions
        ):
            return matrix, translated, scale, gaussian, power
    return None


def _moments_converge(poly, dimension, gaussian, power, inner, outer, assumptions):
    b, exponent = power if power is not None else (sp.S.Zero, sp.S.Zero)
    if power is not None and b != 0 and not _proved(sp.Q.positive(b), assumptions):
        return False
    for monomial, c in poly.terms():
        if c == 0:
            continue
        alpha = sp.Rational(dimension + sum(monomial), 2)
        if inner == 0 and b == 0 and not _proved(sp.Q.positive(alpha + exponent), assumptions):
            return False
        if (
            outer == sp.oo
            and gaussian is None
            and not _proved(sp.Q.negative(alpha + exponent), assumptions)
        ):
            return False
    return True


def _finite_beta(alpha, beta, z):
    """Bounded elementary expansion of integer-parameter incomplete beta moments."""
    if alpha.is_Integer and 0 < alpha <= 16 and beta.is_integer is True:
        return sp.Add(
            *(
                (-1) ** j
                * sp.binomial(alpha - 1, j)
                * (-sp.log(1 - z) if beta + j == 0 else (1 - (1 - z) ** (beta + j)) / (beta + j))
                for j in range(int(alpha))
            )
        )
    return sp.betainc(alpha, beta, 0, z)


def _moment(alpha, gaussian, power, inner, outer):
    b, exponent = power if power is not None else (sp.S.Zero, sp.S.Zero)
    if gaussian is not None and b != 0:
        s = -exponent
        if outer == sp.oo and inner == 0:
            return (
                b ** (alpha - s)
                * sp.meijerg(((1 - alpha,), ()), ((0, s - alpha), ()), gaussian * b)
                / (2 * sp.gamma(s))
            )
        if alpha.is_Integer and 0 < alpha <= 16:
            return (
                sp.exp(gaussian * b)
                / 2
                * sp.Add(
                    *(
                        sp.binomial(alpha - 1, j)
                        * (-b) ** (alpha - 1 - j)
                        * (
                            sp.uppergamma(1 + j - s, gaussian * (b + inner**2))
                            - sp.uppergamma(1 + j - s, gaussian * (b + outer**2))
                        )
                        / gaussian ** (1 + j - s)
                        for j in range(int(alpha))
                    )
                )
            )
        if alpha.is_Rational and (2 * alpha).is_Integer and 0 < alpha <= 16 and outer != sp.oo:
            # The exponential Taylor series converges uniformly on a finite radial interval.
            # Integrate each coefficient against the positive shifted-power weight exactly.
            k = sp.Dummy("radial_series_index", integer=True, nonnegative=True)

            def endpoint(radius):
                if radius == 0:
                    return sp.S.Zero
                return (
                    radius ** (2 * (alpha + k))
                    * sp.hyper((s, alpha + k), (alpha + k + 1,), -(radius**2) / b)
                    / (2 * b**s * (alpha + k))
                )

            return sp.Sum(
                (-gaussian) ** k / sp.factorial(k) * (endpoint(outer) - endpoint(inner)),
                (k, 0, sp.oo),
            )
        raise NotImplementedError(
            "finite mixed radial moments exceed the supported radial-weight budget"
        )
    if gaussian is not None:
        alpha += exponent
        if outer == sp.oo:
            return (
                sp.gamma(alpha) / (2 * gaussian**alpha)
                if inner == 0
                else sp.uppergamma(alpha, gaussian * inner**2) / (2 * gaussian**alpha)
            )
        if inner == 0:
            return sp.lowergamma(alpha, gaussian * outer**2) / (2 * gaussian**alpha)
        return (
            sp.uppergamma(alpha, gaussian * inner**2) - sp.uppergamma(alpha, gaussian * outer**2)
        ) / (2 * gaussian**alpha)
    if b != 0:
        if outer == sp.oo:
            return b ** (alpha + exponent) * sp.gamma(alpha) * sp.gamma(-exponent - alpha) / (
                2 * sp.gamma(-exponent)
            ) - (
                sp.S.Zero
                if inner == 0
                else b ** (alpha + exponent)
                * _finite_beta(alpha, -exponent - alpha, inner**2 / (b + inner**2))
                / 2
            )

        def endpoint(r):
            return (
                sp.S.Zero if r == 0 else _finite_beta(alpha, -exponent - alpha, r**2 / (b + r**2))
            )

        return b ** (alpha + exponent) * (endpoint(outer) - endpoint(inner)) / 2
    order = 2 * (alpha + exponent)
    if order == 0:
        return sp.log(outer / inner)
    return (outer**order - (sp.S.Zero if inner == 0 else inner**order)) / order


def _evaluate(matrix, poly, scale, gaussian, power, inner, outer):
    dimension = matrix.rows
    covariance = matrix.inv() / 2

    @lru_cache(None)
    def wick(exponents):
        if sum(exponents) == 0:
            return sp.S.One
        if sum(exponents) % 2:
            return sp.S.Zero
        index = next(i for i, p in enumerate(exponents) if p)
        remaining = list(exponents)
        remaining[index] -= 1
        total = sp.S.Zero
        for j, count in enumerate(remaining):
            if count:
                reduced = remaining.copy()
                reduced[j] -= 1
                total += count * covariance[index, j] * wick(tuple(reduced))
        return total

    angular_scale = 2 * sp.pi ** sp.Rational(dimension, 2) / sp.sqrt(matrix.det())

    @lru_cache(None)
    def moment(alpha):
        return _moment(alpha, gaussian, power, inner, outer)

    total = sp.S.Zero
    for monomial, c in poly.terms():
        angular = (
            angular_scale * wick(monomial) / sp.gamma(sp.Rational(dimension + sum(monomial), 2))
        )
        if angular != 0:
            total += c * angular * moment(sp.Rational(dimension + sum(monomial), 2))
    return _simplify_preserving_sums(scale * total)


def _parameter_tail_value(expr, variables, ranges, assumptions):
    """Retain a possible zero-shift singular fiber instead of using a positive-shift formula."""
    budget = _normalize_policy(None)
    if not budget.allow_expensive_exact and len(variables) > budget.semialg.max_variables:
        return None
    context = _normalize_assumptions(assumptions)
    for candidate in _candidates(sp.sympify(expr), variables):
        quadratic = positive_quadratic(candidate, variables, assumptions)
        if quadratic is None:
            continue
        matrix, center, minimum = quadratic
        split = _profile(expr, variables, sp.expand(candidate - minimum), assumptions)
        if split is None:
            continue
        poly, scale, gaussian, power = split
        if gaussian is not None or power is None:
            continue
        shift, exponent = power
        if not _proved(sp.Q.nonnegative(shift), assumptions) or _proved(
            sp.Q.positive(shift), assumptions
        ):
            continue
        if not all(
            _proved(sp.Q.negative(exponent + sp.Rational(len(variables) + sum(m), 2)), assumptions)
            for m, c in poly.terms()
            if c != 0
        ):
            continue
        # Only constant-phase densities provide this ordinary divergence proof.
        coefficient = sp.simplify(scale * poly.as_expr())
        divergent = sp.Integral(expr, *ranges)
        if poly.total_degree() == 0:
            if _proved(sp.Q.positive(coefficient), assumptions):
                divergent = sp.oo
            elif _proved(sp.Q.negative(coefficient), assumptions):
                divergent = -sp.oo
        if shift == 0:
            return divergent
        positive_context = _normalize_assumptions((*context.original, sp.Gt(shift, 0)))
        data = weighted_quadratic_data(expr, variables, assumptions=positive_context)
        if data is None:
            continue
        finite = _simplify_preserving_sums(
            _refine_with_assumptions(_evaluate(*data, sp.S.Zero, sp.oo), positive_context)
        )
        for kind in ("ordinary", "absolute"):
            record_convergence(kind, True, "weighted_quadratic", shift > 0)
            record_convergence(
                kind,
                False if divergent in (sp.oo, -sp.oo) else None,
                "zero_shift_threshold",
                sp.Eq(shift, 0),
            )
        return sp.Piecewise((finite, shift > 0), (divergent, True))
    return None


def try_weighted_quadratic(expr, ranges, *, assumptions=None):
    """Exact full-space polynomial moments with Gaussian or inverse-quadratic profiles."""
    if not ranges or not all(lo == -sp.oo and hi == sp.oo for _, lo, hi in ranges):
        return None
    data = weighted_quadratic_data(expr, tuple(r[0] for r in ranges), assumptions=assumptions)
    if data is None:
        transition = _parameter_tail_value(
            sp.sympify(expr), tuple(r[0] for r in ranges), ranges, assumptions
        )
        if transition is not None:
            record_strategy("weighted_quadratic")
        return transition
    value = _simplify_preserving_sums(
        _refine_with_assumptions(
            _evaluate(*data, sp.S.Zero, sp.oo), _normalize_assumptions(assumptions)
        )
    )
    record_coverage(
        "absolute_convergence",
        True,
        "weighted_quadratic",
        basis="all polynomial moments satisfy radial convergence thresholds",
    )
    record_coverage(
        "integration_domain",
        True,
        "weighted_quadratic",
        basis="SPD full-space affine radial transformation",
    )
    record_strategy("weighted_quadratic")
    return value


def radial_region_data(expr, region, *, assumptions=None, variables=None, policy=None):
    """Recognize weighted radial profiles on native and canonical radial geometry."""
    variables = tuple(variables or getattr(region, "variables", ()))
    n = len(variables)
    if not n:
        return None
    budget = _normalize_policy(policy)
    if not budget.allow_expensive_exact and n > budget.semialg.max_variables:
        return None
    if isinstance(region, (DiskRegion, BallRegion)):
        center = region.center
        matrix = sp.eye(n)
        inner, outer = sp.S.Zero, region.radius
    elif isinstance(region, EllipsoidRegion):
        center = region.center
        if not all(_proved(sp.Q.positive(a) & sp.Q.finite(a), assumptions) for a in region.axes):
            return None
        matrix = sp.diag(*(a**-2 for a in region.axes))
        inner, outer = sp.S.Zero, sp.S.One
    elif isinstance(region, (AnnulusRegion, SphericalShellRegion)):
        center = (sp.S.Zero,) * n
        matrix = sp.eye(n)
        inner, outer = region.inner_radius, region.outer_radius
    else:
        from .semialg_geometry import radial_geometry_signature

        signature = radial_geometry_signature(region, variables)
        if signature is None:
            return None
        center, matrix, inner, outer = signature
    if len(center) != n or not all(
        _proved(sp.Q.real(c) & sp.Q.finite(c), assumptions) for c in center
    ):
        return None
    if not _proved(
        sp.Q.nonnegative(inner)
        & sp.Q.finite(inner)
        & sp.Q.positive(outer - inner)
        & sp.Q.finite(outer),
        assumptions,
    ):
        return None
    local = tuple(sp.Dummy(f"radial_local{i}", real=True) for i in range(n))
    transformed = sp.sympify(expr).subs(
        {v: c + u for v, c, u in zip(variables, center, local, strict=True)}, simultaneous=True
    )
    q = sp.expand((sp.Matrix(local).T * matrix * sp.Matrix(local))[0])
    if positive_quadratic(q, local, assumptions) is None:
        return None
    split = _profile(transformed, local, q, assumptions)
    if split is None:
        return None
    poly, scale, gaussian, power = split
    if not budget.allow_expensive_exact and (
        poly.total_degree() > budget.semialg.max_polynomial_degree
        or len(poly.terms()) > budget.semialg.max_pieces
    ):
        return None
    if (
        gaussian is not None
        and power is not None
        and power[0] != 0
        and any(
            sum(m) % 2 == 0 and sp.Rational(n + sum(m), 2) > 16 for m, c in poly.terms() if c != 0
        )
    ):
        return None
    if not _moments_converge(poly, n, gaussian, power, inner, outer, assumptions):
        return None
    return matrix, poly, scale, gaussian, power, inner, outer


def _parameter_radial_value(expr, region, variables, assumptions):
    """Keep positive and zero shifts separate on finite radial geometry."""
    budget = _normalize_policy(None)
    if not budget.allow_expensive_exact and len(variables) > budget.semialg.max_variables:
        return None
    context = _normalize_assumptions(assumptions)
    for candidate in _candidates(sp.sympify(expr), variables):
        quadratic = positive_quadratic(candidate, variables, assumptions)
        if quadratic is None:
            continue
        split = _profile(expr, variables, sp.expand(candidate - quadratic[2]), assumptions)
        if split is None or split[2] is not None or split[3] is None:
            continue
        poly, scale, _, (shift, exponent) = split
        if not _proved(sp.Q.nonnegative(shift), assumptions) or _proved(
            sp.Q.positive(shift), assumptions
        ):
            continue
        if shift == 0:
            continue
        positive_context = _normalize_assumptions((*context.original, shift > 0))
        positive = radial_region_data(
            expr, region, variables=variables, assumptions=positive_context
        )
        if positive is None:
            continue
        zero_context = _normalize_assumptions((*context.original, sp.Eq(shift, 0)))
        zero = radial_region_data(expr, region, variables=variables, assumptions=zero_context)
        if zero is not None:
            zero_value = _simplify_preserving_sums(
                _refine_with_assumptions(_evaluate(*zero), zero_context)
            )
        else:
            from .adapters.semialg import semialgebraic_domain
            from .results import UnevaluatedRegionIntegral

            formula, coords = semialgebraic_domain(region, variables)
            zero_value = UnevaluatedRegionIntegral(expr, formula, coords, "ambient", region=region)
            coefficient = sp.simplify(scale * poly.as_expr())
            if (
                positive[-2] == 0
                and poly.total_degree() == 0
                and _proved(
                    sp.Q.nonpositive(exponent + sp.Rational(len(variables), 2)), assumptions
                )
            ):
                if _proved(sp.Q.positive(coefficient), assumptions):
                    zero_value = sp.oo
                elif _proved(sp.Q.negative(coefficient), assumptions):
                    zero_value = -sp.oo
        value = _simplify_preserving_sums(
            _refine_with_assumptions(_evaluate(*positive), positive_context)
        )
        for kind in ("ordinary", "absolute"):
            record_convergence(kind, True, "weighted_radial", shift > 0)
            record_convergence(
                kind,
                True if zero is not None else False if zero_value in (sp.oo, -sp.oo) else None,
                "zero_shift_threshold",
                sp.Eq(shift, 0),
            )
        return sp.Piecewise((value, shift > 0), (zero_value, True))
    return None


def try_weighted_radial_region(expr, region, *, assumptions=None, variables=None):
    """Evaluate a certified polynomial-weighted radial integral on a native region."""
    data = radial_region_data(expr, region, assumptions=assumptions, variables=variables)
    if data is None:
        coords = tuple(variables or getattr(region, "variables", ()))
        if not coords:
            return None
        value = _parameter_radial_value(sp.sympify(expr), region, coords, assumptions)
        if value is not None:
            record_strategy("weighted_radial")
        return value
    value = _simplify_preserving_sums(
        _refine_with_assumptions(_evaluate(*data), _normalize_assumptions(assumptions))
    )
    record_coverage(
        "absolute_convergence",
        True,
        "weighted_radial",
        basis="absolute polynomial moments on a finite radial domain",
    )
    record_strategy("weighted_radial")
    return value
