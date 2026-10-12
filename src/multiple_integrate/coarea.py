"""Checked scalar coarea reductions with separate geometry and existence evidence."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from functools import cmp_to_key, lru_cache

import sympy as sp
from sympy.core.relational import Relational

from .adapters.semialg import implies
from .assumptions import (
    _ask,
    _condition_to_predicate,
    _normalize_assumptions,
    _refine_with_assumptions,
)
from .boundary_reduction import _Uncertain
from .cell_proofs import native_predicate_true, polynomial_sign
from .cell_proofs import nonnegative as prove_nonnegative
from .fast_rules import _polynomial
from .policy import _normalize_policy, _with_policy, check_semialg_budget
from .results import (
    ConvergenceEvidence,
    CoverageEvidence,
    IntegrationResult,
    IntegrationStatus,
    UnevaluatedRegionIntegral,
    UnresolvedIntegration,
)

_REAL_COORDINATE_FACTS = sp.Symbol("coarea_coordinate", real=True).assumptions0


def _plain_coordinates(variables):
    return all(
        all(_REAL_COORDINATE_FACTS.get(k) == value for k, value in v.assumptions0.items())
        for v in variables
    )


__all__ = [
    "CoareaChart",
    "CoareaPiece",
    "CoareaReduction",
    "reduce_coarea",
    "integrate_coarea",
    "integrate_coarea_cells",
]


@dataclass(frozen=True)
class CoareaChart:
    """Candidate rational chart X(t,u), its inverse, and inner-first ranges.

    ``coordinates[0]`` is the level coordinate. Forward entries follow the source
    variable order; inverse entries follow coordinate order. Caller assertions of
    verification are not accepted: inverse, rank, coverage and overlap are checked.
    """

    coordinates: tuple
    forward: tuple
    inverse: tuple
    ranges: tuple
    chart_exclusions: sp.Expr = sp.false
    source_exclusions: sp.Expr = sp.false
    source_domain: sp.Expr = sp.true
    kind: str = "rational"


@dataclass(frozen=True)
class CoareaPiece:
    """A proved fiber integral followed by its outer level integral."""

    density: sp.Expr
    ranges: tuple
    chart: CoareaChart | None
    jacobian: sp.Expr
    fiber_metric: sp.Expr
    source_image: sp.Expr
    geometry_certificate: object = None
    absolute_majorant: sp.Expr | None = None

    @property
    def integral(self):
        return sp.Integral(self.density, *self.ranges)


@dataclass(frozen=True)
class CoareaReduction:
    """Certified geometric reduction; completeness does not establish convergence."""

    integrand: sp.Expr
    level_map: sp.Expr
    variables: tuple
    region: sp.Expr
    pieces: tuple[CoareaPiece, ...]
    assumptions: object
    coverage: tuple[CoverageEvidence, ...]
    policy: object
    method: str

    @property
    def integral(self):
        return sp.Add(*(piece.integral for piece in self.pieces))

    def evaluate(self):
        """Evaluate children; transfer values only with L1 or Tonelli evidence."""
        from .api.integration import integrate_result

        children = tuple(
            integrate_result(p.density, *p.ranges, assumptions=self.assumptions, policy=self.policy)
            for p in self.pieces
        )
        majorants = tuple(
            integrate_result(
                p.absolute_majorant, *p.ranges, assumptions=self.assumptions, policy=self.policy
            )
            if p.absolute_majorant is not None
            else None
            for p in self.pieces
        )
        absolute = all(
            (
                (
                    m.complete
                    and m.status is IntegrationStatus.EXACT
                    and m.absolute_convergent is not False
                )
                if m is not None
                else c.absolute_convergent is True
                or c.absolute_convergent is None
                and _piece_absolute(p, self.assumptions, self.policy)
            )
            for p, c, m in zip(self.pieces, children, majorants, strict=True)
        )
        weighted = any(p.absolute_majorant is not None for p in self.pieces)
        nonnegative = (
            all(_fiber_nonnegative(p, self.assumptions, 1) for p in self.pieces)
            if not weighted
            else self.integrand.is_nonnegative is True
            or all(_tight_majorant(p, 1) for p in self.pieces)
        )
        positive_divergence = nonnegative and any(
            c.status is IntegrationStatus.DIVERGENT and c.value == sp.oo for c in children
        )
        nonpositive = (
            all(_fiber_nonnegative(p, self.assumptions, -1) for p in self.pieces)
            if not weighted
            else self.integrand.is_nonpositive is True
            or all(_tight_majorant(p, -1) for p in self.pieces)
        )
        negative_divergence = nonpositive and any(
            c.status is IntegrationStatus.DIVERGENT and c.value == -sp.oo for c in children
        )
        divergent = positive_divergence or negative_divergence
        finite_children = all(c.complete and c.status is IntegrationStatus.EXACT for c in children)
        if (
            (nonnegative or nonpositive)
            and finite_children
            and all(c.absolute_convergent is not False for c in children)
        ):
            absolute = True  # Finite same-sign iterated values imply L1 by Tonelli.
        finite = absolute and finite_children
        status = (
            IntegrationStatus.EXACT
            if finite
            else IntegrationStatus.DIVERGENT
            if divergent
            else IntegrationStatus.UNKNOWN
        )
        value = (
            sp.Add(*(c.value for c in children))
            if finite
            else (sp.oo if positive_divergence else -sp.oo)
            if divergent
            else (UnevaluatedRegionIntegral(self.integrand, self.region, self.variables))
        )
        established = True if absolute else False if divergent else None
        return IntegrationResult(
            sp.factor(value) if finite else value,
            status,
            self.method,
            convergent=established,
            exact=finite,
            complete=finite or divergent,
            assumptions=_normalize_assumptions(self.assumptions).predicate_expr,
            coverage=self.coverage
            + (
                (
                    CoverageEvidence(
                        "absolute_convergence",
                        True,
                        self.method,
                        (
                            (
                                "basis",
                                "child L1 evidence, compact-chart or separable endpoint-order comparison, or finite Tonelli reduction",
                            ),
                        ),
                    ),
                )
                if absolute
                else ()
            ),
            convergence_evidence=tuple(
                ConvergenceEvidence(kind, established, self.method)
                for kind in ("ordinary", "absolute")
            ),
            limitations=()
            if finite or divergent
            else (
                "Geometric coverage is proved; child existence/evaluation is unresolved. "
                "Conditional iterated values are not promoted through co-area.",
            ),
            diagnostics={
                "coarea_reduction": self,
                "fiber_results": children,
                "absolute_majorant_results": majorants,
            },
            _convergence_policy=self.policy,
        )


def _sign(expr, context):
    numerator, denominator = sp.fraction(sp.cancel(expr))
    signs = tuple(polynomial_sign(term, context) for term in (numerator, denominator))
    if None in signs or signs[1] == 0:
        raise _Uncertain
    return signs[0] * signs[1]


def _tight_majorant(piece, sign):
    """Equality in the absolute moment bound proves a one-sign fiber a.e.

    Each regular spherical fiber has a finite polynomial amplitude. Since the
    constructed bound dominates its absolute integral, equality with its signed
    moment implies that the negative (or positive) part has zero fiber measure.
    This avoids nonlinear source-domain sign queries.
    """
    return (
        piece.absolute_majorant is not None
        and sp.simplify(piece.absolute_majorant - sign * piece.density) == 0
    )


def _fiber_nonnegative(piece, assumptions, sign):
    return prove_nonnegative(
        sign * piece.density,
        _normalize_assumptions((*assumptions, _range_domain(piece.ranges, interior=True))),
    )


def _piece_absolute(piece, assumptions, policy):
    if _compact_piece(piece, assumptions, policy) or _endpoint_product_absolute(
        piece, assumptions, policy
    ):
        return True
    if len(piece.ranges) == 1:
        v, lo, hi = piece.ranges[0]
        if lo == 0 and hi == 1 and piece.density.has(sp.log):
            from .convergence_comparison import affine_laurent_log_absolute
            from .convergence_evidence import compact_continuity

            cut = sp.Rational(1, 2)
            return affine_laurent_log_absolute(
                piece.density, (v, lo, cut), assumptions
            ) is True and compact_continuity(
                piece.density, ((v, cut, hi),), assumptions=assumptions, policy=policy
            )
    return False


def _endpoint_product_absolute(piece, assumptions, policy):
    """L1 product comparison on independent ranges, without evaluating integrals."""
    from .convergence_comparison import affine_laurent_log_absolute, affine_power_absolute
    from .convergence_evidence import compact_continuity

    if piece.chart is None:
        return False
    variables = tuple(r[0] for r in piece.ranges)
    if any(sp.sympify(b).has(*variables) for r in piece.ranges for b in r[1:]):
        return False
    factors = sp.separatevars(piece.density, symbols=variables, dict=True)
    if (
        factors is None
        or _ask(sp.Q.finite(factors.get("coeff", 1)), _normalize_assumptions(assumptions))
        is not True
    ):
        return False
    for v, lo, hi in piece.ranges:
        f = factors.get(v, sp.S.One)
        if compact_continuity(f, ((v, lo, hi),), assumptions=assumptions, policy=policy):
            continue
        if affine_power_absolute(f, (v, lo, hi), assumptions) is True:
            continue
        if lo == 0 and hi == 1 and f.has(sp.log):
            cut = sp.Rational(1, 2)
            if affine_laurent_log_absolute(
                f, (v, lo, cut), assumptions
            ) is True and compact_continuity(
                f, ((v, cut, hi),), assumptions=assumptions, policy=policy
            ):
                continue
        return False
    return True


def _compact_piece(piece, assumptions, policy):
    """Enclose finite polynomial bounds, then reuse the compact continuity theorem."""
    from .convergence_evidence import compact_continuity

    context = _normalize_assumptions(assumptions)
    bounds = {}
    for v, lo, hi in reversed(piece.ranges):
        if not (lo.has(*bounds) or hi.has(*bounds)):
            if not all(_ask(sp.Q.real(b) & sp.Q.finite(b), context) is True for b in (lo, hi)):
                return False
            bounds[v] = (lo, hi)
            continue
        for b in (lo, hi):
            poly = _polynomial(b, tuple(bounds))
            if poly is None or not all(
                _ask(sp.Q.real(c) & sp.Q.finite(c), context) is True for c in poly.coeffs()
            ):
                return False
        # Polynomial bounds on already compact outer coordinates admit a finite
        # enclosure. A fresh arbitrary positive radius makes continuity checks
        # uniform in that enclosure, without inventing its numerical size.
        radius = sp.Dummy("compact_chart_radius", positive=True)
        bounds[v] = (-radius, radius)
    return compact_continuity(
        piece.density,
        tuple((v, lo, hi) for v, (lo, hi) in bounds.items()),
        assumptions=assumptions,
        policy=policy,
    )


def _clear_rational(predicate, context):
    if isinstance(predicate, Relational):
        numerator, denominator = sp.fraction(sp.cancel(predicate.lhs - predicate.rhs))
        sign = polynomial_sign(denominator, context)
        if sign not in (-1, 1):
            return predicate
        return predicate.func(sign * numerator, 0)
    if predicate.func in (sp.And, sp.Or, sp.Not):
        return predicate.func(*(_clear_rational(arg, context) for arg in predicate.args))
    return predicate


def _holds(premise, conclusion, variables, context, policy):
    if conclusion is sp.true or conclusion == sp.true:
        return True
    if sp.simplify(sp.And(*context.original, premise)) is sp.false:
        return True
    scoped = _normalize_assumptions((*context.original, premise))
    conclusion = _clear_rational(conclusion, scoped)
    if native_predicate_true(_condition_to_predicate(conclusion), scoped):
        return True
    query = _clear_rational(sp.And(*context.original, premise), scoped)
    all_variables = tuple(
        dict.fromkeys((*variables, *sorted(query.free_symbols | conclusion.free_symbols, key=str)))
    )
    allowed, _ = check_semialg_budget(sp.And(query, conclusion), all_variables, policy)
    return allowed and implies(query, conclusion, all_variables) is True


def _range_domain(ranges, *, interior=False):
    return sp.And(
        *(sp.And(v > lo, v < hi) if interior else sp.And(v >= lo, v <= hi) for v, lo, hi in ranges)
    )


def _ranges(ranges, coordinates, context):
    ranges = tuple(tuple(map(sp.sympify, r)) for r in ranges)
    if tuple(r[0] for r in ranges) != tuple(reversed(coordinates)):
        raise ValueError("Ranges must be inner-first, with the level cordinate last")
    for i, (_, lo, hi) in enumerate(ranges):
        if (lo.free_symbols | hi.free_symbols) & {r[0] for r in ranges[: i + 1]}:
            raise ValueError("A bound can depend only on outer coordinates")
        outer = _normalize_assumptions((*context.original, _range_domain(ranges[i + 1 :])))
        if any(
            b not in (-sp.oo, sp.oo) and _ask(sp.Q.real(b), outer) is not True for b in (lo, hi)
        ):
            raise UnresolvedIntegration("Range bounds must be proved real")
        interior = _normalize_assumptions(
            (*context.original, _range_domain(ranges[i + 1 :], interior=True))
        )
        if not _holds(sp.true, hi >= lo, coordinates, outer, _normalize_policy(None)) or not _holds(
            sp.true, hi > lo, coordinates, interior, _normalize_policy(None)
        ):
            raise UnresolvedIntegration("Every chart interval must have proved positive width")
    return ranges


def _verify_chart(chart, f, g, variables, context, policy):
    from .adapters.semialg import chart_geometry

    certificate = chart_geometry(chart, g, variables, assumptions=context.original, policy=policy)
    scoped = _normalize_assumptions((*context.original, certificate.domain))
    measure = _refine_with_assumptions(sp.Abs(certificate.determinant), scoped)
    return CoareaPiece(
        f.subs(dict(zip(variables, chart.forward, strict=True)), simultaneous=True) * measure,
        certificate.chart.ranges,
        chart,
        sp.sqrt(certificate.map_jacobian_squared),
        sp.sqrt(certificate.fiber_metric_squared),
        certificate.image,
        certificate,
    )


def _affine_charts(g, variables, ranges, context):
    if isinstance(g, sp.Tuple):
        return _affine_vector_charts(g, variables, ranges, context)
    poly = _polynomial(g, variables)
    if poly is None or poly.total_degree() != 1:
        raise UnresolvedIntegration("Automatic affine coarea requires a nonconstant affine map")
    rates = tuple(poly.coeff_monomial(v) for v in variables)
    c = poly.coeff_monomial(1)
    if not all(_ask(sp.Q.real(a) & sp.Q.finite(a), context) is True for a in (*rates, c)):
        raise UnresolvedIntegration("Affine map coefficients must be finite and real")
    active = [i for i, a in enumerate(rates) if a != 0]
    index = next((i for i in active if _ask(sp.Q.nonzero(rates[i]), context) is True), None)
    if index is None:
        raise UnresolvedIntegration("A nonzero affine rate is required")
    source = {r[0]: r[1:] for r in ranges}
    if set(source) != set(variables) or any(
        sp.sympify(b).has(*variables) for r in ranges for b in r[1:]
    ):
        raise ValueError("Automatic affine source ranges must be independent")
    t = sp.Dummy("level", real=True)
    others = [i for i in range(len(variables)) if i != index]
    fibers = tuple(sp.Dummy("fiber", real=True) for _ in others)
    coords = (t, *fibers)
    inverse = (g, *(variables[i] for i in others))
    forward = [None] * len(variables)
    for i, u in zip(others, fibers, strict=True):
        forward[i] = u
    rate = rates[index]
    forward[index] = (t - c - sum(rates[i] * forward[i] for i in others)) / rate
    lo, hi = map(sp.sympify, source[variables[index]])
    if len(active) == 1:
        sign = _sign(rate, context)
        endpoints = (rate * lo + c, rate * hi + c) if sign > 0 else (rate * hi + c, rate * lo + c)
        limits = tuple(
            (u, *source[variables[i]]) for i, u in reversed(tuple(zip(others, fibers, strict=True)))
        )
        return (CoareaChart(coords, tuple(forward), inverse, (*limits, (t, *endpoints))),)
    if len(variables) != 2 or any(b in (-sp.oo, sp.oo) for r in ranges for b in r[1:]):
        raise UnresolvedIntegration("Coupled affine maps currently require a finite planar box")
    j = others[0]
    u = fibers[0]
    other_rate = rates[j]
    yl, yh = map(sp.sympify, source[variables[j]])
    asgn, bsgn = _sign(rate, context), _sign(other_rate, context)
    al, ah = (rate * lo, rate * hi) if asgn > 0 else (rate * hi, rate * lo)
    low = (t - c - ah) / other_rate if bsgn > 0 else (t - c - al) / other_rate
    high = (t - c - al) / other_rate if bsgn > 0 else (t - c - ah) / other_rate
    cuts = sorted(
        set(c + rate * X + other_rate * Y for X in (lo, hi) for Y in (yl, yh)),
        key=cmp_to_key(lambda a, b: _sign(a - b, context)),
    )
    charts = []
    for lower, upper in zip(cuts, cuts[1:], strict=False):
        if _sign(upper - lower, context) == 0:
            continue
        midpoint = (lower + upper) / 2
        ul = low if _sign(low.subs(t, midpoint) - yl, context) > 0 else yl
        uh = high if _sign(yh - high.subs(t, midpoint), context) > 0 else yh
        charts.append(
            CoareaChart(coords, tuple(forward), inverse, ((u, ul, uh), (t, lower, upper)))
        )
    return tuple(charts)


def _affine_vector_charts(maps, variables, ranges, context):
    """Independent affine projections; coupled maps use supplied verified charts."""
    if not 1 <= len(maps) < len(variables):
        raise ValueError("Level dimension must be smaller than source dimension")
    source = {r[0]: r[1:] for r in ranges}
    if set(source) != set(variables) or any(
        sp.sympify(b).has(*variables) for r in ranges for b in r[1:]
    ):
        raise ValueError("Automatic affine source ranges must be independent")
    pivots, rates, constants = [], [], []
    for g in maps:
        poly = _polynomial(g, variables)
        if poly is None or poly.total_degree() != 1:
            raise UnresolvedIntegration("Vector automatic maps must be affine projections")
        active = [(v, poly.coeff_monomial(v)) for v in variables if poly.coeff_monomial(v) != 0]
        if len(active) != 1 or active[0][0] in pivots:
            raise UnresolvedIntegration("Coupled vector affine maps require supplied charts")
        v, rate = active[0]
        _sign(rate, context)
        constant = poly.coeff_monomial(1)
        if any(_ask(sp.Q.real(c) & sp.Q.finite(c), context) is not True for c in (rate, constant)):
            raise UnresolvedIntegration("Affine rates must be finite and real")
        pivots.append(v)
        rates.append(rate)
        constants.append(constant)
    levels = tuple(sp.Dummy("level", real=True) for _ in maps)
    free = tuple(v for v in variables if v not in pivots)
    fibers = tuple(sp.Dummy("fiber", real=True) for _ in free)
    coords = (*levels, *fibers)
    forward = dict(zip(free, fibers, strict=True))
    bounds = dict(zip(fibers, (source[v] for v in free), strict=True))
    for v, rate, c, t in zip(pivots, rates, constants, levels, strict=True):
        forward[v] = (t - c) / rate
        lo, hi = map(sp.sympify, source[v])
        ends = (rate * lo + c, rate * hi + c)
        bounds[t] = ends if _sign(rate, context) > 0 else tuple(reversed(ends))
    return (
        CoareaChart(
            coords,
            tuple(forward[v] for v in variables),
            (*maps, *free),
            tuple((v, *bounds[v]) for v in reversed(coords)),
        ),
    )


def _planar_angular_piece(f, g, variables, levels, context):
    """Finite Fourier polynomials and one uniformly positive affine cosine denominator."""
    from .quadratic_radial import positive_quadratic

    if len(variables) != 2 or sp.count_ops(f) > 200:
        raise UnresolvedIntegration("Selected angular families require a bounded planar density")
    q = g.base if g.is_Pow and g.exp == sp.Rational(1, 2) else g
    data = positive_quadratic(q, variables, context)
    if data is None or data[0] != sp.eye(2) or data[2] != 0:
        raise UnresolvedIntegration("Angular levels require Euclidean circles")
    _, center, _ = data
    x, y = variables
    cx, cy = center
    t = sp.Dummy("level", positive=True)
    theta = sp.Dummy("angle", real=True)
    radius = q != g
    r = t if radius else sp.sqrt(t)
    angle = sp.atan2(y - cy, x - cx)
    transformed = (
        f.xreplace({angle: theta})
        .subs(q, r * r)
        .subs({x: cx + r * sp.cos(theta), y: cy + r * sp.sin(theta)}, simultaneous=True)
    )
    if transformed.has(sp.atan2):
        raise UnresolvedIntegration("Unsupported angular branch")
    for atom in transformed.atoms(sp.sin, sp.cos):
        frequency = sp.simplify(atom.args[0] / theta)
        if not frequency.is_Integer or abs(frequency) > 8:
            raise UnresolvedIntegration("Angular frequencies must be bounded integers")
    transformed = sp.expand_trig(transformed)
    C, S = sp.Dummy("cosine", real=True), sp.Dummy("sine", real=True)
    transformed = transformed.xreplace({sp.cos(theta): C, sp.sin(theta): S})
    if transformed.has(theta):
        raise UnresolvedIntegration("Unsupported angular expression")
    numerator, denominator = sp.fraction(sp.cancel(transformed))
    pn, pd = _polynomial(numerator, (C, S)), _polynomial(denominator, (C, S))
    if pn is None or pd is None or pd.total_degree() > 1 or pd.coeff_monomial(S) != 0:
        raise UnresolvedIntegration("Angular denominator must be affine in cosine")
    A, B = pd.coeff_monomial(1), pd.coeff_monomial(C)
    content = sp.gcd(A, B)
    if content.has(t):
        if (
            _ask(sp.Q.nonzero(content), _normalize_assumptions((*context.original, t > 0)))
            is not True
        ):
            raise UnresolvedIntegration("Radial denominator factor must be nonzero")
        pn = _polynomial(pn.as_expr() / content, (C, S))
        A, B = sp.cancel(A / content), sp.cancel(B / content)
    # Coefficeints may carry radial profiles, but denominator bounds must be uniform in the level.
    if (A.has(t) or B.has(t)) and B != 0:
        raise UnresolvedIntegration("Angular denominator must be independent of the level")
    if B != 0 and _ask(sp.Q.positive(-A - sp.Abs(B)), context) is True:
        A, B, pn = -A, -B, _polynomial(-pn.as_expr(), (C, S))
    if B != 0 and _ask(sp.Q.zero(B), context) is True:
        B = sp.S.Zero
    elif B != 0 and _ask(sp.Q.nonzero(B), context) is not True:
        raise UnresolvedIntegration("Split the zero angular coefficient fiber explicitly")
    lower = A - sp.Abs(B)
    if B != 0 and (
        _ask(sp.Q.positive(lower), context) is not True
        or any(_ask(sp.Q.real(v) & sp.Q.finite(v), context) is not True for v in (A, B))
    ):
        raise UnresolvedIntegration("Prove A > |B| for the angular denominator")
    regular = _normalize_assumptions((*context.original, t > 0))
    if B == 0 and _ask(sp.Q.nonzero(A), regular) is not True:
        raise UnresolvedIntegration("Angular scalar denominator must be nonzero")

    def sphere_moment(i, j):
        return (
            2
            * sp.gamma(sp.Rational(i + 1, 2))
            * sp.gamma(sp.Rational(j + 1, 2))
            / sp.gamma(sp.Rational(i + j + 2, 2))
        )

    cosine_moments = {0: 2 * sp.pi / sp.sqrt(A * A - B * B)} if B != 0 else {}

    def reciprocal_moment(i):
        if B == 0:
            return (sphere_moment(i, 0) if i % 2 == 0 else sp.S.Zero) / A
        for k in range(1, i + 1):
            if k not in cosine_moments:
                previous = sphere_moment(k - 1, 0) if (k - 1) % 2 == 0 else sp.S.Zero
                cosine_moments[k] = sp.cancel((previous - A * cosine_moments[k - 1]) / B)
        return cosine_moments[i]

    angular = sp.S.Zero
    bound = sp.S.Zero
    for (i, j), coefficient in pn.terms():
        if _ask(sp.Q.real(coefficient) & sp.Q.finite(coefficient), regular) is not True:
            raise UnresolvedIntegration("Angular coefficients must be real on each regular level")
        bound += sp.Abs(coefficient) * sphere_moment(i, j) / (lower if B != 0 else sp.Abs(A))
        if j % 2 == 0:
            angular += coefficient * sum(
                (-1) ** k * sp.binomial(j // 2, k) * reciprocal_moment(i + 2 * k)
                for k in range(j // 2 + 1)
            )
    jac = sp.S.One if radius else 2 * sp.sqrt(t)
    weight = r / jac
    source = sp.And(g >= levels[0], g <= levels[1])
    return CoareaPiece(
        sp.simplify(angular * weight),
        ((t, *levels),),
        None,
        jac,
        2 * sp.pi * r,
        source,
        absolute_majorant=_refine_with_assumptions(sp.simplify(bound * weight), regular),
    ), source


def _radial_piece(f, g, variables, level_bounds, context):
    from .quadratic_radial import positive_quadratic

    squared = g.base if g.is_Pow and g.exp == sp.Rational(1, 2) else g
    radius = squared != g
    if _polynomial(squared, variables) is None:
        raise UnresolvedIntegration("Radial map degree/expansion budget exceeded")
    data = positive_quadratic(squared, variables, context)
    if data is None or data[0] != sp.eye(len(variables)) or data[2] != 0:
        raise UnresolvedIntegration("Radial coarea requires Euclidean radius or squared radius")
    _, center, _ = data
    lower, upper = map(sp.sympify, level_bounds)
    if not native_predicate_true(lower >= 0, context) or not native_predicate_true(
        upper > lower, context
    ):
        raise UnresolvedIntegration("Prove 0 <= lower < upper for radial levels")
    if (
        lower.is_real is not True
        or upper not in (sp.oo,)
        and _ask(sp.Q.real(upper) & sp.Q.finite(upper), context) is not True
    ):
        raise UnresolvedIntegration("Radial endpoints must be real and finite, except +infinity")
    t = sp.Dummy("level", positive=True)
    point = dict(zip(variables, center, strict=True))
    point[variables[0]] += t if radius else sp.sqrt(t)
    profile = sp.simplify(f.subs(point, simultaneous=True))
    reconstructed = profile.subs(t, g)
    n = len(variables)
    sphere = 2 * sp.pi ** (sp.Rational(n, 2)) / sp.gamma(sp.Rational(n, 2))
    metric = sphere * t ** (n - 1) if radius else sphere * t ** (sp.Rational(n - 1, 2))
    jac = sp.S.One if radius else 2 * sp.sqrt(t)
    weight = metric / jac
    majorant = None
    if sp.simplify(f - reconstructed) == 0:
        density = profile * weight
    else:
        polynomial = sp.S.One
        radial = sp.S.One
        for factor in sp.Mul.make_args(f):
            if _polynomial(factor, variables) is not None:
                polynomial *= factor
            else:
                candidate = sp.simplify(factor.subs(point, simultaneous=True))
                if sp.simplify(factor - candidate.subs(t, g)) != 0:
                    return _planar_angular_piece(f, g, variables, level_bounds, context)
                radial *= candidate
        shifted = _polynomial(
            polynomial.subs(
                {v: v + c for v, c in zip(variables, center, strict=True)}, simultaneous=True
            ),
            variables,
        )
        if shifted is None:
            raise UnresolvedIntegration("Angular polynomial expansion/degree budget exceeded")
        angular = sp.S.Zero
        bound = sp.S.Zero
        r = t if radius else sp.sqrt(t)
        for alpha, coefficient in shifted.terms():
            if _ask(sp.Q.real(coefficient) & sp.Q.finite(coefficient), context) is not True:
                raise UnresolvedIntegration("Angular coefficients must be finite and real")
            absolute_moment = (
                2
                * sp.prod(sp.gamma(sp.Rational(k + 1, 2)) for k in alpha)
                / sp.gamma(sp.Rational(n + sum(alpha), 2))
            )
            factor = r ** sum(alpha) * metric / sphere / jac * absolute_moment
            bound += sp.Abs(coefficient) * factor
            if not any(k % 2 for k in alpha):
                angular += coefficient * factor
        density = radial * angular
        majorant = sp.Abs(radial) * bound
        majorant = _refine_with_assumptions(
            majorant, _normalize_assumptions((*context.original, t > 0))
        )
    region = sp.And(g >= lower, g <= upper)
    return CoareaPiece(
        sp.factor(density),
        ((t, lower, upper),),
        None,
        jac,
        metric,
        region,
        absolute_majorant=sp.factor(majorant) if majorant is not None else None,
    ), region


@_with_policy
def reduce_coarea(
    f,
    *,
    level_map,
    variables,
    ranges=None,
    region=None,
    charts=None,
    level_bounds=None,
    assumptions=None,
    policy=None,
):
    """Construct a certified scalar reduction, without claiming integral existence.

    Automatic affine maps use independent source ranges (coupled maps: finite 2D boxes).
    Radius/squared-radius maps use level_bounds and radial densities.
    Supplied rational charts require an explicit Boolean source region.
    """
    variables = tuple(variables)
    f = sp.sympify(f)
    g = (
        sp.Tuple(*level_map)
        if isinstance(level_map, (tuple, list, sp.Tuple))
        else sp.sympify(level_map)
    )
    if isinstance(g, sp.Tuple) and not 1 <= len(g) < len(variables):
        raise ValueError("Level dimension must be smaller than source dimension")
    if (
        len(variables) < 2
        or len(set(variables)) != len(variables)
        or any(not isinstance(v, sp.Symbol) or v.is_real is not True for v in variables)
    ):
        raise ValueError("Coarea requires at least two distinct real source coordinates")
    if not _plain_coordinates(variables):
        raise UnresolvedIntegration(
            "Use plain real source coordinates; put coordinate restrictions in the domain"
        )
    context = _normalize_assumptions(assumptions)
    if any(condition.has(*variables) for condition in context.original):
        raise ValueError(
            "Coarea assumptions must concern parameters; put coordinate restrictions in the source region"
        )
    policy = _normalize_policy(policy)
    if not policy.allow_expensive_exact and len(variables) > policy.semialg.max_variables:
        raise UnresolvedIntegration("Coarea dimension budget exceeded")
    try:
        if level_bounds is not None:
            if isinstance(g, sp.Tuple):
                raise ValueError("Radial level bounds require a scalar map")
            if ranges is not None or charts is not None or region is not None:
                raise ValueError("Radial level_bounds specify the source shell; omit other domains")
            lo, hi = map(sp.sympify, level_bounds)
            if sp.simplify(hi - lo) == 0 and native_predicate_true(lo >= 0, context):
                from .quadratic_radial import positive_quadratic

                q = g.base if g.is_Pow and g.exp == sp.Rational(1, 2) else g
                data = positive_quadratic(q, variables, context)
                if data is None or data[0] != sp.eye(len(variables)) or data[2] != 0:
                    raise UnresolvedIntegration(
                        "Collapsed radial source requires a proved Euclidean level"
                    )
                from .adapters.semialg import null_geometry

                null = null_geometry(
                    sp.Eq(q, lo**2 if q != g else lo),
                    variables,
                    assumptions=context.original,
                    policy=policy,
                )
                region = sp.Eq(g, lo)
                level = sp.Dummy("null_level", real=True)
                piece = CoareaPiece(
                    sp.S.Zero, ((level, 0, 1),), None, sp.S.One, sp.S.Zero, region, null
                )
            else:
                piece, region = _radial_piece(f, g, variables, level_bounds, context)
            pieces = (piece,)
            method = "coarea_radial"
            basis = "Euclidean spheres; the center is an ambient null set"
        else:
            generated = charts is None
            if generated:
                if ranges is None or region is not None:
                    raise ValueError("Automatic affine maps require ranges alone")
                ranges = _ranges(ranges, tuple(reversed(tuple(r[0] for r in ranges))), context)
                region = _range_domain(ranges)
                charts = _affine_charts(g, variables, ranges, context)
            elif region is None or ranges is not None:
                raise ValueError("Supplied charts require region alone")
            region = sp.sympify(region)
            charts = tuple(charts)
            if (
                not charts
                or not policy.allow_expensive_exact
                and len(charts) > policy.semialg.max_pieces
            ):
                raise UnresolvedIntegration("Coarea chart count is empty or exceeds budget")
            allowed, reason = check_semialg_budget(region, variables, policy)
            if not allowed:
                raise UnresolvedIntegration(reason)
            external = (
                f.free_symbols
                | g.free_symbols
                | region.free_symbols
                | context.predicate_expr.free_symbols
            ) - set(variables)
            from .adapters.semialg import geometry_chart_type

            geometry_type = geometry_chart_type()
            chart_types = (CoareaChart,) if geometry_type is None else (CoareaChart, geometry_type)
            if any(not isinstance(c, chart_types) for c in charts):
                raise ValueError("charts must contain CoareaChart candidates")
            if any(set(c.coordinates) & external for c in charts):
                raise ValueError("Chart coordinates cannot capture external parameters")
            pieces = tuple(_verify_chart(c, f, g, variables, context, policy) for c in charts)
            if not generated:
                from .adapters.semialg import chart_cover

                cover = chart_cover(
                    region,
                    tuple(p.geometry_certificate for p in pieces),
                    assumptions=context.original,
                    policy=policy,
                )
            else:
                cover = None
            method = "coarea_affine" if generated else "coarea_charts"
            basis = (
                "affine box partition"
                if generated
                else "Semialg chart and null-set certificates, coverage and disjoint interiors"
            )
    except _Uncertain as exc:
        raise UnresolvedIntegration("A whole-cell affine ordering/sign is unproved") from exc
    coverage = (
        CoverageEvidence(
            "integration_domain",
            True,
            method,
            (
                ("source_region", region),
                ("level_map", g),
                ("pieces", pieces),
                ("basis", basis),
                ("geometry_cover", cover if method != "coarea_radial" else None),
            ),
        ),
        CoverageEvidence(
            "coarea_fibers",
            True,
            method,
            (
                (
                    "basis",
                    "induced fiber metric divided by map Jacobian; critical-set treatment explicit",
                ),
            ),
        ),
    )
    return CoareaReduction(
        f, g, variables, region, pieces, tuple(context.original), coverage, policy, method
    )


def integrate_coarea(f, **request):
    """Evaluate a proved reduction, retaining unknown outcomes on failed proofs."""
    try:
        return reduce_coarea(f, **request).evaluate()
    except UnresolvedIntegration as exc:
        region = request.get("region")
        if region is None:
            region = (
                _range_domain(request["ranges"])
                if request.get("ranges")
                else sp.And(
                    request["level_map"] >= request["level_bounds"][0],
                    request["level_map"] <= request["level_bounds"][1],
                )
            )
        return IntegrationResult(
            UnevaluatedRegionIntegral(f, region, tuple(request["variables"])),
            IntegrationStatus.UNKNOWN,
            "coarea",
            complete=False,
            limitations=(str(exc),),
        )


def _cheap_exponential_profile(expr, g, variables, assumptions, center=None):
    """Only a polynomial amplitude times exp(c-a*g**k) enters automatically."""
    context = _normalize_assumptions(assumptions)
    amplitude = sp.S.One
    nonlinear = []
    for factor in sp.Mul.make_args(expr):
        if _polynomial(factor, variables) is not None:
            amplitude *= factor
        else:
            nonlinear.append(factor)
    if (
        len(nonlinear) != 1
        or nonlinear[0].func != sp.exp
        or _polynomial(amplitude, variables) is None
    ):
        return False
    level = sp.Dummy("coarea_level", positive=True)
    if center is not None:
        point = dict(zip(variables, center, strict=True))
        point[variables[0]] += sp.sqrt(level)
    else:
        poly = _polynomial(g, variables)
        if poly is None or poly.total_degree() != 1:
            return False
        index = next(
            (
                i
                for i, v in enumerate(variables)
                if _ask(sp.Q.nonzero(poly.coeff_monomial(v)), context) is True
            ),
            None,
        )
        if index is None:
            return False
        point = dict.fromkeys(variables, sp.S.Zero)
        point[variables[index]] = (level - poly.coeff_monomial(1)) / poly.coeff_monomial(
            variables[index]
        )
    phase = nonlinear[0].args[0]
    profile = sp.expand(phase.subs(point, simultaneous=True))
    if sp.cancel(phase - profile.subs(level, g)) != 0:
        return False
    polynomial = _polynomial(profile, (level,))
    if polynomial is None:
        return False
    terms = [(powers[0], c) for powers, c in polynomial.terms() if powers[0] != 0]
    if len(terms) != 1 or not 1 <= terms[0][0] <= 8:
        return False
    return _ask(sp.Q.negative(terms[0][1]), context) is True and all(
        _ask(sp.Q.real(c) & sp.Q.finite(c), context) is True for c in polynomial.coeffs()
    )


def _cheap_radial_profile(expr, g, variables, assumptions, center, levels):
    """Bounded candidate gate: no integration or general geometry query here."""
    if _cheap_exponential_profile(expr, g, variables, assumptions, center=center):
        return True
    context = _normalize_assumptions(assumptions)
    lo, hi = levels
    compact = all(_ask(sp.Q.finite(b), context) is True for b in (lo, hi))
    polynomial = _polynomial(expr, variables)
    if polynomial is not None:
        return compact and all(
            _ask(sp.Q.real(c) & sp.Q.finite(c), context) is True for c in polynomial.coeffs()
        )
    level = sp.Dummy("coarea_level", positive=True)
    point = dict(zip(variables, center, strict=True))
    point[variables[0]] += sp.sqrt(level)
    amplitude = sp.S.One
    profile = sp.S.One
    for factor in sp.Mul.make_args(expr):
        if _polynomial(factor, variables) is not None:
            amplitude *= factor
        else:
            candidate = sp.simplify(factor.subs(point, simultaneous=True))
            if sp.simplify(factor - candidate.subs(level, g)) != 0:
                return False
            profile *= candidate
    angular = _polynomial(amplitude, variables)
    if angular is None:
        return False
    if profile.has(sp.Piecewise):
        pieces = profile.atoms(sp.Piecewise)
        return (
            compact
            and len(pieces) == 1
            and len(next(iter(pieces)).args) <= 4
            and all(
                _polynomial(value, (level,)) is not None and not condition.has(*variables)
                for value, condition in next(iter(pieces)).args
            )
        )
    numerator, denominator = sp.fraction(sp.cancel(profile))
    pn, pd = _polynomial(numerator, (level,)), _polynomial(denominator, (level,))
    if pn is None or pd is None:
        return False
    if not all(
        _ask(sp.Q.real(c) & sp.Q.finite(c), context) is True
        for c in (*pn.coeffs(), *pd.coeffs(), *angular.coeffs())
    ):
        return False
    # Positive coefficients and positive constant term exclude all shell poles.
    if _ask(sp.Q.positive(pd.coeff_monomial(1)), context) is not True or not all(
        _ask(sp.Q.nonnegative(c), context) is True for c in pd.coeffs()
    ):
        return False
    return compact or (
        lo.is_nonnegative is True
        and hi == sp.oo
        and len(variables) + angular.total_degree() + 2 * (pn.degree() - pd.degree()) < 0
    )


def _affine_level_profile(expr, g, variables, assumptions):
    """Avoid replacing fast iterated routes for separately dependent amplitudes."""
    poly = _polynomial(g, variables)
    if poly is None or poly.total_degree() != 1:
        return False
    context = _normalize_assumptions(assumptions)
    pivot = next(
        (v for v in variables if _ask(sp.Q.nonzero(poly.coeff_monomial(v)), context) is True), None
    )
    if pivot is None:
        return False
    level = sp.Dummy("affine_profile", real=True)
    point = dict.fromkeys(variables, sp.S.Zero)
    point[pivot] = (level - poly.coeff_monomial(1)) / poly.coeff_monomial(pivot)
    profile = expr.subs(point, simultaneous=True)
    return sp.cancel(expr - profile.subs(level, g)) == 0


def _shifted_gaussian_center(expr, variables, assumptions):
    from .quadratic_radial import positive_quadratic

    atoms = tuple(expr.atoms(sp.exp))
    if len(atoms) != 1:
        return None
    data = positive_quadratic(-atoms[0].args[0], variables, _normalize_assumptions(assumptions))
    if data is None:
        return None
    matrix, center, _ = data
    if matrix != sp.eye(len(variables)) * matrix[0, 0]:
        return None
    return tuple(center)


_AUTOMATIC_DEPTH = contextvars.ContextVar("multiple_integrate_coarea_depth", default=0)


@lru_cache(maxsize=256)
def _automatic_geometry(expr, variables, ranges, radial_data, assumptions, policy):
    """Bounded candidate discovery, unsuccessful recognition is cached as well."""
    if (
        len(variables) < 2
        or len(variables) > policy.semialg.max_variables
        or sp.count_ops(expr) > 200
        or policy.semialg.max_pieces < 1
        or policy.semialg.max_polynomial_degree < 1
    ):
        return None
    if radial_data is not None:
        if policy.semialg.max_polynomial_degree < 2:
            return None
        center, lo, hi = radial_data
        g = sum((v - c) ** 2 for v, c in zip(variables, center, strict=True))
        return (
            (g, (lo, hi))
            if _cheap_radial_profile(expr, g, variables, assumptions, center, (lo, hi))
            else None
        )
    if any(sp.sympify(b).has(*variables) for r in ranges for b in r[1:]):
        return None
    if len(variables) != 2 or len(ranges) != len(variables):
        return None
    candidates = set()
    for atom in expr.atoms(sp.Function, sp.Pow):
        argument = atom.base if atom.is_Pow else atom.args[0] if len(atom.args) == 1 else None
        if argument is None:
            continue
        if atom.func in (sp.exp, sp.sin, sp.cos):
            phase = _polynomial(argument, variables)
            if phase is not None and phase.total_degree() <= 1:
                continue
        if not argument.has(*variables):
            continue
        for node in sp.preorder_traversal(argument):
            if len(node.free_symbols & set(variables)) < 2:
                continue
            poly = _polynomial(node, variables)
            if poly is not None and poly.total_degree() == 1:
                candidates.add(node)
    for g in sorted(candidates, key=sp.default_sort_key):
        if _cheap_exponential_profile(expr, g, variables, assumptions):
            return g, None
        if expr.is_rational_function(*variables) and all(
            sp.sympify(b).is_finite is True for r in ranges for b in r[1:]
        ):
            from .convergence_evidence import compact_continuity

            if _affine_level_profile(expr, g, variables, assumptions) and compact_continuity(
                expr, ranges, assumptions=assumptions, policy=policy
            ):
                return g, None
    return None


def try_automatic_coarea(expr, region, ranges, assumptions, policy=None):
    """Selected cheap coarea routes before general integration fallback."""
    if _AUTOMATIC_DEPTH.get():
        return None
    from .planner import record_convergence, record_coverage
    from .policy import _normalize_policy
    from .region_types import AnnulusRegion, BallRegion, BoxRegion, DiskRegion, SphericalShellRegion

    variables = tuple(region.variables)
    policy = _normalize_policy(policy)
    if (
        not _plain_coordinates(variables)
        or len(variables) > policy.semialg.max_variables
        or sp.count_ops(expr) > 200
    ):
        return None
    radial_data = None
    if isinstance(region, (DiskRegion, BallRegion)):
        radial_data = (tuple(region.center), sp.S.Zero, region.radius**2)
    elif isinstance(region, (AnnulusRegion, SphericalShellRegion)):
        radial_data = (
            tuple(sp.S.Zero for _ in variables),
            region.inner_radius**2,
            region.outer_radius**2,
        )
    elif isinstance(region, BoxRegion) and all(
        lo == -sp.oo and hi == sp.oo for _, lo, hi in ranges
    ):
        center = _shifted_gaussian_center(expr, variables, assumptions)
        radial_data = (center or tuple(sp.S.Zero for _ in variables), sp.S.Zero, sp.oo)
    policy = _normalize_policy(policy)
    context = _normalize_assumptions(assumptions)
    candidate = _automatic_geometry(
        expr, variables, tuple(ranges), radial_data, tuple(context.original), policy
    )
    if candidate is None:
        return None
    g, levels = candidate
    token = _AUTOMATIC_DEPTH.set(1)
    try:
        result = integrate_coarea(
            expr,
            level_map=g,
            variables=variables,
            assumptions=context.original,
            policy=policy,
            **({"level_bounds": levels} if levels is not None else {"ranges": ranges}),
        )
    finally:
        _AUTOMATIC_DEPTH.reset(token)
    if not result.complete:
        return None
    for evidence in result.coverage:
        record_coverage(
            evidence.scope, evidence.complete, evidence.method, **dict(evidence.details)
        )
    for evidence in result.convergence_evidence:
        record_convergence(evidence.kind, evidence.established, evidence.method, evidence.condition)
    return result.method, result.value


def clear_coarea_cache():
    _automatic_geometry.cache_clear()
    from .adapters.semialg import clear_geometry_cache

    clear_geometry_cache()


@_with_policy
def integrate_coarea_cells(f, *, cells, parameters, assumptions=None, policy=None, **request):
    """Co-area on proved caller cells, with exact specialization of equality fibers.

    Rank-zero maps are evaluated as source integrals; a zero Jacobian does not
    imply a zero integral. Failed geometry and coefficient poles stay unknown.
    """
    from dataclasses import replace

    from .adapters.semialg import is_satisfiable
    from .api.integration import integrate_result
    from .parameter_cells import _validate_parameter_cells
    from .results import IntegralCase, ParameterStratum

    parameters = tuple(parameters)
    variables = tuple(request["variables"])
    if request.get("level_bounds") is not None:
        if any(request.get(name) is not None for name in ("ranges", "region", "charts")):
            raise ValueError("Radial level bounds specify the source; omit other domains")
    elif request.get("charts") is None:
        if request.get("ranges") is None or request.get("region") is not None:
            raise ValueError("Automatic affine maps require source ranges alone")
    elif request.get("region") is None or request.get("ranges") is not None:
        raise ValueError("Supplied charts require a source region alone")
    if len(set(parameters)) != len(parameters) or any(
        not isinstance(p, sp.Symbol) or p.is_real is not True for p in parameters
    ):
        raise ValueError("parameters must be distinct real Symbols")
    if set(parameters) & set(variables):
        raise ValueError("parameters cannot be source coordinates")
    parent = sp.And(*_normalize_assumptions(assumptions).original)
    if not parent.free_symbols <= set(parameters):
        raise ValueError("cell assumptions concern declared parameters only")
    policy = _normalize_policy(policy)
    active = _validate_parameter_cells(tuple(cells), parameters, parent, policy)
    cases = []
    children = []
    evidence = []
    coverage = []
    from .adapters.semialg import (
        chart_parameter_geometry_events,
        parameter_geometry_event_candidates,
    )

    events = set()
    event_limitations = []
    event_expressions = []
    for _, lo, hi in request.get("ranges") or ():
        event_expressions.extend((sp.sympify(lo), sp.sympify(hi), sp.sympify(hi) - sp.sympify(lo)))
    if request.get("level_bounds") is not None:
        lo, hi = map(sp.sympify, request["level_bounds"])
        event_expressions.extend((lo, hi, hi - lo))
    maps = (
        request["level_map"]
        if isinstance(request["level_map"], (tuple, list, sp.Tuple))
        else (request["level_map"],)
    )
    for g in maps:
        polynomial = _polynomial(g, variables)
        if polynomial is not None and polynomial.total_degree() <= 1:
            event_expressions.extend(polynomial.coeff_monomial(v) for v in variables)
            if (
                len(variables) == 2
                and request.get("ranges")
                and all(sp.sympify(b).is_finite is True for r in request["ranges"] for b in r[1:])
            ):
                from itertools import combinations, product

                source = {r[0]: r[1:] for r in request["ranges"]}
                if set(source) == set(variables):
                    corners = [
                        sp.sympify(g).subs(
                            dict(zip(variables, point, strict=True)), simultaneous=True
                        )
                        for point in product(*(source[v] for v in variables))
                    ]
                    event_expressions.extend(
                        sp.expand(left - right) for left, right in combinations(corners, 2)
                    )
    event_expressions = tuple(
        e
        for e in event_expressions
        if e.free_symbols <= set(parameters) and e not in (-sp.oo, sp.oo)
    )
    try:
        events.update(
            parameter_geometry_event_candidates(event_expressions, parameters, policy=policy)
        )
    except (ImportError, UnresolvedIntegration) as exc:
        event_limitations.append(str(exc))
    for chart in request.get("charts") or ():
        try:
            events.update(
                chart_parameter_geometry_events(
                    chart, request["level_map"], variables, parameters, policy=policy
                )
            )
        except (ImportError, UnresolvedIntegration) as exc:
            event_limitations.append(str(exc))
    original_g = (
        sp.Tuple(*request["level_map"])
        if isinstance(request["level_map"], (tuple, list, sp.Tuple))
        else sp.sympify(request["level_map"])
    )
    if (
        len(variables) < 2
        or len(set(variables)) != len(variables)
        or any(not isinstance(v, sp.Symbol) or v.is_real is not True for v in variables)
    ):
        raise ValueError("Coarea requires distinct real source coordinates")
    if isinstance(original_g, sp.Tuple) and not 1 <= len(original_g) < len(variables):
        raise ValueError("Level dimension must be smaller than source dimension")
    if isinstance(original_g, sp.Tuple) and request.get("level_bounds") is not None:
        raise ValueError("Radial level bounds require a scalar map")
    original_region = request.get("region")
    if original_region is None:
        original_region = (
            _range_domain(request["ranges"])
            if request.get("ranges")
            else sp.And(
                original_g >= request["level_bounds"][0], original_g <= request["level_bounds"][1]
            )
            if request.get("level_bounds")
            else sp.true
        )
    unknown = UnevaluatedRegionIntegral(f, original_region, variables)
    for cell in active:
        substitutions = dict(cell.substitutions)
        local = dict(request)
        local["level_map"] = original_g.subs(substitutions, simultaneous=True)
        for name in ("region", "level_bounds", "ranges"):
            if local.get(name) is not None:
                local[name] = sp.sympify(local[name]).subs(substitutions, simultaneous=True)
        if local.get("charts") is not None:
            local["charts"] = tuple(
                replace(
                    c,
                    **{
                        name: sp.sympify(getattr(c, name)).subs(substitutions, simultaneous=True)
                        for name in (
                            "forward",
                            "inverse",
                            "ranges",
                            "chart_exclusions",
                            "source_exclusions",
                            "source_domain",
                        )
                    },
                )
                for c in local["charts"]
            )
        density = sp.sympify(f).subs(substitutions, simultaneous=True)
        g = local["level_map"]
        scoped = sp.And(parent, cell.condition)
        invalid = any(
            sp.sympify(e).has(sp.nan, sp.zoo)
            for e in (
                density,
                g,
                local.get("ranges") or (),
                local.get("level_bounds") or (),
                *(getattr(c, "inverse", ()) for c in local.get("charts") or ()),
            )
        )
        maps = tuple(g) if isinstance(g, sp.Tuple) else (g,)
        defined = not any(
            sp.sympify(e).has(sp.nan, sp.zoo)
            for e in (
                density,
                g,
                local["region"] if local.get("region") is not None else sp.true,
                local.get("ranges") or (),
            )
        )
        rank_loss = False
        if substitutions and defined and all(_polynomial(h, variables) is not None for h in maps):
            gradient = sp.Matrix(maps).jacobian(variables)
            rank_loss = sp.simplify((gradient * gradient.T).det()) == 0
            for chart in local.get("charts") or ():
                if any(sp.simplify(hi - lo) == 0 for _, lo, hi in chart.ranges):
                    rank_loss = True
                if any(_polynomial(h, chart.coordinates) is None for h in chart.forward):
                    continue
                matrix = sp.Matrix(chart.forward).jacobian(chart.coordinates)
                if not matrix.has(sp.nan, sp.zoo) and sp.simplify(matrix.det()) == 0:
                    rank_loss = True
        fallback = (
            not local.get("level_bounds") and defined and (not g.has(*variables) or rank_loss)
        )
        if fallback:
            child = integrate_result(
                density,
                *(local.get("ranges") or ()),
                region=local.get("region"),
                variables=variables if local.get("region") is not None else None,
                assumptions=scoped,
                policy=policy,
            )
        elif invalid:
            child = IntegrationResult(
                unknown,
                IntegrationStatus.UNKNOWN,
                "coarea_parameter_pole",
                complete=False,
                limitations=("parameter specialization makes the density or geometry undefined",),
            )
        else:
            child = integrate_coarea(density, assumptions=scoped, policy=policy, **local)
        children.append(child)
        for branch in child.cases or (
            IntegralCase(child.value, sp.true, child.status, child.convergent),
        ):
            cases.append(
                replace(
                    branch,
                    condition=sp.And(
                        cell.condition, branch.condition.subs(substitutions, simultaneous=True)
                    ),
                )
            )
        evidence.extend(
            replace(
                e,
                condition=sp.And(
                    cell.condition, e.condition.subs(substitutions, simultaneous=True)
                ),
            )
            for e in child.convergence_evidence
        )
        coverage.extend(
            CoverageEvidence(
                e.scope, e.complete, e.method, (*e.details, ("parameter_cell", cell.condition))
            )
            for e in child.coverage
        )
    union = sp.Or(*(c.condition for c in active))
    complement = sp.And(parent, sp.Not(union))
    allowed, reason = check_semialg_budget(complement, parameters, policy)
    if not allowed:
        raise UnresolvedIntegration(reason)
    covered = complement is sp.false or is_satisfiable(complement, parameters) is False
    if not covered:
        cases.append(IntegralCase(unknown, sp.Not(union), IntegrationStatus.UNKNOWN, None))
    value = sp.Piecewise(*((c.value, c.condition) for c in cases), (unknown, True), evaluate=False)
    return IntegrationResult(
        value,
        IntegrationStatus.CONDITIONAL,
        "coarea_parameter_cells",
        cases=tuple(cases),
        parameter_strata=tuple(
            ParameterStratum(c.condition, c.value, c.status, c.convergent, parameters)
            for c in cases
        ),
        complete=covered and all(c.complete for c in children),
        exact=covered and all(c.exact for c in children),
        assumptions=parent,
        convergence_evidence=tuple(evidence),
        coverage=(
            CoverageEvidence(
                "parameter_space",
                covered,
                "caller_supplied_coarea_cells",
                (("cells", len(active)),),
            ),
            *coverage,
        ),
        diagnostics={
            "cell_results": tuple(children),
            "geometry_events": tuple(sorted(events, key=sp.default_sort_key)),
            "geometry_event_limitations": tuple(event_limitations),
        },
        _convergence_policy=policy,
        limitations=(() if covered else ("uncovered parameter values retained as unknown",))
        + (
            ()
            if all(c.complete for c in children)
            else ("one or more supplied coarea cells remains unresolved; inspect cell_results",)
        ),
    )
