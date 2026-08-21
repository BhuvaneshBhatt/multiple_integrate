from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp
from sympy.core.relational import Relational

_REGION_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


def _normalize_ranges_input(ranges):
    if isinstance(ranges, Region):
        return ranges
    if isinstance(ranges, tuple) and len(ranges) == 1 and isinstance(ranges[0], Region):
        return ranges[0]
    if isinstance(ranges, tuple):
        return list(ranges)
    return ranges


def _clean_expr(expr):
    expr = sp.sympify(expr)
    try:
        return sp.simplify(expr)
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError):
        return expr


def _split_dependence(
    expr: sp.Expr, radial_vars: tuple[sp.Symbol, ...], angular_vars: tuple[sp.Symbol, ...]
):
    """Split a transformed expression into radial and angular factors when possible."""
    expr = _clean_expr(expr)
    radial_set = set(radial_vars)
    angular_set = set(angular_vars)
    if expr.is_Mul:
        radial = sp.Integer(1)
        angular = sp.Integer(1)
        for factor in expr.args:
            fsyms = factor.free_symbols
            if fsyms & radial_set and not (fsyms & angular_set):
                radial *= factor
            elif fsyms & angular_set and not (fsyms & radial_set):
                angular *= factor
            elif not fsyms & (radial_set | angular_set):
                radial *= factor
            else:
                return None
        return _clean_expr(radial), _clean_expr(angular)
    fsyms = expr.free_symbols
    if fsyms & radial_set and not (fsyms & angular_set):
        return _clean_expr(expr), sp.Integer(1)
    if fsyms & angular_set and not (fsyms & radial_set):
        return sp.Integer(1), _clean_expr(expr)
    if not fsyms & (radial_set | angular_set):
        return _clean_expr(expr), sp.Integer(1)
    return None


def _ball_poly_moment(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    radius: sp.Expr,
    center: tuple[sp.Expr, ...],
) -> sp.Expr | None:
    """Integrate a polynomial over a centered or translated Euclidean ball."""
    dim = len(variables)
    local_vars = sp.symbols(f"_ball_u0:{dim}", real=True)
    substitutions = {
        var: local + shift for var, local, shift in zip(variables, local_vars, center, strict=True)
    }
    transformed = sp.expand(sp.sympify(expr).subs(substitutions))
    try:
        poly = sp.Poly(transformed, *local_vars)
    except sp.PolynomialError:
        return None

    total = sp.Integer(0)
    for monomial, coefficient in poly.terms():
        if any(exponent % 2 for exponent in monomial):
            continue
        degree = sum(monomial)
        numerator = sp.Integer(1)
        for exponent in monomial:
            numerator *= sp.gamma(sp.Rational(exponent + 1, 2))
        denominator = sp.gamma(sp.Rational(degree + dim, 2) + 1)
        total += coefficient * sp.sympify(radius) ** (degree + dim) * numerator / denominator
    return sp.simplify(total)


def _radial_profile(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    center: tuple[sp.Expr, ...],
    scales: tuple[sp.Expr, ...] | None = None,
) -> sp.Expr | None:
    """Return h(t) when ``expr == h(rho**2)`` can be proved symbolically."""
    scales = scales or tuple(sp.Integer(1) for _ in variables)
    t = sp.Symbol("_rho2", nonnegative=True, real=True)
    profiles = []
    for variable, scale, shift in zip(variables, scales, center, strict=True):
        substitutions = dict(zip(variables, center, strict=True))
        substitutions[variable] = shift + sp.sympify(scale) * sp.sqrt(t)
        profiles.append(_clean_expr(sp.sympify(expr).subs(substitutions)))

    profile = profiles[0]
    if any(sp.simplify(other - profile) != 0 for other in profiles[1:]):
        return None
    if set(variables) & profile.free_symbols:
        return None

    rho2 = sp.Add(
        *(
            (variable - shift) ** 2 / sp.sympify(scale) ** 2
            for variable, scale, shift in zip(variables, scales, center, strict=True)
        )
    )
    candidate = _clean_expr(profile.subs(t, rho2))
    if sp.simplify(sp.sympify(expr) - candidate) != 0:
        return None
    return profile


def _radial_ball_integral(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    center: tuple[sp.Expr, ...],
    radius: sp.Expr,
    scales: tuple[sp.Expr, ...] | None = None,
) -> sp.Expr | None:
    """Integrate a proved radial profile over a scaled Euclidean ball."""
    scales = scales or tuple(sp.Integer(1) for _ in variables)
    profile = _radial_profile(expr, variables, center, scales)
    if profile is None:
        return None
    dim = len(variables)
    t = sp.Symbol("_rho2", nonnegative=True, real=True)
    radial_var = sp.Symbol("_r", nonnegative=True, real=True)
    scale = sp.prod(sp.Abs(axis) for axis in scales)
    sphere_area = 2 * sp.pi ** sp.Rational(dim, 2) / sp.gamma(sp.Rational(dim, 2))
    integrand = _clean_expr(
        profile.subs(t, radial_var**2) * scale * sphere_area * radial_var ** (dim - 1)
    )
    return sp.integrate(integrand, (radial_var, 0, radius))


def _is_definitely_true(condition: sp.Expr) -> bool:
    """Return True only when SymPy can prove ``condition``."""
    simplified = sp.simplify(condition)
    return simplified in (True, sp.true)


def _validate_radius(radius: sp.Expr, *, name: str = "radius", allow_zero: bool = False) -> sp.Expr:
    value = sp.sympify(radius)
    if value.has(sp.nan, sp.zoo):
        raise ValueError(f"{name} must be finite or symbolically well-defined")
    if value.is_real is False:
        raise ValueError(f"{name} must be real")
    if allow_zero:
        if value.is_negative is True or _is_definitely_true(value < 0):
            raise ValueError(f"{name} must be nonnegative")
    elif value.is_nonpositive is True or _is_definitely_true(value <= 0):
        raise ValueError(f"{name} must be positive")
    return value


def _validate_center(
    center: tuple[sp.Expr, ...], variables: tuple[sp.Symbol, ...]
) -> tuple[sp.Expr, ...]:
    if not center:
        return tuple(sp.Integer(0) for _ in variables)
    if len(center) != len(variables):
        raise ValueError("center must have one coordinate per region variable")
    values = tuple(sp.sympify(c) for c in center)
    if any(c.has(sp.nan, sp.zoo, sp.oo, -sp.oo) for c in values):
        raise ValueError("center coordinates must be finite and symbolically well-defined")
    if any(c.is_real is False for c in values):
        raise ValueError("center coordinates must be real")
    if any(c.free_symbols & set(variables) for c in values):
        raise ValueError("center coordinates cannot depend on region variables")
    return values


def _valid_dependency_order(ranges: tuple[tuple, ...], *, inner_first: bool) -> bool:
    """Check nested-bound dependencies for one consistent range orientation."""
    variables = [r[0] for r in ranges]
    positions = {v: i for i, v in enumerate(variables)}
    for i, (_var, lo, hi) in enumerate(ranges):
        deps = (sp.sympify(lo).free_symbols | sp.sympify(hi).free_symbols) & set(variables)
        for dep in deps:
            j = positions[dep]
            if inner_first and j <= i:
                return False
            if not inner_first and j >= i:
                return False
    return True


def _validate_nested_ranges(ranges: tuple[tuple, ...]) -> None:
    """Accept either consistently inner-first or consistently outer-first descriptions."""
    if len(ranges) <= 1:
        return
    if not (
        _valid_dependency_order(ranges, inner_first=True)
        or _valid_dependency_order(ranges, inner_first=False)
    ):
        raise ValueError("region ranges must use a consistent nested dependency order")


def _radial_interval(region: Region):
    """Return a concentric radial interval signature when available."""
    if isinstance(region, DiskRegion):
        return region.variables, region.center, sp.Integer(0), sp.sympify(region.radius)
    if isinstance(region, AnnulusRegion):
        return (
            region.variables,
            (sp.Integer(0), sp.Integer(0)),
            sp.sympify(region.inner_radius),
            sp.sympify(region.outer_radius),
        )
    if isinstance(region, BallRegion):
        return region.variables, region.center, sp.Integer(0), sp.sympify(region.radius)
    if isinstance(region, SphericalShellRegion):
        return (
            region.variables,
            tuple(sp.Integer(0) for _ in region.variables),
            sp.sympify(region.inner_radius),
            sp.sympify(region.outer_radius),
        )
    return None


def _has_positive_overlap(left: Region, right: Region) -> bool | None:
    """Detect obvious positive-measure overlap for simple concentric radial pieces."""
    if left == right:
        vol = left.constant_volume()
        if vol is not None and not _is_definitely_true(sp.Eq(vol, 0)):
            return True
    a = _radial_interval(left)
    b = _radial_interval(right)
    if a is None or b is None:
        return None
    vars_a, center_a, lo_a, hi_a = a
    vars_b, center_b, lo_b, hi_b = b
    if vars_a != vars_b or len(center_a) != len(center_b):
        return False
    if any(sp.simplify(x - y) != 0 for x, y in zip(center_a, center_b, strict=True)):
        return None
    overlap_lo = sp.Max(lo_a, lo_b)
    overlap_hi = sp.Min(hi_a, hi_b)
    if _is_definitely_true(overlap_hi > overlap_lo):
        return True
    if _is_definitely_true(overlap_hi <= overlap_lo):
        return False
    return None


@dataclass(frozen=True)
class Region:
    """Base region class for structured multiple-integration domains."""

    ranges: tuple[tuple, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized = tuple(tuple(r) for r in self.ranges)
        if any(len(r) != 3 for r in normalized):
            raise ValueError("each region range must be a (variable, lower, upper) triple")
        variables = tuple(r[0] for r in normalized)
        if any(not isinstance(v, sp.Symbol) for v in variables):
            raise TypeError("region variables must be SymPy Symbol objects")
        if len(set(variables)) != len(variables):
            raise ValueError("region variables must be distinct")
        for var, lo, hi in normalized:
            lo_s = sp.sympify(lo)
            hi_s = sp.sympify(hi)
            if lo_s.has(sp.nan, sp.zoo) or hi_s.has(sp.nan, sp.zoo):
                raise ValueError("region bounds cannot contain nan or zoo")
            if var in lo_s.free_symbols or var in hi_s.free_symbols:
                raise ValueError("a region bound cannot depend on its own integration variable")
        object.__setattr__(self, "ranges", normalized)

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return tuple(r[0] for r in self.ranges)

    def normalized_ranges(self) -> tuple[tuple, ...]:
        return tuple((v, _clean_expr(lo), _clean_expr(hi)) for v, lo, hi in self.ranges)

    def constant_volume(self) -> sp.Expr | None:
        return None

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        return None

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        return None

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return False

    def symmetric_range(self, var: sp.Symbol) -> tuple[sp.Expr, sp.Expr] | None:
        return None


@dataclass(frozen=True)
class BoxRegion(Region):
    def __post_init__(self) -> None:
        super().__post_init__()
        variables = set(self.variables)
        for _, lo, hi in self.ranges:
            if (sp.sympify(lo).free_symbols | sp.sympify(hi).free_symbols) & variables:
                raise ValueError("BoxRegion bounds must be independent of all region variables")

    def constant_volume(self) -> sp.Expr | None:
        vol = sp.Integer(1)
        vars_set = set(self.variables)
        for _, lo, hi in self.ranges:
            lo_s = sp.sympify(lo)
            hi_s = sp.sympify(hi)
            if (lo_s.free_symbols | hi_s.free_symbols) & vars_set:
                return None
            vol *= hi_s - lo_s
        return sp.simplify(vol)

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return self.symmetric_range(var) is not None

    def symmetric_range(self, var: sp.Symbol) -> tuple[sp.Expr, sp.Expr] | None:
        for v, lo, hi in self.ranges:
            if v == var:
                lo_s = sp.sympify(lo)
                hi_s = sp.sympify(hi)
                if sp.simplify(lo_s + hi_s) == 0:
                    return lo_s, hi_s
        return None


@dataclass(frozen=True)
class IteratedRegion(Region):
    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_nested_ranges(self.ranges)

    def symmetric_range(self, var: sp.Symbol) -> tuple[sp.Expr, sp.Expr] | None:
        """Return a symmetric range only when the *whole region* is reflection invariant.

        In inner-first range order, an outer variable can occur in earlier (inner)
        bounds.  Checking only the variable's own interval is therefore insufficient:
        every other bound that depends on ``var`` must also be unchanged by
        ``var -> -var``.
        """
        own = None
        for v, lo, hi in self.ranges:
            if v == var:
                lo_s = sp.sympify(lo)
                hi_s = sp.sympify(hi)
                if sp.simplify(lo_s + hi_s) != 0:
                    return None
                own = (lo_s, hi_s)
                break
        if own is None:
            return None

        for v, lo, hi in self.ranges:
            if v == var:
                continue
            lo_s = sp.sympify(lo)
            hi_s = sp.sympify(hi)
            if var in lo_s.free_symbols and sp.simplify(lo_s.subs(var, -var) - lo_s) != 0:
                return None
            if var in hi_s.free_symbols and sp.simplify(hi_s.subs(var, -var) - hi_s) != 0:
                return None
        return own

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return self.symmetric_range(var) is not None


@dataclass(frozen=True)
class GraphRegion(IteratedRegion):
    outer_var: sp.Symbol | None = None
    inner_var: sp.Symbol | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.ranges) != 2:
            raise ValueError("GraphRegion requires exactly two ranges")
        variables = set(self.variables)
        if self.outer_var is not None and self.outer_var not in variables:
            raise ValueError("outer_var must be one of the region variables")
        if self.inner_var is not None and self.inner_var not in variables:
            raise ValueError("inner_var must be one of the region variables")
        if (
            self.outer_var is not None
            and self.inner_var is not None
            and self.outer_var == self.inner_var
        ):
            raise ValueError("outer_var and inner_var must be distinct")

    def constant_volume(self) -> sp.Expr | None:
        if len(self.ranges) != 2:
            return None
        (y, lo, hi), (x, a, b) = self.ranges
        lo_s = sp.sympify(lo)
        hi_s = sp.sympify(hi)
        if lo_s.free_symbols - {x} or hi_s.free_symbols - {x}:
            return None
        try:
            return sp.simplify(sp.integrate(hi_s - lo_s, (x, sp.sympify(a), sp.sympify(b))))
        except _REGION_ERRORS:
            return None

    def _linear_data(self):
        if len(self.ranges) != 2:
            return None
        (y, lo, hi), (x, a, b) = self.ranges
        a = sp.sympify(a)
        b = sp.sympify(b)
        lo = sp.expand(sp.sympify(lo))
        hi = sp.expand(sp.sympify(hi))
        if a.free_symbols | b.free_symbols:
            return None
        if lo.free_symbols - {x} or hi.free_symbols - {x}:
            return None
        try:
            plo = sp.Poly(lo, x)
            phi = sp.Poly(hi, x)
        except sp.PolynomialError:
            return None
        if plo.degree() > 1 or phi.degree() > 1:
            return None
        m1 = plo.nth(1) if plo.degree() >= 1 else sp.Integer(0)
        c1 = plo.nth(0)
        m2 = phi.nth(1) if phi.degree() >= 1 else sp.Integer(0)
        c2 = phi.nth(0)
        return x, y, a, b, m1, c1, m2, c2

    def reversed_pieces(self) -> list[list[tuple]] | None:
        data = self._linear_data()
        if data is None:
            return None
        x, y, a, b, m1, c1, m2, c2 = data

        def y_at(m, c, xv):
            return sp.simplify(m * xv + c)

        pts = [y_at(m1, c1, a), y_at(m1, c1, b), y_at(m2, c2, a), y_at(m2, c2, b)]
        try:
            if sp.simplify(m1 - m2) != 0:
                x_cross = sp.simplify((c2 - c1) / (m1 - m2))
                if float(sp.N(a)) - 1e-12 <= float(sp.N(x_cross)) <= float(sp.N(b)) + 1e-12:
                    pts.append(y_at(m1, c1, x_cross))
        except _REGION_ERRORS:
            pass

        uniq = []
        for p in pts:
            p = sp.simplify(p)
            if p not in uniq:
                uniq.append(p)
        try:
            uniq = sorted(uniq, key=lambda t: float(sp.N(t)))
        except _REGION_ERRORS:
            return None
        if len(uniq) < 2:
            return None

        # Reverse the graph by slicing the y-axis into intervals where the
        # active lower and upper x-bounds stay on the same candidate lines.
        def inv(m, c):
            if sp.simplify(m) == 0:
                return None
            return sp.simplify((y - c) / m)

        lower_cands = [a]
        upper_cands = [b]
        inv1 = inv(m1, c1)
        inv2 = inv(m2, c2)
        if inv1 is not None:
            if sp.N(m1) > 0:
                upper_cands.append(inv1)
            else:
                lower_cands.append(inv1)
        if inv2 is not None:
            if sp.N(m2) > 0:
                lower_cands.append(inv2)
            else:
                upper_cands.append(inv2)

        pieces = []
        for left, right in zip(uniq[:-1], uniq[1:], strict=False):
            if sp.simplify(left - right) == 0:
                continue
            mid = sp.simplify((left + right) / 2)

            def choose(cands, kind, midpoint=mid):
                best = None
                best_val = None
                for cand in cands:
                    val = float(sp.N(cand.subs(y, midpoint) if hasattr(cand, "subs") else cand))
                    if (
                        best is None
                        or kind == "max"
                        and val > best_val + 1e-12
                        or kind == "min"
                        and val < best_val - 1e-12
                    ):
                        best, best_val = cand, val
                return sp.simplify(best)

            xlo = choose(lower_cands, "max")
            xhi = choose(upper_cands, "min")
            try:
                if float(sp.N(xlo.subs(y, mid))) <= float(sp.N(xhi.subs(y, mid))) + 1e-12:
                    pieces.append([(x, xlo, xhi), (y, left, right)])
            except _REGION_ERRORS:
                return None
        return pieces or None


@dataclass(frozen=True)
class SimplexRegion(IteratedRegion):
    dimension: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dimension <= 0:
            raise ValueError("SimplexRegion dimension must be positive")
        if len(self.ranges) != self.dimension:
            raise ValueError("SimplexRegion dimension must match its number of ranges")

    def constant_volume(self) -> sp.Expr | None:
        return sp.simplify(sp.Integer(1) / sp.factorial(self.dimension)) if self.dimension else None

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        vars_ = list(self.variables)
        try:
            poly = sp.Poly(sp.expand(expr), *vars_)
        except sp.PolynomialError:
            return None
        total = sp.Integer(0)
        for monom, coeff in poly.terms():
            deg_sum = sum(monom)
            numer = sp.Integer(1)
            for a in monom:
                numer *= sp.factorial(a)
            denom = sp.factorial(self.dimension + deg_sum)
            total += coeff * numer / denom
        return sp.simplify(total)


@dataclass(frozen=True)
class AffineSimplexRegion(Region):
    shifts: tuple[sp.Expr, ...] = field(default_factory=tuple)
    scales: tuple[sp.Expr, ...] = field(default_factory=tuple)
    dimension: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dimension <= 0:
            raise ValueError("AffineSimplexRegion dimension must be positive")
        if len(self.ranges) != self.dimension:
            raise ValueError("AffineSimplexRegion dimension must match its number of ranges")
        if len(self.shifts) != self.dimension or len(self.scales) != self.dimension:
            raise ValueError("AffineSimplexRegion requires one shift and scale per dimension")
        scales = tuple(sp.sympify(s) for s in self.scales)
        if any(s.is_zero is True or _is_definitely_true(sp.Eq(s, 0)) for s in scales):
            raise ValueError("AffineSimplexRegion scales must be nonzero")
        object.__setattr__(self, "shifts", tuple(sp.sympify(s) for s in self.shifts))
        object.__setattr__(self, "scales", scales)

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return tuple(v for v, _, _ in self.ranges)

    def normalized_ranges(self) -> tuple[tuple, ...]:
        return (
            "AffineSimplexRegion",
            tuple(sp.simplify(s) for s in self.shifts),
            tuple(sp.simplify(s) for s in self.scales),
        )

    def constant_volume(self) -> sp.Expr | None:
        if not self.dimension or len(self.scales) != self.dimension:
            return None
        scale = sp.Integer(1)
        for s in self.scales:
            scale *= sp.Abs(sp.sympify(s))
        return sp.simplify(scale / sp.factorial(self.dimension))

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        vars_ = self.variables
        if len(vars_) != self.dimension:
            return None
        uvars = sp.symbols(f"_u0:{self.dimension}", real=True)
        subs = {
            v: sp.sympify(a) + sp.sympify(s) * u
            for v, a, s, u in zip(vars_, self.shifts, self.scales, uvars, strict=True)
        }
        jac = sp.Integer(1)
        for s in self.scales:
            jac *= sp.Abs(sp.sympify(s))
        transformed = sp.expand(sp.sympify(expr).subs(subs) * jac)
        simplex_ranges = tuple((u, 0, 1 - sum(uvars[:i])) for i, u in enumerate(uvars))
        simplex = SimplexRegion(simplex_ranges, dimension=self.dimension)
        return simplex.polynomial_moment(transformed)


@dataclass(frozen=True)
class DiskRegion(IteratedRegion):
    radius: sp.Expr = sp.Integer(1)
    center: tuple[sp.Expr, sp.Expr] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.variables) != 2:
            raise ValueError("DiskRegion requires exactly two variables")
        radius = _validate_radius(self.radius, allow_zero=True)
        center = _validate_center(tuple(self.center), self.variables)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "center", center)

    def constant_volume(self) -> sp.Expr | None:
        return sp.simplify(sp.pi * self.radius**2)

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        return _ball_poly_moment(expr, self.variables, self.radius, self.center)

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        radial = _radial_ball_integral(sp.sympify(expr), self.variables, self.center, self.radius)
        if radial is not None:
            return _clean_expr(radial)
        x, y = self.variables
        cx, cy = self.center
        r = sp.Symbol("_r", nonnegative=True, real=True)
        theta = sp.Symbol("_theta", real=True)
        # Translate to the disk center, then convert to polar coordinates.
        polar_expr = _clean_expr(
            sp.trigsimp(
                sp.expand_trig(
                    sp.sympify(expr).subs({x: cx + r * sp.cos(theta), y: cy + r * sp.sin(theta)})
                )
            )
        )
        pieces = _split_dependence(polar_expr, (r,), (theta,))
        if pieces is None:
            return None
        radial_part, angular_part = pieces
        angular_val = sp.integrate(angular_part, (theta, 0, 2 * sp.pi))
        radial_val = sp.integrate(sp.simplify(radial_part * r), (r, 0, self.radius))
        return _clean_expr(angular_val * radial_val)


@dataclass(frozen=True)
class AnnulusRegion(Region):
    variables_xy: tuple[sp.Symbol, sp.Symbol] = field(default_factory=tuple)
    inner_radius: sp.Expr = sp.Integer(0)
    outer_radius: sp.Expr = sp.Integer(1)

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.variables_xy) != 2 or any(
            not isinstance(v, sp.Symbol) for v in self.variables_xy
        ):
            raise ValueError("AnnulusRegion requires exactly two SymPy Symbol variables")
        if len(set(self.variables_xy)) != 2:
            raise ValueError("AnnulusRegion variables must be distinct")
        inner = _validate_radius(self.inner_radius, name="inner_radius", allow_zero=True)
        outer = _validate_radius(self.outer_radius, name="outer_radius")
        if _is_definitely_true(inner >= outer):
            raise ValueError("inner_radius must be smaller than outer_radius")
        object.__setattr__(self, "variables_xy", tuple(self.variables_xy))
        object.__setattr__(self, "inner_radius", inner)
        object.__setattr__(self, "outer_radius", outer)

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return self.variables_xy

    def normalized_ranges(self) -> tuple[tuple, ...]:
        return ("AnnulusRegion", sp.simplify(self.inner_radius), sp.simplify(self.outer_radius))

    def constant_volume(self) -> sp.Expr | None:
        return sp.simplify(sp.pi * (self.outer_radius**2 - self.inner_radius**2))

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        center = (sp.Integer(0), sp.Integer(0))
        outer = _ball_poly_moment(expr, self.variables_xy, self.outer_radius, center)
        inner = _ball_poly_moment(expr, self.variables_xy, self.inner_radius, center)
        if outer is None or inner is None:
            return None
        return sp.simplify(outer - inner)

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        center = (sp.Integer(0), sp.Integer(0))
        outer = _radial_ball_integral(expr, self.variables_xy, center, self.outer_radius)
        inner = _radial_ball_integral(expr, self.variables_xy, center, self.inner_radius)
        if outer is None or inner is None:
            return None
        return sp.simplify(outer - inner)

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return var in self.variables_xy


@dataclass(frozen=True)
class BallRegion(IteratedRegion):
    radius: sp.Expr = sp.Integer(1)
    dimension: int = 0
    center: tuple[sp.Expr, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dimension <= 0:
            raise ValueError("BallRegion dimension must be positive")
        if len(self.variables) != self.dimension:
            raise ValueError("BallRegion dimension must match its number of variables")
        radius = _validate_radius(self.radius, allow_zero=True)
        center = _validate_center(tuple(self.center), self.variables)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "center", center)

    def constant_volume(self) -> sp.Expr | None:
        dim = self.dimension
        return sp.simplify(
            sp.pi ** (sp.Rational(dim, 2)) * self.radius**dim / sp.gamma(sp.Rational(dim, 2) + 1)
        )

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        return _ball_poly_moment(expr, self.variables, self.radius, self.center)

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        vars_ = self.variables
        radial = _radial_ball_integral(sp.sympify(expr), vars_, self.center, self.radius)
        if radial is not None:
            return _clean_expr(radial)
        if self.dimension != 3 or len(vars_) != 3:
            return None
        x, y, z = vars_
        cx, cy, cz = self.center
        r = sp.Symbol("_r", nonnegative=True, real=True)
        phi = sp.Symbol("_phi", real=True)
        theta = sp.Symbol("_theta", real=True)
        # The 3D case uses spherical coordinates and keeps only products that
        # separate cleanly into radial and angular pieces.
        spherical_expr = _clean_expr(
            sp.trigsimp(
                sp.expand_trig(
                    sp.sympify(expr).subs(
                        {
                            x: cx + r * sp.sin(phi) * sp.cos(theta),
                            y: cy + r * sp.sin(phi) * sp.sin(theta),
                            z: cz + r * sp.cos(phi),
                        }
                    )
                )
            )
        )
        pieces = _split_dependence(spherical_expr, (r,), (theta, phi))
        if pieces is None:
            return None
        radial_part, angular_part = pieces
        angular_val = sp.integrate(
            angular_part * sp.sin(phi), (theta, 0, 2 * sp.pi), (phi, 0, sp.pi)
        )
        radial_val = sp.integrate(_clean_expr(radial_part * r**2), (r, 0, self.radius))
        return _clean_expr(angular_val * radial_val)


@dataclass(frozen=True)
class EllipsoidRegion(Region):
    variables_nd: tuple[sp.Symbol, ...] = field(default_factory=tuple)
    axes: tuple[sp.Expr, ...] = field(default_factory=tuple)
    center: tuple[sp.Expr, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.variables_nd or any(not isinstance(v, sp.Symbol) for v in self.variables_nd):
            raise ValueError("EllipsoidRegion requires SymPy Symbol variables")
        if len(set(self.variables_nd)) != len(self.variables_nd):
            raise ValueError("EllipsoidRegion variables must be distinct")
        if len(self.axes) != len(self.variables_nd):
            raise ValueError("EllipsoidRegion requires one semi-axis per variable")
        axes = tuple(_validate_radius(a, name="semi-axis") for a in self.axes)
        center = _validate_center(tuple(self.center), tuple(self.variables_nd))
        object.__setattr__(self, "variables_nd", tuple(self.variables_nd))
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "center", center)

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return self.variables_nd

    def normalized_ranges(self) -> tuple[tuple, ...]:
        return (
            "EllipsoidRegion",
            tuple(sp.simplify(a) for a in self.axes),
            tuple(sp.simplify(c) for c in self.center),
        )

    def constant_volume(self) -> sp.Expr | None:
        dim = len(self.axes)
        scale = sp.Integer(1)
        for a in self.axes:
            scale *= sp.Abs(sp.sympify(a))
        return sp.simplify(
            scale * sp.pi ** (sp.Rational(dim, 2)) / sp.gamma(sp.Rational(dim, 2) + 1)
        )

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        variables = self.variables_nd
        local_vars = sp.symbols(f"_u0:{len(variables)}", real=True)
        jacobian = sp.Integer(1)
        substitutions = {}
        for variable, axis, shift, local in zip(
            variables, self.axes, self.center, local_vars, strict=True
        ):
            substitutions[variable] = shift + axis * local
            jacobian *= sp.Abs(axis)
        transformed = sp.expand(sp.sympify(expr).subs(substitutions) * jacobian)
        zeros = tuple(sp.Integer(0) for _ in local_vars)
        return _ball_poly_moment(transformed, local_vars, sp.Integer(1), zeros)

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        return _radial_ball_integral(
            sp.sympify(expr),
            self.variables_nd,
            self.center,
            sp.Integer(1),
            self.axes,
        )

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        if var not in self.variables_nd:
            return False
        index = self.variables_nd.index(var)
        return sp.simplify(self.center[index]) == 0


@dataclass(frozen=True)
class SphericalShellRegion(Region):
    variables_nd: tuple[sp.Symbol, ...] = field(default_factory=tuple)
    inner_radius: sp.Expr = sp.Integer(0)
    outer_radius: sp.Expr = sp.Integer(1)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.variables_nd or any(not isinstance(v, sp.Symbol) for v in self.variables_nd):
            raise ValueError("SphericalShellRegion requires SymPy Symbol variables")
        if len(set(self.variables_nd)) != len(self.variables_nd):
            raise ValueError("SphericalShellRegion variables must be distinct")
        inner = _validate_radius(self.inner_radius, name="inner_radius", allow_zero=True)
        outer = _validate_radius(self.outer_radius, name="outer_radius")
        if _is_definitely_true(inner >= outer):
            raise ValueError("inner_radius must be smaller than outer_radius")
        object.__setattr__(self, "variables_nd", tuple(self.variables_nd))
        object.__setattr__(self, "inner_radius", inner)
        object.__setattr__(self, "outer_radius", outer)

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return self.variables_nd

    def normalized_ranges(self) -> tuple[tuple, ...]:
        return (
            "SphericalShellRegion",
            sp.simplify(self.inner_radius),
            sp.simplify(self.outer_radius),
            len(self.variables_nd),
        )

    def constant_volume(self) -> sp.Expr | None:
        dim = len(self.variables_nd)
        return sp.simplify(
            sp.pi ** (sp.Rational(dim, 2))
            * (self.outer_radius**dim - self.inner_radius**dim)
            / sp.gamma(sp.Rational(dim, 2) + 1)
        )

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        center = tuple(sp.Integer(0) for _ in self.variables_nd)
        outer = _ball_poly_moment(expr, self.variables_nd, self.outer_radius, center)
        inner = _ball_poly_moment(expr, self.variables_nd, self.inner_radius, center)
        if outer is None or inner is None:
            return None
        return sp.simplify(outer - inner)

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        center = tuple(sp.Integer(0) for _ in self.variables_nd)
        outer = _radial_ball_integral(expr, self.variables_nd, center, self.outer_radius)
        inner = _radial_ball_integral(expr, self.variables_nd, center, self.inner_radius)
        if outer is None or inner is None:
            return None
        return sp.simplify(outer - inner)

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return var in self.variables_nd


@dataclass(frozen=True)
class UnionRegion(Region):
    """Finite disjoint union of regions with a common ordered variable tuple.

    Pieces must be disjoint up to measure-zero boundaries.  The constructor rejects
    overlaps it can prove for supported concentric radial pieces; when overlap is
    symbolically undecidable, callers remain responsible for the disjointness claim.
    """

    pieces: tuple[Region, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super().__post_init__()
        pieces = tuple(self.pieces)
        if not pieces:
            raise ValueError("UnionRegion requires at least one piece")
        if any(not isinstance(piece, Region) for piece in pieces):
            raise TypeError("UnionRegion pieces must be Region objects")
        variables = pieces[0].variables
        if any(piece.variables != variables for piece in pieces[1:]):
            raise ValueError("UnionRegion pieces must use the same ordered variables")
        for i, left in enumerate(pieces):
            for right in pieces[i + 1 :]:
                if _has_positive_overlap(left, right) is True:
                    raise ValueError(
                        "UnionRegion pieces must be disjoint up to measure-zero boundaries"
                    )
        object.__setattr__(self, "pieces", pieces)

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return self.pieces[0].variables if self.pieces else tuple()

    def normalized_ranges(self) -> tuple[tuple, ...]:
        return tuple((type(reg).__name__, reg.normalized_ranges()) for reg in self.pieces)

    def constant_volume(self) -> sp.Expr | None:
        total = sp.Integer(0)
        for reg in self.pieces:
            vol = reg.constant_volume()
            if vol is None:
                return None
            total += vol
        return sp.simplify(total)

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        total = sp.Integer(0)
        for reg in self.pieces:
            val = reg.polynomial_moment(expr)
            if val is None:
                return None
            total += val
        return sp.simplify(total)

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        total = sp.Integer(0)
        for reg in self.pieces:
            val = reg.radial_integral(expr)
            if val is None:
                return None
            total += val
        return sp.simplify(total)

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return all(reg.is_reflection_invariant(var) for reg in self.pieces)


def _poly_is_affine(expr: sp.Expr, var: sp.Symbol) -> bool:
    try:
        poly = sp.Poly(sp.expand(expr), var)
    except sp.PolynomialError:
        return False
    return poly.degree() <= 1


def _structural_ranges(ranges: list[tuple]) -> list[tuple]:
    """Convert public inner-first integration ranges to outer-first structural order."""
    return list(reversed(ranges))


def _range_order_candidates(ranges: list[tuple], structural_order: str = "auto"):
    """Yield plausible outer-first structural orders and canonical inner-first ranges.

    ``inner-first`` is strict and is used by :func:`multiple_integrate`; ``auto``
    additionally accepts an outer-first structural description for standalone region
    classification utilities.
    """
    public = list(ranges)
    if structural_order not in {"auto", "inner-first", "outer-first"}:
        raise ValueError("structural_order must be 'auto', 'inner-first', or 'outer-first'")
    if structural_order == "inner-first":
        candidates = [list(reversed(public))]
    elif structural_order == "outer-first":
        candidates = [public]
    else:
        candidates = [list(reversed(public))]
        if public != candidates[0]:
            candidates.append(public)
    for structural in candidates:
        yield structural, tuple(reversed(structural))


def match_standard_simplex(
    ranges: list[tuple], *, structural_order: str = "auto"
) -> SimplexRegion | None:
    for structural, canonical in _range_order_candidates(ranges, structural_order):
        seen = []
        matched = True
        for idx, (var, lo, hi) in enumerate(structural):
            lo_s = sp.sympify(lo)
            hi_s = sp.expand(sp.sympify(hi))
            if lo_s != 0:
                matched = False
                break
            expected = sp.Integer(1) if idx == 0 else 1 - sum(seen)
            if sp.simplify(hi_s - expected) != 0:
                matched = False
                break
            seen.append(var)
        if matched:
            return SimplexRegion(canonical, dimension=len(ranges))
    return None


def match_affine_simplex(
    ranges: list[tuple], *, structural_order: str = "auto"
) -> AffineSimplexRegion | None:
    if len(ranges) < 2:
        return None

    for sranges, canonical_ranges in _range_order_candidates(ranges, structural_order):
        if any(hi is None for _, _, hi in sranges):
            continue

        shifts = []
        scales = []
        prev_vars = []
        matched = True
        for idx, (var, lo, hi) in enumerate(sranges):
            lo_s = sp.simplify(sp.sympify(lo))
            hi_s = sp.expand(sp.sympify(hi))
            if idx == 0:
                if hi_s.free_symbols or lo_s.free_symbols:
                    matched = False
                    break
                scale = sp.simplify(hi_s - lo_s)
                if scale == 0:
                    matched = False
                    break
                shifts.append(lo_s)
                scales.append(scale)
            else:
                if lo_s.free_symbols:
                    matched = False
                    break
                if hi_s.free_symbols - set(prev_vars):
                    matched = False
                    break
                if any(not _poly_is_affine(hi_s, pv) for pv in prev_vars):
                    matched = False
                    break
                target = sp.Integer(1)
                for pv, sh, sc in zip(prev_vars, shifts, scales, strict=True):
                    target -= sp.simplify((pv - sh) / sc)
                if target == 0:
                    matched = False
                    break
                sc_i = sp.simplify((hi_s - lo_s) / target)
                if sc_i.free_symbols:
                    matched = False
                    break
                if sp.simplify(lo_s + sc_i * target - hi_s) != 0:
                    matched = False
                    break
                shifts.append(lo_s)
                scales.append(sc_i)
            prev_vars.append(var)

        if not matched:
            continue
        if all(sp.simplify(sh) == 0 for sh in shifts) and all(
            sp.simplify(sc - 1) == 0 for sc in scales
        ):
            continue
        return AffineSimplexRegion(
            canonical_ranges,
            shifts=tuple(reversed(shifts)),
            scales=tuple(reversed(scales)),
            dimension=len(ranges),
        )
    return None


def match_graph_region(
    ranges: list[tuple], *, structural_order: str = "auto"
) -> GraphRegion | None:
    if len(ranges) != 2:
        return None

    for sranges, canonical_ranges in _range_order_candidates(ranges, structural_order):
        (x, a, b), (y, lo, hi) = sranges
        a_s = sp.sympify(a)
        b_s = sp.sympify(b)
        lo_s = sp.sympify(lo)
        hi_s = sp.sympify(hi)
        if a_s.free_symbols or b_s.free_symbols:
            continue
        if lo_s.free_symbols - {x} or hi_s.free_symbols - {x}:
            continue
        if not ((lo_s.free_symbols | hi_s.free_symbols) & {x}):
            continue
        if not (_poly_is_affine(lo_s, x) and _poly_is_affine(hi_s, x)):
            continue
        return GraphRegion(canonical_ranges, outer_var=x, inner_var=y)
    return None


def match_standard_disk(
    ranges: list[tuple], *, structural_order: str = "auto"
) -> DiskRegion | None:
    """Recognize an axis-aligned disk, including a translated disk."""
    if len(ranges) != 2:
        return None
    region_vars = {r[0] for r in ranges}
    for sranges, canonical_ranges in _range_order_candidates(ranges, structural_order):
        (x, a, b), (y, lo, hi) = sranges
        a_s = sp.sympify(a)
        b_s = sp.sympify(b)
        if (a_s.free_symbols | b_s.free_symbols) & region_vars:
            continue
        cx = sp.simplify((a_s + b_s) / 2)
        radius = sp.simplify((b_s - a_s) / 2)
        if radius.is_nonpositive is True:
            continue
        lo_s = sp.sympify(lo)
        hi_s = sp.sympify(hi)
        cy = sp.simplify((lo_s + hi_s) / 2)
        if cy.free_symbols & region_vars:
            continue
        half = sp.simplify((hi_s - lo_s) / 2)
        target = sp.sqrt(radius**2 - (x - cx) ** 2)
        if sp.simplify(half - target) != 0:
            continue
        canonical_vars = tuple(r[0] for r in canonical_ranges)
        structural_centers = {x: cx, y: cy}
        center = tuple(structural_centers[v] for v in canonical_vars)
        return DiskRegion(canonical_ranges, radius=radius, center=center)
    return None


def match_standard_ball(
    ranges: list[tuple], *, structural_order: str = "auto"
) -> BallRegion | None:
    """Recognize an axis-aligned ball of dimension at least three, with translation."""
    if len(ranges) < 3:
        return None
    region_vars = {r[0] for r in ranges}

    for sranges, canonical_ranges in _range_order_candidates(ranges, structural_order):
        x0, a0, b0 = sranges[0]
        a0 = sp.sympify(a0)
        b0 = sp.sympify(b0)
        if (a0.free_symbols | b0.free_symbols) & region_vars:
            continue
        c0 = sp.simplify((a0 + b0) / 2)
        radius = sp.simplify((b0 - a0) / 2)
        if radius.is_nonpositive is True:
            continue
        centers = [c0]
        variables = [x0]
        sumsq = (x0 - c0) ** 2
        matched = True
        for var, lo, hi in sranges[1:]:
            lo_s = sp.sympify(lo)
            hi_s = sp.sympify(hi)
            center = sp.simplify((lo_s + hi_s) / 2)
            if center.free_symbols & region_vars:
                matched = False
                break
            half = sp.simplify((hi_s - lo_s) / 2)
            target = sp.sqrt(radius**2 - sumsq)
            if sp.simplify(half - target) != 0:
                matched = False
                break
            variables.append(var)
            centers.append(center)
            sumsq += (var - center) ** 2
        if matched:
            center_by_var = dict(zip(variables, centers, strict=True))
            canonical_center = tuple(center_by_var[r[0]] for r in canonical_ranges)
            return BallRegion(
                canonical_ranges,
                radius=radius,
                dimension=len(ranges),
                center=canonical_center,
            )
    return None


def match_standard_ellipsoid(
    ranges: list[tuple], *, structural_order: str = "auto"
) -> EllipsoidRegion | None:
    """Recognize an axis-aligned ellipsoid, including translated ellipsoids."""
    if len(ranges) < 2:
        return None
    region_vars = {r[0] for r in ranges}

    for sranges, canonical_ranges in _range_order_candidates(ranges, structural_order):
        variables = []
        centers = []
        axes = []
        x0, a0, b0 = sranges[0]
        a0 = sp.sympify(a0)
        b0 = sp.sympify(b0)
        if (a0.free_symbols | b0.free_symbols) & region_vars:
            continue
        c0 = sp.simplify((a0 + b0) / 2)
        axis0 = sp.simplify((b0 - a0) / 2)
        if axis0.is_nonpositive is True:
            continue
        variables.append(x0)
        centers.append(c0)
        axes.append(axis0)
        q = (x0 - c0) ** 2 / axis0**2
        matched = True
        for var, lo, hi in sranges[1:]:
            lo_s = sp.sympify(lo)
            hi_s = sp.sympify(hi)
            center = sp.simplify((lo_s + hi_s) / 2)
            if center.free_symbols & region_vars:
                matched = False
                break
            half = sp.simplify((hi_s - lo_s) / 2)
            ratio = sp.simplify(half / sp.sqrt(1 - q))
            if ratio.free_symbols & region_vars or ratio.is_nonpositive is True:
                matched = False
                break
            target = sp.simplify(ratio * sp.sqrt(1 - q))
            if sp.simplify(half - target) != 0:
                matched = False
                break
            variables.append(var)
            centers.append(center)
            axes.append(ratio)
            q += (var - center) ** 2 / ratio**2
        if not matched:
            continue
        if all(sp.simplify(axis - axes[0]) == 0 for axis in axes):
            continue
        axis_by_var = dict(zip(variables, axes, strict=True))
        center_by_var = dict(zip(variables, centers, strict=True))
        canonical_vars = tuple(r[0] for r in canonical_ranges)
        return EllipsoidRegion(
            canonical_ranges,
            variables_nd=canonical_vars,
            axes=tuple(axis_by_var[v] for v in canonical_vars),
            center=tuple(center_by_var[v] for v in canonical_vars),
        )
    return None


def boole(cond: sp.Expr) -> sp.Piecewise:
    """Return an indicator expression for a Boolean condition."""
    return sp.Piecewise((sp.Integer(1), sp.sympify(cond)), (sp.Integer(0), True))


def indicator_condition(expr: sp.Expr) -> sp.Expr | None:
    """Return the condition for an indicator-like Piecewise, else None."""
    expr = sp.sympify(expr)
    if not isinstance(expr, sp.Piecewise) or len(expr.args) != 2:
        return None
    (a1, c1), (a2, c2) = expr.args
    if c2 not in (True, sp.true):
        return None
    a1s = sp.sympify(a1)
    a2s = sp.sympify(a2)
    if a1s == 1 and a2s == 0:
        return sp.sympify(c1)
    if a1s == 0 and a2s == 1:
        return sp.Not(sp.sympify(c1))
    return None


def _extract_rel_bound(cond: sp.Expr):
    cond = sp.sympify(cond)
    if not isinstance(cond, Relational):
        return None
    lhs = sp.sympify(cond.lhs)
    rhs = sp.sympify(cond.rhs)
    return cond.rel_op, lhs, rhs


def _restrict_interval(region: Region, cond: sp.Expr) -> Region | None:
    if len(region.ranges) != 1:
        return None
    x, lo, hi = region.ranges[0]
    lo = sp.sympify(lo)
    hi = sp.sympify(hi)
    data = _extract_rel_bound(cond)
    if data is None:
        return None
    op, lhs, rhs = data
    if lhs == x and not rhs.free_symbols:
        if op in ("<", "<="):
            hi = sp.Min(hi, rhs)
        elif op in (">", ">="):
            lo = sp.Max(lo, rhs)
        else:
            return None
    elif rhs == x and not lhs.free_symbols:
        if op in ("<", "<="):
            lo = sp.Max(lo, lhs)
        elif op in (">", ">="):
            hi = sp.Min(hi, lhs)
        else:
            return None
    else:
        return None
    if sp.simplify(lo - hi) == 0:
        return BoxRegion(((x, lo, hi),))
    if lo.has(sp.Max) or hi.has(sp.Min):
        # If symbolic endpoint ordering is undecidable, retaining the box
        # avoids inventing a narrower or larger region.
        return BoxRegion(((x, lo, hi),))
    if lo.is_real and hi.is_real and (lo.is_number and hi.is_number) and lo > hi:
        return None
    return BoxRegion(((x, lo, hi),))


def _truth_value(expr: sp.Expr) -> bool | None:
    """Return a definite Boolean value when SymPy can prove one."""
    expr = sp.simplify(expr)
    if expr in (True, sp.true):
        return True
    if expr in (False, sp.false):
        return False
    return None


def _affine_extrema(expr: sp.Expr, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr):
    """Return endpoint extrema for an affine expression, or ``None``."""
    expr = sp.expand(sp.sympify(expr))
    try:
        poly = sp.Poly(expr, var)
    except sp.PolynomialError:
        return None
    if poly.degree() > 1:
        return None
    vals = (sp.simplify(expr.subs(var, lo)), sp.simplify(expr.subs(var, hi)))
    try:
        return sp.Min(*vals), sp.Max(*vals)
    except (TypeError, ValueError):
        return None


def _classify_graph_clip(expr, var, var_lo, var_hi, lower, upper):
    """Classify an affine candidate bound relative to fixed box limits.

    Returns ``inside`` when it stays between the box limits, ``below`` or ``above``
    when it lies wholly outside on one side, otherwise ``None``.
    """
    extrema = _affine_extrema(expr, var, var_lo, var_hi)
    if extrema is None:
        return None
    min_e, max_e = extrema
    if _truth_value(sp.Ge(min_e, lower)) is True and _truth_value(sp.Le(max_e, upper)) is True:
        return "inside"
    if _truth_value(sp.Le(max_e, lower)) is True:
        return "below"
    if _truth_value(sp.Ge(min_e, upper)) is True:
        return "above"
    return None


def _box_contains_disk(xlo, xhi, ylo, yhi, radius) -> bool:
    r = sp.sympify(radius)
    checks = [sp.Le(xlo, -r), sp.Ge(xhi, r), sp.Le(ylo, -r), sp.Ge(yhi, r)]
    return all(_truth_value(c) is True for c in checks)


def _box_inside_disk(xlo, xhi, ylo, yhi, radius_sq) -> bool:
    corners = [xlo**2 + ylo**2, xlo**2 + yhi**2, xhi**2 + ylo**2, xhi**2 + yhi**2]
    return all(_truth_value(sp.Le(sp.simplify(c), radius_sq)) is True for c in corners)


def _restrict_box_2d(region: BoxRegion, cond: sp.Expr) -> Region | None:
    if len(region.ranges) != 2:
        return None
    (x, xlo, xhi), (y, ylo, yhi) = region.ranges
    xlo = sp.sympify(xlo)
    xhi = sp.sympify(xhi)
    ylo = sp.sympify(ylo)
    yhi = sp.sympify(yhi)
    data = _extract_rel_bound(cond)
    if data is not None:
        op, lhs, rhs = data
        if lhs == x and not rhs.free_symbols:
            reg = _restrict_interval(BoxRegion(((x, xlo, xhi),)), cond)
            if reg is None:
                return None
            x, xlo, xhi = reg.ranges[0]
            return BoxRegion(((x, xlo, xhi), (y, ylo, yhi)))
        if lhs == y and not rhs.free_symbols:
            reg = _restrict_interval(BoxRegion(((y, ylo, yhi),)), cond)
            if reg is None:
                return None
            y, ylo, yhi = reg.ranges[0]
            return BoxRegion(((x, xlo, xhi), (y, ylo, yhi)))
        if lhs == y and (rhs.free_symbols <= {x}):
            location = _classify_graph_clip(rhs, x, xlo, xhi, ylo, yhi)
            if op in ("<", "<="):
                if location == "above":
                    return region
                if location == "inside":
                    return GraphRegion(((y, ylo, rhs), (x, xlo, xhi)), outer_var=x, inner_var=y)
                return None
            if op in (">", ">="):
                if location == "below":
                    return region
                if location == "inside":
                    return GraphRegion(((y, rhs, yhi), (x, xlo, xhi)), outer_var=x, inner_var=y)
                return None
        if rhs == y and (lhs.free_symbols <= {x}):
            location = _classify_graph_clip(lhs, x, xlo, xhi, ylo, yhi)
            if op in ("<", "<="):
                if location == "below":
                    return region
                if location == "inside":
                    return GraphRegion(((y, lhs, yhi), (x, xlo, xhi)), outer_var=x, inner_var=y)
                return None
            if op in (">", ">="):
                if location == "above":
                    return region
                if location == "inside":
                    return GraphRegion(((y, ylo, lhs), (x, xlo, xhi)), outer_var=x, inner_var=y)
                return None
        r2 = sp.expand(x**2 + y**2)
        if lhs == r2 and not rhs.free_symbols:
            if op in ("<", "<="):
                if _box_inside_disk(xlo, xhi, ylo, yhi, rhs):
                    return region
                radius = sp.sqrt(rhs)
                if _box_contains_disk(xlo, xhi, ylo, yhi, radius):
                    return DiskRegion(
                        (
                            (y, -sp.sqrt(rhs - x**2), sp.sqrt(rhs - x**2)),
                            (x, -radius, radius),
                        ),
                        radius=radius,
                    )
                return None
            # A disk complement inside a box is non-convex and cannot be
            # represented by one primitive region, so this shortcut declines it.
            if op in (">", ">="):
                return None
        if rhs == r2 and not lhs.free_symbols and op in (">", ">="):
            if _box_inside_disk(xlo, xhi, ylo, yhi, lhs):
                return region
            radius = sp.sqrt(lhs)
            if _box_contains_disk(xlo, xhi, ylo, yhi, radius):
                return DiskRegion(
                    (
                        (y, -sp.sqrt(lhs - x**2), sp.sqrt(lhs - x**2)),
                        (x, -radius, radius),
                    ),
                    radius=radius,
                )
            return None
    return None


def _provably_disjoint(left: Region, right: Region) -> bool:
    """Prove disjointness for simple pieces; uncertainty is treated as failure."""
    overlap = _has_positive_overlap(left, right)
    if overlap is False:
        return True
    if overlap is True:
        return False
    if left.variables != right.variables:
        return False
    if len(left.ranges) != len(right.ranges) or not left.ranges:
        return False
    for lrange, rrange in zip(left.ranges, right.ranges, strict=True):
        _, llo, lhi = lrange
        _, rlo, rhi = rrange
        if _is_definitely_true(sp.sympify(lhi) <= sp.sympify(rlo)):
            return True
        if _is_definitely_true(sp.sympify(rhi) <= sp.sympify(llo)):
            return True
    return False


def restrict_region(region: Region, cond: sp.Expr) -> Region | None:
    """Restrict a supported region by a simple Boolean condition.

    This is intentionally conservative and only supports conditions that can be
    represented by the region model.
    """
    cond = sp.sympify(cond)
    if cond in (True, sp.true):
        return region
    if cond in (False, sp.false):
        return None
    if isinstance(cond, sp.Or):
        pieces = []
        for arg in cond.args:
            reg = restrict_region(region, arg)
            if reg is not None:
                pieces.append(reg)
        if not pieces:
            return None
        if len(pieces) == 1:
            return pieces[0]
        for i, left in enumerate(pieces):
            for right in pieces[i + 1 :]:
                if not _provably_disjoint(left, right):
                    return None
        return UnionRegion(pieces=tuple(pieces))
    if isinstance(cond, sp.And):
        cur = region
        for arg in cond.args:
            cur = restrict_region(cur, arg)
            if cur is None:
                return None
        return cur
    if isinstance(region, BoxRegion):
        if len(region.ranges) == 1:
            return _restrict_interval(region, cond)
        if len(region.ranges) == 2:
            return _restrict_box_2d(region, cond)
    if isinstance(region, IteratedRegion) and not isinstance(
        region, (SimplexRegion, GraphRegion, DiskRegion, BallRegion)
    ):
        box_like = BoxRegion(tuple(region.ranges))
        reg = restrict_region(box_like, cond)
        if reg is not None:
            return reg
    return None


def region_from_ranges(ranges, *, structural_order: str = "auto") -> Region:
    """Classify nested bounds into a structured region.

    ``structural_order='auto'`` preserves the flexible standalone classifier.
    ``'inner-first'`` is strict and is used by :func:`multiple_integrate` so that
    classification never reinterprets an invalid iterated-integral order.
    """
    norm = _normalize_ranges_input(ranges)
    if isinstance(norm, Region):
        return norm
    ranges = norm

    simplex = match_standard_simplex(ranges, structural_order=structural_order)
    if simplex is not None:
        return simplex
    disk = match_standard_disk(ranges, structural_order=structural_order)
    if disk is not None:
        return disk
    ball = match_standard_ball(ranges, structural_order=structural_order)
    if ball is not None:
        return ball
    ellipsoid = match_standard_ellipsoid(ranges, structural_order=structural_order)
    if ellipsoid is not None:
        return ellipsoid
    affine_simplex = match_affine_simplex(ranges, structural_order=structural_order)
    if affine_simplex is not None:
        return affine_simplex
    graph = match_graph_region(ranges, structural_order=structural_order)
    if graph is not None:
        return graph

    vars_set = {r[0] for r in ranges}
    is_box = True
    for _, lo, hi in ranges:
        lo_s = sp.sympify(lo)
        hi_s = sp.sympify(hi)
        if (lo_s.free_symbols | hi_s.free_symbols) & vars_set:
            is_box = False
            break
    if is_box:
        return BoxRegion(tuple(ranges))
    return IteratedRegion(tuple(ranges))
