"""Structured integration domains and exact analytic shortcuts."""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from multiple_integrate.analytic_region_formulas import (
    _ball_poly_moment,
    _clean_expr,
    _radial_ball_integral,
    _split_dependence,
)

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
    """Cartesian product of independent one-dimensional intervals."""

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
            vol *= sp.Abs(hi_s - lo_s)
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
    """Region represented by nested inner-first integration ranges."""

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
    """Two-dimensional region bounded by graphs over one outer variable."""

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
        """Reverse proved ordered finite affine bounds using exact comparisons."""
        from functools import cmp_to_key

        data = self._linear_data()
        if data is None:
            return None
        x, y, a, b, m1, c1, m2, c2 = data
        if any(v.is_real is not True or v.is_finite is not True for v in data[2:]):
            return None
        if sp.simplify(b - a).is_positive is not True or any(
            sp.simplify((m2 - m1) * endpoint + c2 - c1).is_nonnegative is not True
            for endpoint in (a, b)
        ):
            # Crossing or reversed limits carry orientation, which a geometric
            # decomposition of only the positive pieces cannot preserve.
            return None

        def compare(first, second):
            delta = sp.simplify(first - second)
            if delta == 0:
                return 0
            if delta.is_positive is True:
                return 1
            if delta.is_negative is True:
                return -1
            raise ValueError("affine boundary order is not established")

        pts = [
            sp.simplify(m * endpoint + c) for m, c in ((m1, c1), (m2, c2)) for endpoint in (a, b)
        ]
        lower_cands, upper_cands = [a], [b]
        try:
            for slope, offset, lower in ((m1, c1, True), (m2, c2, False)):
                direction = compare(slope, 0)
                if direction == 0:
                    continue
                inverse = sp.simplify((y - offset) / slope)
                target = lower_cands if (direction < 0) == lower else upper_cands
                target.append(inverse)
            pts = sorted(set(pts), key=cmp_to_key(compare))
            pieces = []
            for left, right in zip(pts, pts[1:], strict=False):
                if compare(left, right) == 0:
                    continue
                mid = sp.simplify((left + right) / 2)

                def choose(candidates, direction, midpoint=mid):
                    best = candidates[0]
                    for candidate in candidates[1:]:
                        if (
                            compare(candidate.subs(y, midpoint), best.subs(y, midpoint))
                            == direction
                        ):
                            best = candidate
                    return sp.simplify(best)

                xlo, xhi = choose(lower_cands, 1), choose(upper_cands, -1)
                if compare(xlo.subs(y, mid), xhi.subs(y, mid)) <= 0:
                    pieces.append([(x, xlo, xhi), (y, left, right)])
            return pieces or None
        except (TypeError, ValueError, NotImplementedError):
            return None


@dataclass(frozen=True)
class SimplexRegion(IteratedRegion):
    """Standard simplex represented in iterated-range form."""

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
    """Affine image of a standard simplex with exact scale and shift data."""

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

    def _standard_simplex_pullback(self, expr: sp.Expr):
        """Return a fresh standard simplex and its Jacobian-weighted integrand."""
        vars_ = self.variables
        uvars = tuple(sp.Dummy(f"u{i}", real=True) for i in range(self.dimension))
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
        return simplex, transformed

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        simplex, transformed = self._standard_simplex_pullback(expr)
        return simplex.polynomial_moment(transformed)


@dataclass(frozen=True)
class DiskRegion(IteratedRegion):
    """Two-dimensional Euclidean disk with exact center and radius."""

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
        r = sp.Dummy("r", nonnegative=True, real=True)
        theta = sp.Dummy("theta", real=True)
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
    """Planar annulus centered at the origin."""

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
    """Euclidean ball in arbitrary finite dimension."""

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
        r = sp.Dummy("r", nonnegative=True, real=True)
        phi = sp.Dummy("phi", real=True)
        theta = sp.Dummy("theta", real=True)
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
    """Axis-aligned ellipsoid represented by center and semi-axis lengths."""

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
        local_vars = tuple(sp.Dummy(f"u{i}", real=True) for i in range(len(variables)))
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
    """Region between two concentric spheres in arbitrary finite dimension."""

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
