"""Recognition of structured regions from symbolic integration ranges."""

from __future__ import annotations

from functools import lru_cache

import sympy as sp

from multiple_integrate.region_types import (
    AffineSimplexRegion,
    BallRegion,
    BoxRegion,
    DiskRegion,
    EllipsoidRegion,
    GraphRegion,
    IteratedRegion,
    Region,
    SimplexRegion,
    _normalize_ranges_input,
)

_REGION_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


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

    ``inner-first`` is strict and is used by :func:`integrate`; ``auto``
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
                if sc_i == 0 or sc_i.free_symbols:
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


def region_from_ranges(ranges, *, structural_order: str = "auto") -> Region:
    """Classify nested bounds into a structured region.

    ``structural_order='auto'`` preserves the flexible standalone classifier.
    ``'inner-first'`` is strict and is used by :func:`integrate` so that
    classification never reinterprets an invalid iterated-integral order.
    """
    norm = _normalize_ranges_input(ranges)
    if isinstance(norm, Region):
        return norm
    return _recognize_ranges(tuple(tuple(r) for r in norm), structural_order)


@lru_cache(maxsize=512)
def _recognize_ranges(ranges, structural_order):
    """Cache immutable geometric classification, independently of densities."""
    ranges = list(ranges)

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
