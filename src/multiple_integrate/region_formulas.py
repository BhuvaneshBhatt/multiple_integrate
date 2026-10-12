"""Boolean region formulas and exact restriction operations."""

from __future__ import annotations

import sympy as sp
from sympy.core.relational import Relational

from multiple_integrate.region_types import (
    BallRegion,
    BoxRegion,
    DiskRegion,
    GraphRegion,
    IteratedRegion,
    Region,
    SimplexRegion,
    UnionRegion,
    _has_positive_overlap,
    _is_definitely_true,
)

_REGION_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


def Boole(cond: sp.Expr) -> sp.Piecewise:
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

    This conversion only supports conditions that can represented by the region model.
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
