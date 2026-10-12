"""Exact unit-sphere moments used by radial integration strategies."""

from __future__ import annotations

import sympy as sp


def sphere_surface_area(dimension: int) -> sp.Expr:
    """Surface measure of ``S^(dimension-1)`` embedded in ``R^dimension``."""
    if dimension <= 0:
        raise ValueError("sphere dimension must be positive")
    half = sp.Rational(dimension, 2)
    return sp.simplify(2 * sp.pi**half / sp.gamma(half))


def sphere_monomial_integral(exponents: tuple[int, ...]) -> sp.Expr:
    """Integrate a coordinate monomial over the unit sphere exactly."""
    if not exponents:
        raise ValueError("sphere monomial requires at least one exponent")
    if any(exponent < 0 for exponent in exponents):
        raise ValueError("sphere monomial exponents must be nonnegative")
    if any(exponent % 2 for exponent in exponents):
        return sp.Integer(0)
    numerator = sp.Integer(2)
    for exponent in exponents:
        numerator *= sp.gamma(sp.Rational(exponent + 1, 2))
    denominator = sp.gamma(sp.Rational(sum(exponents) + len(exponents), 2))
    return sp.simplify(numerator / denominator)


def radial_sphere_reduction(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    center: tuple[sp.Expr, ...],
    radial: sp.Symbol,
    *,
    axes: tuple[sp.Expr, ...] | None = None,
    affine_matrix: sp.MatrixBase | None = None,
) -> sp.Expr | None:
    """Reduce a radial polynomial/rational integrand to one radial variable.

    The recognized denominator is a scalar multiple of an integer power of the
    local squared radius.  The numerator may be any polynomial.  Angular
    monomials are integrated using exact unit-sphere moments, so odd terms
    vanish without introducing hyperspherical angles.
    """
    dimension = len(variables)
    if dimension < 2 or len(center) != dimension:
        return None
    if axes is not None and affine_matrix is not None:
        return None
    if affine_matrix is None:
        axes = tuple(sp.Integer(1) for _ in variables) if axes is None else tuple(axes)
        if len(axes) != dimension:
            return None
        affine = sp.ImmutableMatrix(sp.diag(*axes))
    else:
        affine = sp.ImmutableMatrix(affine_matrix)
        if affine.shape != (dimension, dimension):
            return None

    locals_ = tuple(sp.Dummy(f"pv_u{i}", real=True) for i in range(dimension))
    local_vector = sp.ImmutableMatrix(locals_)
    transformed = sp.ImmutableMatrix(center) + affine * local_vector
    substitutions = {
        variable: sp.simplify(transformed[index]) for index, variable in enumerate(variables)
    }
    local_expr = sp.cancel(sp.sympify(expr).subs(substitutions))
    numerator, denominator = sp.fraction(local_expr)
    try:
        numerator_poly = sp.Poly(numerator, *locals_)
        denominator_poly = sp.Poly(denominator, *locals_)
    except sp.PolynomialError:
        return None

    degree = denominator_poly.total_degree()
    if degree < 0 or degree % 2:
        return None
    power = degree // 2
    radius_sq = sp.Add(*(local**2 for local in locals_))
    denominator_scale = sp.simplify(denominator / radius_sq**power) if power else denominator
    if any(denominator_scale.has(local) for local in locals_):
        return None
    if denominator_scale == 0:
        return None

    angular_sum = sp.Integer(0)
    for exponents, coefficient in numerator_poly.terms():
        angular = sphere_monomial_integral(tuple(int(value) for value in exponents))
        if angular == 0:
            continue
        angular_sum += coefficient * radial ** sum(exponents) * angular

    scale = sp.Abs(sp.simplify(affine.det()))
    return sp.simplify(
        scale * radial ** (dimension - 1 - 2 * power) * angular_sum / denominator_scale
    )


__all__ = [
    "radial_sphere_reduction",
    "sphere_linear_forms_integral",
    "sphere_monomial_integral",
    "sphere_surface_area",
]


def _pairings(items: tuple[int, ...]):
    if not items:
        yield ()
        return
    first = items[0]
    for offset in range(1, len(items)):
        second = items[offset]
        rest = items[1:offset] + items[offset + 1 :]
        for tail in _pairings(rest):
            yield ((first, second), *tail)


def sphere_linear_forms_integral(
    vectors: tuple[sp.MatrixBase | tuple[sp.Expr, ...], ...],
) -> sp.Expr:
    """Integrate a product of linear forms ``(v_i · omega)`` over a unit sphere.

    This is the isotropic tensor contraction formula.  It avoids expanding
    quadratic forms and tensor contractions into potentially huge coordinate
    monomial sums.
    """
    if not vectors:
        raise ValueError("at least one linear form is required")
    normalized = tuple(sp.ImmutableMatrix(vector) for vector in vectors)
    dimension = normalized[0].rows
    if dimension <= 0 or any(vector.shape != (dimension, 1) for vector in normalized):
        raise ValueError("all vectors must be same-dimensional column vectors")
    order = len(normalized)
    if order % 2:
        return sp.Integer(0)
    pair_count = order // 2
    denominator = sp.Integer(1)
    for index in range(pair_count):
        denominator *= dimension + 2 * index
    contraction = sp.Integer(0)
    for pairing in _pairings(tuple(range(order))):
        term = sp.Integer(1)
        for left, right in pairing:
            term *= (normalized[left].T * normalized[right])[0]
        contraction += term
    return sp.simplify(sphere_surface_area(dimension) * contraction / denominator)
