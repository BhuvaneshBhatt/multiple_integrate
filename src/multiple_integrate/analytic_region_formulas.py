"""Reusable exact formulas for structured integration domains."""

from __future__ import annotations

import sympy as sp


def _clean_expr(expr):
    expr = sp.sympify(expr)
    try:
        return sp.simplify(expr)
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError):
        return expr


def _split_dependence(
    expr: sp.Expr, radial_vars: tuple[sp.Symbol, ...], angular_vars: tuple[sp.Symbol, ...]
):
    expr = _clean_expr(expr)
    radial_set = set(radial_vars)
    angular_set = set(angular_vars)
    if expr.is_Mul:
        radial = sp.Integer(1)
        angular = sp.Integer(1)
        for factor in expr.args:
            symbols = factor.free_symbols
            if symbols & radial_set and not symbols & angular_set:
                radial *= factor
            elif symbols & angular_set and not symbols & radial_set:
                angular *= factor
            elif not symbols & (radial_set | angular_set):
                radial *= factor
            else:
                return None
        return _clean_expr(radial), _clean_expr(angular)
    symbols = expr.free_symbols
    if symbols & radial_set and not symbols & angular_set:
        return _clean_expr(expr), sp.Integer(1)
    if symbols & angular_set and not symbols & radial_set:
        return sp.Integer(1), _clean_expr(expr)
    if not symbols & (radial_set | angular_set):
        return _clean_expr(expr), sp.Integer(1)
    return None


def _ball_poly_moment(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    radius: sp.Expr,
    center: tuple[sp.Expr, ...],
) -> sp.Expr | None:
    dim = len(variables)
    local_vars = tuple(sp.Dummy(f"ball_u{i}", real=True) for i in range(dim))
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
        numerator = sp.prod(sp.gamma(sp.Rational(exponent + 1, 2)) for exponent in monomial)
        denominator = sp.gamma(sp.Rational(degree + dim, 2) + 1)
        total += coefficient * sp.sympify(radius) ** (degree + dim) * numerator / denominator
    return sp.simplify(total)


def _radial_profile(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    center: tuple[sp.Expr, ...],
    scales: tuple[sp.Expr, ...] | None = None,
    *,
    radius_squared: sp.Symbol | None = None,
) -> sp.Expr | None:
    scales = scales or tuple(sp.Integer(1) for _ in variables)
    if radius_squared is None:
        radius_squared = sp.Dummy("rho2", nonnegative=True, real=True)
    profiles = []
    for variable, scale, shift in zip(variables, scales, center, strict=True):
        substitutions = dict(zip(variables, center, strict=True))
        substitutions[variable] = shift + sp.sympify(scale) * sp.sqrt(radius_squared)
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
    if sp.simplify(sp.sympify(expr) - _clean_expr(profile.subs(radius_squared, rho2))) != 0:
        return None
    return profile


def _radial_ball_integral(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    center: tuple[sp.Expr, ...],
    radius: sp.Expr,
    scales: tuple[sp.Expr, ...] | None = None,
) -> sp.Expr | None:
    scales = scales or tuple(sp.Integer(1) for _ in variables)
    radius_squared = sp.Dummy("rho2", nonnegative=True, real=True)
    profile = _radial_profile(expr, variables, center, scales, radius_squared=radius_squared)
    if profile is None:
        return None
    dim = len(variables)
    radial_var = sp.Dummy("r", nonnegative=True, real=True)
    scale = sp.prod(sp.Abs(axis) for axis in scales)
    sphere_area = 2 * sp.pi ** sp.Rational(dim, 2) / sp.gamma(sp.Rational(dim, 2))
    integrand = _clean_expr(
        profile.subs(radius_squared, radial_var**2) * scale * sphere_area * radial_var ** (dim - 1)
    )
    return sp.integrate(integrand, (radial_var, 0, radius))
