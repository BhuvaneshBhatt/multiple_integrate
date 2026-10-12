"""Exact analytic strategies used by the integration planner."""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp
from sympy import (
    Dummy,
    det,
    gamma,
    oo,
    pi,
    sqrt,
)
from sympy.matrices import Matrix

from multiple_integrate.assumptions import _ask, _integrate
from multiple_integrate.recognition import (
    _is_even_function,
    _quadratic_coefficients,
)

_STRATEGY_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


def _integrate_quadratic(
    f_outer: Callable, A_mat: Matrix, b_vec: Matrix, c_val: sp.Expr, n: int, opts: dict
) -> sp.Expr | None:
    """
    ∫_{ℝⁿ} f(xᵀAx + b·x + c) dx  via ellipsoid surface-area layer-cake.
    Requires A positive definite.
    """
    # This is a real ellipsoid pushforward, not a contour-shift theorem.
    # Complex translations can cross poles of the outer function, and positive
    # eigenvalues alone do not certify a complex matrix's real geometry.
    coefficients = (*A_mat, *b_vec, sp.sympify(c_val))
    if not all(_ask(sp.Q.real(entry) & sp.Q.finite(entry)) is True for entry in coefficients):
        return None
    if A_mat != A_mat.T:
        return None
    try:
        A_inv = A_mat.inv()
    except _STRATEGY_ERRORS:
        return None
    try:
        evs = A_mat.eigenvals()
        if not evs or any(_ask(sp.Q.positive(ev)) is not True for ev in evs):
            return None
    except _STRATEGY_ERRORS:
        return None
    det_A = det(A_mat)
    if det_A == 0:
        return None

    y_min = c_val - (b_vec.T * A_inv * b_vec)[0, 0] / 4
    y = Dummy("y")
    fac = pi ** sp.Rational(n, 2) / (sqrt(det_A) * gamma(sp.Rational(n, 2) + 1))
    surface = n * sp.Rational(1, 2) * (y - y_min) ** (sp.Rational(n, 2) - 1)
    result = _integrate(fac * surface * f_outer(y), (y, y_min, oo), **opts)
    return None if result.has(sp.Integral) else result


def _try_fullspace_quadratic(f_outer, g, vars_, ranges, opts):
    if not all(r[1] == -oo and r[2] == oo for r in ranges):
        return None
    if not any(sp.degree(g, v) == 2 for v in vars_ if v in g.free_symbols):
        return None
    try:
        c, b_vec, A = _quadratic_coefficients(g, vars_)
    except _STRATEGY_ERRORS:
        return None
    return _integrate_quadratic(f_outer, A, b_vec, c, len(vars_), opts)


def _try_halfspace_quadratic(f_outer, g, vars_, ranges, opts):
    half = sum(1 for r in ranges if r[1] == 0 and r[2] == oo)
    full = sum(1 for r in ranges if r[1] == -oo and r[2] == oo)
    if half + full != len(vars_):
        return None
    try:
        if sp.Poly(g, *vars_).total_degree() != 2:
            return None
        c, b_vec, A = _quadratic_coefficients(g, vars_)
    except _STRATEGY_ERRORS:
        return None
    for r in ranges:
        if r[1] == 0 and r[2] == oo and not _is_even_function(f_outer(g), r[0]):
            return None
    full_result = _integrate_quadratic(f_outer, A, b_vec, c, len(vars_), opts)
    if full_result is None:
        return None
    return full_result / sp.Integer(2) ** half


def _try_gauss_linear_map(
    expr: sp.Expr, parsed_ranges: list[tuple], assumptions=None
) -> sp.Expr | None:
    vars_ = tuple(r[0] for r in parsed_ranges)
    if not vars_ or not all(
        sp.sympify(lo) == -sp.oo and sp.sympify(hi) == sp.oo for _, lo, hi in parsed_ranges
    ):
        return None
    expr = sp.sympify(expr)
    if expr.func != sp.exp or len(expr.args) != 1:
        return None
    q = sp.expand(expr.args[0])
    if any(
        v in sp.sympify(lo).free_symbols or v in sp.sympify(hi).free_symbols
        for v, lo, hi in parsed_ranges
    ):
        return None
    H = sp.hessian(q, vars_)
    A = sp.simplify(-H / 2)
    try:
        if any(entry.free_symbols & set(vars_) for entry in A):
            return None
    except _STRATEGY_ERRORS:
        return None
    quad = sp.expand((sp.Matrix(vars_).T * A * sp.Matrix(vars_))[0])
    linear_part = sp.expand(q + quad)
    b_entries = []
    for v in vars_:
        dv = sp.diff(linear_part, v)
        if dv.free_symbols & set(vars_):
            return None
        b_entries.append(sp.simplify(dv))
    bvec = sp.Matrix(b_entries)
    c = sp.simplify(linear_part - sum(b * v for b, v in zip(b_entries, vars_, strict=True)))
    if c.free_symbols & set(vars_):
        return None
    # SymPy also calls complex accretive matrices positive definite. For
    # those matrices sqrt(det(A)) can have the opposite sign to the analytic
    # Gaussian branch (already in dimension three). A diagonal complex form
    # instead has independent Gaussian factors, each with its own proved
    # right-half-plane square-root branch. General complex forms fall back.
    real_form = all(_ask(sp.Q.real(entry), assumptions) is True for entry in A)
    if not real_form and not (
        A.is_diagonal()
        and all(_ask(sp.Q.positive(sp.re(entry)), assumptions) is True for entry in A.diagonal())
    ):
        return None
    detA = sp.simplify(A.det())
    if detA == 0:
        return None
    try:
        if A.is_positive_definite is False:
            return None
    except _STRATEGY_ERRORS:
        pass
    if A.is_positive_definite is not True:
        # Avoid returning branch-sensitive square roots unless positivity is clear.
        return None
    try:
        shift_term = sp.simplify((bvec.T * A.LUsolve(bvec))[0] / 4)
    except _STRATEGY_ERRORS:
        return None
    return sp.simplify(
        sp.pi ** (sp.Rational(len(vars_), 2))
        * sp.exp(c + shift_term)
        / (sp.sqrt(detA) if real_form else sp.prod(sp.sqrt(entry) for entry in A.diagonal()))
    )
