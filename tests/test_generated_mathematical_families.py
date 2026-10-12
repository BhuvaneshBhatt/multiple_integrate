"""Bounded generated singular, moment, affine and parameter-cell contracts."""

import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from multiple_integrate import (
    HadamardFinitePart,
    ParameterCell,
    clear_cache,
    integrate_cells,
    integrate_result,
    regularized_result,
    verify_regularization_result,
)

x, y, a = sp.symbols("x y a", real=True)
small = st.integers(1, 3)
shift = st.integers(-2, 2)


@settings(max_examples=12, deadline=None, derandomize=True)
@given(order=small, width=small, center=shift)
def test_generated_endpoint_vanishing(order, width, center):
    clear_cache()
    t = x - center
    density = sp.Mul(t**order * (1 + t), t ** (-order - sp.Rational(1, 2)), evaluate=False)
    result = integrate_result(density, (x, center, center + width))
    expected = 2 * sp.sqrt(width) + sp.Rational(2, 3) * width ** sp.Rational(3, 2)
    assert sp.simplify(result.value - expected) == 0
    assert result.absolute_convergent is True


@settings(max_examples=12, deadline=None, derandomize=True)
@given(center=shift, rate=small, degree=st.sampled_from([3, 4, 6]), power=st.integers(0, 2))
def test_generated_shifted_gamma_moments(center, rate, degree, power):
    clear_cache()
    t = x - center
    result = integrate_result(t**power * sp.exp(-rate * t**degree), (x, center, sp.oo))
    exponent = sp.Rational(power + 1, degree)
    expected = sp.gamma(exponent) / (degree * sp.Integer(rate) ** exponent)
    assert sp.simplify(result.value - expected) == 0
    assert result.absolute_convergent is True


@settings(max_examples=12, deadline=None, derandomize=True)
@given(cx=shift, cy=shift, sx=small, sy=small)
def test_generated_affine_jacobian(cx, cy, sx, sy):
    density = ((x - cx) / sx) ** 2 * ((y - cy) / sy)
    result = integrate_result(density, (x, cx, cx + sx), (y, cy, cy + sy))
    assert result.value == sp.Rational(sx * sy, 6)
    assert result.complete and result.absolute_convergent is True


@settings(max_examples=10, deadline=None, derandomize=True)
@given(power=st.integers(0, 3), width=small)
def test_generated_oriented_parameter_cells(power, width):
    result = integrate_cells(
        x**power,
        (x, 0, a),
        parameters=(a,),
        cells=(ParameterCell(a < 0), ParameterCell(sp.Eq(a, 0), {a: 0}), ParameterCell(a > 0)),
    )
    assert result.complete and result.coverage[0].complete
    for fiber in (-width, 0, width):
        assert result.value.subs(a, fiber) == sp.Rational(fiber ** (power + 1), power + 1)
    assert result.absolute_convergent is True


@settings(max_examples=8, deadline=None, derandomize=True)
@given(left=small, right=small, bottom=small, top=small)
def test_generated_shared_singular_intersection(left, right, bottom, top):
    result = regularized_result(
        (x * y) ** -2,
        (x, -left, right),
        (y, -bottom, top),
        regularization=HadamardFinitePart(hypersurface=x * y),
    )
    expected = (sp.Rational(1, left) + sp.Rational(1, right)) * (
        sp.Rational(1, bottom) + sp.Rational(1, top)
    )
    assert result.value == expected
    assert verify_regularization_result(result).valid
    assert result.ordinary_convergent is not True
