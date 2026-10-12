"""Independent series and cutoff integrals; numerics do not certify existence."""

import mpmath as mp
import pytest
import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    IntegrationOutcome,
    integrate_result,
    regularized_result,
)
from multiple_integrate.selected_families import selected_interval_rule

x, y = sp.symbols("x y", real=True)


@pytest.mark.parametrize("frequency", [1, 2, 3])
def test_reciprocal_oscillation_against_integrated_sine_series(frequency):
    result = integrate_result(sp.sin(frequency / x) / x, (x, 0, 1))
    assert result.ordinary_convergent is True and result.absolute_convergent is False
    # Independently integrate the entire sine series from 0 to frequency;
    # the classical Dirichlet integral supplies the remaining pi/2 constant.
    for digits in (35, 65):
        with mp.workdps(digits):
            sine_integral = sum(
                (-1) ** k
                * mp.mpf(frequency) ** (2 * k + 1)
                / ((2 * k + 1) * mp.factorial(2 * k + 1))
                for k in range(80)
            )
            expected = mp.pi / 2 - sine_integral
            actual = mp.mpf(str(result.value.evalf(digits)))
            assert abs(actual - expected) < mp.mpf(10) ** (-digits + 5)


@pytest.mark.slow
def test_improper_tail_against_direct_oscillatory_quadrature():
    result = integrate_result(sp.sin(1 / x) / x, (x, 0, 1))
    for digits in (25, 40):
        with mp.workdps(digits):
            expected = mp.quadosc(lambda t: mp.sin(t) / t, [1, mp.inf], omega=1)
            assert abs(mp.mpf(str(result.value.evalf(digits))) - expected) < mp.mpf(10) ** (
                -digits + 5
            )
    assert result.absolute_convergent is False


@pytest.mark.parametrize("width,height", [(1, 1), (2, 3), (3, 2)])
def test_shared_cutoff_constant_against_direct_quadrature(width, height):
    result = regularized_result(
        1 / (x * y),
        (x, 0, width),
        (y, 0, height),
        regularization=HadamardFinitePart(hypersurface=x * y),
    )
    assert result.outcome is IntegrationOutcome.FINITE
    # Directly integrate y from epsilon/x to height, then x from epsilon/height.
    # Subtract the divergent terms derived by hand, without replay helpers.
    for digits in (35, 65):
        with mp.workdps(digits):
            eps = mp.mpf("1e-8")
            wh = mp.mpf(width * height)
            # v=log(height*x/epsilon) converts the cutoff to integral v dv.
            cutoff = mp.quad(lambda v: v, [0, mp.log(wh / eps)])
            constant = cutoff - mp.log(eps) ** 2 / 2 + mp.log(wh) * mp.log(eps)
            actual = mp.mpf(str(result.value.evalf(digits)))
            assert abs(actual - constant) < mp.mpf(10) ** (-digits + 5)
    assert result.ordinary_convergent is not True


@pytest.mark.parametrize("lower", [3, 5])
@pytest.mark.parametrize("width", [sp.Rational(1, 10**4), sp.Rational(1, 10**12)])
def test_short_gamma_tail_difference_precision_escalation(lower, width):
    _, value = selected_interval_rule(sp.exp(-((x - 1) ** 4)), x, lower, lower + width)
    errors = []
    for digits in (40, 70):
        with mp.workdps(digits):
            h = mp.mpf(width.p) / width.q
            base = (lower - 1) ** 4
            expected = (
                h
                * mp.exp(-base)
                * mp.quad(lambda u, base=base, h=h: mp.exp(base - (lower + h * u - 1) ** 4), [0, 1])
            )
            actual = mp.mpf(str(value.evalf(digits)))
            error = abs(actual / expected - 1)
            errors.append(error)
            assert error < mp.mpf(10) ** (-digits + 8)
    assert errors[1] < errors[0] or errors[0] == 0
