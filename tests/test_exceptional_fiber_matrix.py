"""Exact equality fibers are first-class contracts, not generic sample points."""

import pytest
import sympy as sp

from multiple_integrate import (
    IntegrationOutcome,
    IntegrationPolicy,
    ParameterCell,
    integrate_cells,
    integrate_result,
)

x, a, b, p = sp.symbols("x a b p", real=True)
policy = IntegrationPolicy(max_singular_queries=0)


@pytest.mark.parametrize("power", [-2, -1, sp.Rational(-1, 2), 0, 2])
@pytest.mark.parametrize("amplitude", [-2, 0, 3])
def test_endpoint_threshold_and_zero_amplitude(power, amplitude):
    result = integrate_result(
        b * x**p, (x, 0, 1), assumptions=sp.And(sp.Eq(p, power), sp.Eq(b, amplitude)), policy=policy
    )
    if amplitude == 0 or power > -1:
        assert result.outcome is IntegrationOutcome.FINITE
        assert (
            sp.simplify(
                result.value.subs({p: power, b: amplitude})
                - (0 if amplitude == 0 else sp.Integer(amplitude) / (power + 1))
            )
            == 0
        )
        assert result.absolute_convergent is True
    else:
        assert result.outcome is IntegrationOutcome.DIVERGENT
        assert result.ordinary_convergent is False


@pytest.mark.parametrize("rate", [-1, 0, 1])
@pytest.mark.parametrize("amplitude", [0, 2])
def test_decay_equality_fiber(rate, amplitude):
    result = integrate_result(
        b * sp.exp(-a * x),
        (x, 0, sp.oo),
        assumptions=sp.And(sp.Eq(a, rate), sp.Eq(b, amplitude)),
        policy=policy,
    )
    if amplitude == 0 or rate > 0:
        assert result.finite and result.value.subs({a: rate, b: amplitude}) == (
            0 if amplitude == 0 else sp.Rational(amplitude, rate)
        )
        assert result.absolute_convergent is True
    else:
        assert result.outcome is IntegrationOutcome.DIVERGENT


@pytest.mark.parametrize("coefficient", [1 / a, sp.log(a), sp.gamma(a)])
def test_undefined_pole_fiber_is_not_divergence(coefficient):
    result = integrate_cells(
        coefficient * x**-2,
        (x, 0, 1),
        parameters=(a,),
        assumptions=sp.Eq(a, 0),
        cells=(ParameterCell(sp.Eq(a, 0)),),
        policy=policy,
    )
    assert result.cases[0].outcome is IntegrationOutcome.UNKNOWN
    assert not result.complete
    assert result.ordinary_convergent is None


@pytest.mark.parametrize("fiber", [-1, 0, 1, 2])
@pytest.mark.parametrize("dimension", [0, 1])
def test_coincident_boundaries_empty_point_interval(fiber, dimension):
    # At a=0 the closed interval collapses to a point; count and length differ.
    result = integrate_cells(
        1,
        region=sp.And(x >= 0, x <= a),
        variables=(x,),
        parameters=(a,),
        measure_dimension=dimension,
        assumptions=sp.Eq(a, fiber),
        cells=(ParameterCell(sp.Eq(a, fiber), {a: fiber}),),
        policy=policy,
    )
    if dimension == 1:
        assert result.value.subs(a, fiber) == (fiber if fiber > 0 else 0)
    elif fiber <= 0:
        assert result.value.subs(a, fiber) == (1 if fiber == 0 else 0)
    else:
        # Zero-dimensional counting measure on a positive-dimensional interval
        # is outside this region-measure contract, not an invented finite count.
        assert not result.finite


def test_coverage_can_be_complete_while_pole_fiber_is_unknown():
    result = integrate_cells(
        1 / a,
        (x, 0, 1),
        parameters=(a,),
        cells=(ParameterCell(a < 0), ParameterCell(sp.Eq(a, 0)), ParameterCell(a > 0)),
        policy=policy,
    )
    assert result.coverage[0].complete
    assert not result.complete and result.absolute_convergent is None
    assert result.value.subs(a, 2) == sp.Rational(1, 2)
    assert result.value.subs(a, 0).has(sp.Integral)
