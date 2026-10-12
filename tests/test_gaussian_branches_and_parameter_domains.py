"""Independent regressions for Gaussian branches and exceptional cell fibers."""

import pytest
import sympy as sp

from multiple_integrate import (
    IntegrationOutcome,
    IntegrationPolicy,
    IntegrationStatus,
    ParameterCell,
    integrate_cells,
    integrate_result,
)
from multiple_integrate.quadratic_gaussian import _try_gauss_linear_map
from multiple_integrate.symbolic_geometry import symbolic_boxes

x, y, z, a = sp.symbols("x y z a", real=True)
policy = IntegrationPolicy(max_singular_queries=0)


@pytest.mark.parametrize("coefficient", [1 + 2 * sp.I, 1 - 2 * sp.I, 1 + 10 * sp.I])
def test_complex_diagonal_gaussian_preserves_independent_root_branches(coefficient):
    density = sp.exp(-coefficient * (x**2 + y**2 + z**2) + x + y + z)
    result = integrate_result(density, *((v, -sp.oo, sp.oo) for v in (x, y, z)), policy=policy)
    expected = (
        sp.pi ** sp.Rational(3, 2)
        * sp.exp(sp.Rational(3, 4) / coefficient)
        / sp.sqrt(coefficient) ** 3
    )
    assert sp.simplify(result.value / expected) == 1
    assert result.outcome is IntegrationOutcome.FINITE


def test_coupled_complex_gaussian_declines_unproved_determinant_branch():
    coefficient = 1 + 2 * sp.I
    density = sp.exp(-coefficient * (x**2 + y**2 + z**2) - x * y / 3 + x + y + z)
    assert _try_gauss_linear_map(density, [(v, -sp.oo, sp.oo) for v in (x, y, z)]) is None


@pytest.mark.parametrize("leading", [a, -a])
@pytest.mark.parametrize("power", [1, 2])
def test_weak_cell_cannot_fix_polynomial_degree(leading, power):
    assert (
        symbolic_boxes(leading * x**power >= 0, (x,), assumptions=a >= 0, bounds=[(x, -1, 1)])
        is None
    )


def test_degree_drop_cell_is_unknown_until_partitioned():
    request = dict(region=a * x >= 0, variables=(x,), bounds=[(x, -1, 1)], parameters=(a,))
    weak = integrate_cells(1, cells=[ParameterCell(a >= 0)], policy=policy, **request)
    assert weak.cases[0].status is IntegrationStatus.UNKNOWN
    split = integrate_cells(
        1,
        cells=[ParameterCell(a > 0), ParameterCell(sp.Eq(a, 0))],
        assumptions=a >= 0,
        policy=policy,
        **request,
    )
    assert split.complete is True
    assert split.value.subs(a, 1) == 1
    assert split.value.subs(a, 0) == 2


@pytest.mark.parametrize(
    "density", [sp.tan(a), sp.sec(a), sp.cot(a), sp.csc(a), sp.atanh(a), sp.acoth(a), sp.zeta(a)]
)
def test_unexcluded_function_poles_are_not_certified_finite(density):
    result = integrate_result(density, (x, 0, 1), assumptions=a >= 0, policy=policy)
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert result.complete is False
    assert result.ordinary_convergent is not True
    assert result.absolute_convergent is not True


@pytest.mark.parametrize(
    "density,condition",
    [
        (sp.tan(a), sp.Ne(sp.cos(a), 0)),
        (sp.sec(a), sp.Ne(sp.cos(a), 0)),
        (sp.cot(a), sp.Ne(sp.sin(a), 0)),
        (sp.csc(a), sp.Ne(sp.sin(a), 0)),
        (sp.atanh(a), a > 1),
        (sp.acoth(a), a > 1),
        (sp.zeta(a), a > 1),
        (sp.tanh(a), a >= 0),
        (sp.sech(a), a >= 0),
        (sp.coth(a), a > 0),
        (sp.csch(a), a > 0),
    ],
)
def test_explicit_function_domain_remains_supported(density, condition):
    assert integrate_result(density, (x, 0, 1), assumptions=condition, policy=policy).finite is True


@pytest.mark.parametrize("density", [1 / a, sp.log(a), sp.gamma(a), sp.tan(a)])
def test_moving_point_measure_obeys_parameter_domain(density):
    result = integrate_cells(
        density,
        region=sp.Eq(x, a),
        variables=(x,),
        parameters=(a,),
        cells=[ParameterCell(a >= 0)],
        measure_dimension="intrinsic",
        policy=policy,
    )
    assert result.cases[0].status is IntegrationStatus.UNKNOWN
    assert result.complete is False
    assert all(e.established is not True for e in result.convergence_evidence)


def test_finite_moving_point_retains_counting_measure():
    result = integrate_cells(
        1 / a,
        region=sp.Eq(x, a),
        variables=(x,),
        parameters=(a,),
        cells=[ParameterCell(a > 0)],
        assumptions=a > 0,
        measure_dimension="intrinsic",
        policy=policy,
    )
    assert result.complete is True
    assert result.cases[0].value == 1 / a


@pytest.mark.parametrize(
    "density",
    [
        sp.Ei(a),
        sp.Ci(a),
        sp.Chi(a),
        sp.bessely(0, a),
        sp.besselk(0, a),
        sp.hankel1(0, a),
        sp.hankel2(0, a),
    ],
)
def test_special_function_zero_poles_are_guarded(density):
    assert (
        integrate_result(density, (x, 0, 1), assumptions=a >= 0, policy=policy).outcome
        is IntegrationOutcome.UNKNOWN
    )
    assert integrate_result(density, (x, 0, 1), assumptions=a > 0, policy=policy).finite is True


def test_logarithmic_integral_excludes_one_but_allows_zero():
    assert (
        integrate_result(sp.li(a), (x, 0, 1), assumptions=a >= 0, policy=policy).outcome
        is IntegrationOutcome.UNKNOWN
    )
    assert integrate_result(sp.li(a), (x, 0, 1), assumptions=a < 1, policy=policy).finite is True
    assert integrate_result(sp.li(0), (x, 0, 1), policy=policy).value == 0


@pytest.mark.parametrize("function", [sp.atan, sp.acot])
def test_inverse_trigonometric_complex_poles(function):
    b = sp.Symbol("complex_parameter")
    assert (
        integrate_result(function(b), (x, 0, 1), policy=policy).outcome
        is IntegrationOutcome.UNKNOWN
    )
    assert integrate_result(function(a), (x, 0, 1), policy=policy).finite is True


def test_piecewise_moving_point_scopes_parameter_prerequisites():
    result = integrate_cells(
        sp.Piecewise((1 / a, a > 0), (0, True)),
        region=sp.Eq(x, a),
        variables=(x,),
        parameters=(a,),
        cells=[ParameterCell(a >= 0)],
        assumptions=a >= 0,
        measure_dimension="intrinsic",
        policy=policy,
    )
    assert result.complete is True
    assert result.value.subs(a, 0) == 0
    assert result.value.subs(a, 1) == 1


@pytest.mark.parametrize("coefficient", [1 / a, sp.log(a), sp.tan(a), sp.Ei(a)])
def test_symmetry_cannot_erase_density_parameter_poles(coefficient):
    result = integrate_result(x * coefficient, (x, -1, 1), assumptions=a >= 0, policy=policy)
    assert result.value == 0
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert result.complete is False


def test_symmetry_cannot_erase_undefined_parameter_bounds():
    result = integrate_result(x, (x, -1 / a, 1 / a), assumptions=a >= 0, policy=policy)
    assert result.value == 0
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert integrate_result(x, (x, -1 / a, 1 / a), assumptions=a > 0, policy=policy).finite is True


def test_valid_piecewise_density_and_coordinate_singularities_are_preserved():
    density = sp.Piecewise((x / a, a > 0), (0, True))
    result = integrate_result(density, (x, -1, 1), policy=policy)
    assert result.finite is True
    assert integrate_result(1 / sp.sqrt(x), (x, 0, 1), policy=policy).value == 2
    list_range = integrate_result(1 / sp.sqrt(x), [x, 0, 1], policy=policy)
    assert list_range.finite is True and list_range.value == 2


def test_moving_point_substitution_cannot_cancel_original_parameter_pole():
    result = integrate_cells(
        x / a,
        region=sp.Eq(x, a),
        variables=(x,),
        parameters=(a,),
        cells=[ParameterCell(a >= 0)],
        measure_dimension="intrinsic",
        policy=policy,
    )
    assert result.cases[0].value == 1
    assert result.cases[0].status is IntegrationStatus.UNKNOWN
