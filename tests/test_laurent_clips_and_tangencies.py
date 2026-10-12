"""Independent moment/cutoff oracles and broken-prerequisite checks."""

import re
from pathlib import Path

import mpmath as mp
import pytest
import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    HypersurfacePrincipalValue,
    IntegrationOutcome,
    IntegrationPolicy,
    ParameterCell,
    StrategyBudget,
    integrate_cells,
    integrate_result,
    regularized_result,
    verify_regularization_result,
)
from multiple_integrate.clipped_quadratic import integrate_clipped_quadratic_cell
from multiple_integrate.convergence_evidence import absolute_rule
from multiple_integrate.selected_families import selected_interval_rule
from multiple_integrate.tangent_cutoff import tangent_graph_cutoff

x, y, z, a = sp.symbols("x y z a", real=True)


@pytest.mark.parametrize(
    "amplitude", [(1 + x) / x**2, 2 / x**3 - 3 / x + x**2, (1 + x) ** 2 / x**3]
)
@pytest.mark.parametrize("bounds", [(1, 2), (1, sp.oo), (2, 1)])
def test_laurent_moments_independent_oracle(amplitude, bounds):
    density = amplitude * sp.exp(-2 * x**2)
    result = selected_interval_rule(density, x, *bounds)
    assert result is not None
    with mp.workdps(60):
        lo, hi = [mp.inf if v == sp.oo else v for v in bounds]
        oracle = mp.quad(sp.lambdify(x, density, "mpmath"), [lo, hi])
        assert abs(mp.mpf(str(result[1].evalf(55))) - oracle) < mp.mpf("1e-48")


def test_shifted_laurent_and_exceptional_cutoff():
    f = (1 + (x - 2)) / (x - 2) ** 2 * sp.exp(-((x - 2) ** 4))
    assert selected_interval_rule(f, x, 3, sp.oo) is not None
    assert selected_interval_rule(f, x, 2, sp.oo) is None
    assert selected_interval_rule(f, x, -sp.oo, sp.oo) is None
    assert selected_interval_rule(sp.exp(-x * x) / (x - 1), x, 2, 3) is None
    assert selected_interval_rule(sp.exp(-x * x) * (1 + x) ** 30 / x**2, x, 1, 2) is None


@pytest.mark.parametrize(
    "density,bounds,expected",
    [
        ((1 + x) / (x**3 * sp.log(x) ** 2), (2, sp.oo), True),
        ((1 + x) / (x * sp.log(x) ** 2), (2, sp.oo), None),
        ((1 + x) / (x * sp.log(x) ** 2), (0, sp.Rational(1, 2)), True),
        ((1 + x) / (x * sp.log(x)), (0, sp.Rational(1, 2)), None),
        ((1 + x) / (x * sp.log(x) ** 2), (0, 2), None),
        ((1 + (x - 2)) / ((x - 2) * sp.log(x - 2) ** 2), (2, sp.Rational(5, 2)), True),
        ((1 + (2 - x)) / ((2 - x) * sp.log(2 - x) ** 2), (sp.Rational(3, 2), 2), True),
        ((1 - sp.cos(x)) / x**2, (0, 1), True),
        ((1 - sp.cos(x)) / x**4, (0, 1), None),
        (sp.cos(x) ** sp.Rational(1, 2) / (1 + x * x), (-sp.oo, sp.oo), True),
    ],
)
def test_additional_scoped_comparisons(density, bounds, expected):
    assert absolute_rule(density, ((x, *bounds),))[0] is expected


def test_reversed_and_translated_log_tail_comparisons():
    density = (1 + x) / (x**3 * sp.log(x) ** 2)
    assert absolute_rule(density, ((x, sp.oo, 2),))[0] is True
    shifted = (3 - x) / ((2 - x) ** 3 * sp.log(2 - x) ** 2)
    assert absolute_rule(shifted, ((x, -sp.oo, 0),))[0] is True


def test_public_clip_rule_precedes_geometric_fallback(monkeypatch):
    import multiple_integrate.core as core

    monkeypatch.setattr(
        core, "reduce_domain", lambda *a, **k: pytest.fail("slow decomposition used")
    )
    result = integrate_result(x, region=sp.And(x * x + y * y <= 1, x >= 0), variables=(x, y))
    assert result.value == sp.Rational(2, 3)
    assert result.method == "clipped_quadratic_cell"
    assert result.complete and result.absolute_convergent


def test_variable_exponents_are_not_uniform_bounds():
    from multiple_integrate.convergence_comparison import bounded_multiplier

    assert not bounded_multiplier(2**x, (x,), ((x, 0, sp.oo),), None, lambda *a, **k: False)
    positive = sp.Symbol("positive", positive=True)
    density = (1 + positive) / (positive * sp.log(positive) ** 2)
    assert absolute_rule(density, ((positive, -1, 1),))[0] is not True


@pytest.mark.parametrize("n", [2, 3])
def test_half_ball_polynomial_moments(n):
    variables = (x, y, z)[:n]
    domain = sp.And(sum(v * v for v in variables) <= 1, x >= 0)
    result = integrate_clipped_quadratic_cell(1, domain, variables)
    expected = sp.pi ** sp.Rational(n, 2) / (2 * sp.gamma(sp.Rational(n, 2) + 1))
    assert result.complete and result.absolute_convergent
    assert sp.simplify(result.value - expected) == 0
    odd = integrate_clipped_quadratic_cell(y, domain, variables)
    assert odd.value == 0


def test_clipped_disk_cap_and_complement():
    circle = x * x + y * y <= 1
    cap = integrate_clipped_quadratic_cell(1, sp.And(circle, x >= sp.Rational(1, 2)), (x, y))
    rest = integrate_clipped_quadratic_cell(1, sp.And(circle, x <= sp.Rational(1, 2)), (x, y))
    oracle = sp.pi / 3 - sp.sqrt(3) / 4
    assert abs((cap.value - oracle).evalf(55)) < sp.Float("1e-48")
    assert abs((cap.value + rest.value - sp.pi).evalf(55)) < sp.Float("1e-48")


def test_rotated_ellipsoid_parallel_clip_covariance():
    # The map x=u+v, y=v has determinant 1 and sends the ellipse to a disk.
    circle = (x - y) ** 2 + y * y <= 1
    domain = sp.And(circle, x - y >= 0)
    result = integrate_clipped_quadratic_cell((x - y) ** 2 * y * y, domain, (x, y))
    assert sp.simplify(result.value - sp.pi / 48) == 0
    permuted = integrate_clipped_quadratic_cell((x - y) ** 2 * y * y, domain, (y, x))
    assert sp.simplify(permuted.value - result.value) == 0


def test_rotated_three_dimensional_clip_independent_sphere_identity():
    variables = (x, y, z)
    matrix = sp.Matrix([[3, 1, 0], [1, 2, 1], [0, 1, 4]])
    centered = sp.Matrix([x - 1, y - 2, z - 3])
    q = (centered.T * matrix * centered)[0]
    result = integrate_clipped_quadratic_cell(
        (x - 1) ** 2 * (y - 2) ** 2, sp.And(q <= 1, x + y + z >= 6), variables
    )
    inverse = matrix.inv()
    expected = (
        2
        * sp.pi
        / (105 * sp.sqrt(matrix.det()))
        * (inverse[0, 0] * inverse[1, 1] + 2 * inverse[0, 1] ** 2)
    )
    assert sp.simplify(result.value - expected) == 0


def test_symbolic_clip_order_uses_caller_prerequisites():
    domain = sp.And(x * x + y * y <= 1, x >= a)
    result = integrate_clipped_quadratic_cell(1, domain, (x, y), assumptions=sp.And(a > 0, a < 1))
    assert result is not None and result.complete
    expected = sp.pi / 3 - sp.sqrt(3) / 4
    assert abs((result.value.subs(a, sp.Rational(1, 2)) - expected).evalf(55)) < sp.Float("1e-48")
    assert integrate_clipped_quadratic_cell(1, domain, (x, y)) is None


def test_clipped_transition_and_intrinsic_slice_prerequisites():
    circle = x * x + y * y <= 1
    assert integrate_clipped_quadratic_cell(1, sp.And(circle, x >= 2), (x, y)).value == 0
    assert integrate_clipped_quadratic_cell(1, sp.And(circle, x >= 1), (x, y)).value == 0
    slice_domain = sp.And(circle, x >= 0, x <= 0)
    assert integrate_clipped_quadratic_cell(1, slice_domain, (x, y)).value == 0
    assert (
        integrate_clipped_quadratic_cell(1, slice_domain, (x, y), measure_dimension="intrinsic")
        is None
    )
    assert integrate_clipped_quadratic_cell(1, sp.And(circle, x >= 0, y >= 0), (x, y)) is None
    assert integrate_clipped_quadratic_cell(1, sp.And(x * x - y * y <= 1, x >= 0), (x, y)) is None


def test_parameter_cell_half_ellipsoid():
    result = integrate_cells(
        1,
        region=sp.And(x * x + y * y <= a * a + 1, x >= 0),
        variables=(x, y),
        parameters=(a,),
        cells=[ParameterCell(sp.true)],
    )
    assert result.complete
    assert sp.simplify(result.value - sp.pi * (a * a + 1) / 2) == 0
    assert result.diagnostics["cell_results"][0].method == "clipped_quadratic_cell"


@pytest.mark.parametrize("scale", [1, 2, -3])
@pytest.mark.parametrize("order", [2, 3])
def test_tangent_finite_part_preserves_regulator(scale, order):
    g = y * (y - x**order)
    domain = sp.And(x > 0, x < 1, y > 0, y < x**order)
    result = regularized_result(
        x ** (3 * order - 1) / g,
        region=domain,
        variables=(x, y),
        regularization=HadamardFinitePart(hypersurface=scale * g),
    )
    assert sp.simplify(result.value - (1 - sp.log(abs(scale))) / order) == 0
    details = result.provenance.details
    assert details["realization"] == "tangent_graph_cutoff"
    assert details["coverage_complete"]
    assert verify_regularization_result(result).valid
    targets = details["target_variables"]
    mapping = dict(zip(details["source_variables"], details["forward_map"], strict=True))
    assert (
        sp.simplify((scale * g).subs(mapping, simultaneous=True) - details["transformed_level"])
        == 0
    )
    assert (
        sp.simplify(
            sp.Matrix(details["forward_map"]).jacobian(targets).det() - details["chart_jacobian"]
        )
        == 0
    )
    assert (
        tuple(sp.simplify(v.subs(mapping, simultaneous=True)) for v in details["inverse_map"])
        == targets
    )


def test_tangent_cutoff_independent_original_coordinate_oracle():
    g = y * (y - 3 * x * x)
    e = sp.Symbol("epsilon", positive=True)
    domain = sp.And(x > 0, x < 2, y > 0, y < 3 * x * x)
    value, details = tangent_graph_cutoff(x**5 / g, domain, 2 * g, e, (x, y))
    with mp.workdps(60):
        eps = mp.mpf("0.01")
        floor = mp.root(4 * eps / (2 * 9), 4)

        def inner(t):
            h = 3 * t * t
            root = mp.sqrt(max(mp.mpf(0), h * h - 4 * eps / 2))
            lower = (h - root) / 2
            upper = (h + root) / 2
            return 2 * t**3 / 3 * mp.log(lower / upper)

        oracle = mp.quad(inner, [floor, 1, 2])
        actual = mp.mpf(str(value.subs(e, sp.Rational(1, 100)).evalf(55)))
        assert abs(actual - oracle) < mp.mpf("1e-45")
    assert details["regulator_scale"] == 18


def test_tangent_principal_value_divergence_is_not_a_finite_part():
    g = y * (y - x * x)
    result = regularized_result(
        x**5 / g,
        region=sp.And(x > 0, x < 1, y > 0, y < x * x),
        variables=(x, y),
        principal_value=HypersurfacePrincipalValue(g),
    )
    assert result.outcome is IntegrationOutcome.DIVERGENT
    assert result.value == -sp.oo


def test_tangent_broken_chart_and_radial_order_decline():
    g = y * (y - x * x)
    e = sp.Symbol("epsilon", positive=True)
    domain = sp.And(x > 0, x < 1, y > 0, y < x * x)
    assert tangent_graph_cutoff(1 / g, domain, g, e, (x, y)) is None
    assert tangent_graph_cutoff(x**5 / (g * (x - sp.Rational(1, 2))), domain, g, e, (x, y)) is None
    assert (
        tangent_graph_cutoff(x**5 / g, sp.And(x > -1, x < 1, y > 0, y < x * x), g, e, (x, y))
        is None
    )
    assert tangent_graph_cutoff(x**5 / g, domain, y * (y - x**3), e, (x, y)) is None
    tiny = IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=1))
    assert selected_interval_rule(sp.exp(-x * x) * (1 + x) / x**2, x, 1, 2, policy=tiny) is None


def test_guide_examples_execute():
    guide = Path(__file__).parents[1] / "docs/laurent-clips-and-tangencies.md"
    snippets = re.findall(r"```python\n(.*?)```", guide.read_text(), flags=re.S)
    assert len(snippets) == 3
    for snippet in snippets:
        exec(compile(snippet, str(guide), "exec"), {})


def test_tangent_coordinate_permutation_preserves_absolute_jacobian():
    g = y * (y - x * x)
    epsilon = sp.Symbol("epsilon", positive=True)
    domain = sp.And(x > 0, x < 1, y > 0, y < x * x)
    original = tangent_graph_cutoff(x**5 / g, domain, g, epsilon, (x, y))
    permuted = tangent_graph_cutoff(x**5 / g, domain, g, epsilon, (y, x))
    assert sp.simplify(original[0] - permuted[0]) == 0
    details = permuted[1]
    determinant = sp.Matrix(details["forward_map"]).jacobian(details["target_variables"]).det()
    assert sp.simplify(sp.Abs(determinant) - details["chart_jacobian"]) == 0


@pytest.mark.parametrize("extra", [-x * x + y * y, x * y, x * x, x * x + y * y - 2 * x])
def test_unrecognized_second_quadratic_is_never_dropped(extra):
    domain = sp.And(extra <= 0, x * x + y * y <= 1, x >= 0)
    assert integrate_clipped_quadratic_cell(1, domain, (x, y)) is None
