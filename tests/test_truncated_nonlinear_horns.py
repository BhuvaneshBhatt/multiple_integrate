"""Independent oracles and exceptional-fiber tests for the new reductions."""

import mpmath as mp
import pytest
import sympy as sp

from multiple_integrate import (
    IntegrationPolicy,
    StrategyBudget,
    StrategyBudgetExceeded,
    integrate_cells,
    integrate_result,
)
from multiple_integrate.convergence_evidence import absolute_rule
from multiple_integrate.cusp_horns import try_cusp_horn
from multiple_integrate.quadratic_cell_geometry import integrate_quadratic_cell
from multiple_integrate.selected_families import selected_interval_rule
from multiple_integrate.singularities import stratify_singular_locus
from multiple_integrate.transition_cells import linear_quadratic_cells

x, y, a, b = sp.symbols("x y a b", real=True)


@pytest.mark.parametrize("power", [-1, -2, -sp.Rational(5, 2)])
@pytest.mark.parametrize("bounds", [(3, 4), (3, sp.oo), (4, 3)])
def test_negative_shifted_truncated_moments(power, bounds):
    density = (x - 2) ** power * sp.exp(-2 * (x - 2) ** 4)
    rule = selected_interval_rule(density, x, *bounds)
    assert rule is not None
    with mp.workdps(50):
        lo, hi = [mp.inf if z == sp.oo else z for z in bounds]
        oracle = mp.quad(sp.lambdify(x, density, "mpmath"), [lo, hi])
        assert abs(mp.mpf(str(rule[1].evalf(45))) - oracle) < mp.mpf("1e-40")
    assert absolute_rule(density, ((x, *sorted(bounds, key=lambda z: float(z))),))[0] is True


@pytest.mark.parametrize("power", [-1, -2])
def test_zero_cutoff_keeps_origin_threshold(power):
    assert selected_interval_rule(x**power * sp.exp(-(x**4)), x, 0, sp.oo) is None


def test_symbolic_negative_order_positive_cutoff():
    p, lo = sp.symbols("p lo", real=True)
    result = integrate_result(
        x**p * sp.exp(-(x**2)), (x, lo, sp.oo), assumptions=sp.And(p <= -1, lo > 0)
    )
    assert result.absolute_convergent is True and result.complete
    assert sp.simplify(result.value - sp.uppergamma((p + 1) / 2, lo**2) / 2) == 0


@pytest.mark.parametrize(
    "phase,ranges,expected",
    [
        (-x, ((x, 0, sp.oo),), True),
        (x, ((x, -sp.oo, 0),), True),
        (-x, ((x, -sp.oo, 0),), False),
        (-x, ((x, sp.oo, 0),), True),
        (-x, ((x, 0, -sp.oo),), False),
        (-(x**2) - y**2 + sp.I * x * y, ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo)), True),
    ],
)
def test_scoped_exponential_multiplier(phase, ranges, expected):
    from multiple_integrate.convergence_comparison import polynomial_exponential_bound

    assert (
        polynomial_exponential_bound(phase, tuple(r[0] for r in ranges), ranges, None) is expected
    )


def test_exponential_majorant_propagates_absolute_evidence():
    assert absolute_rule(sp.exp(-x) / (1 + x**2), ((x, 0, sp.oo),))[0] is True
    assert absolute_rule(sp.exp(x) / (1 + x**2), ((x, 0, sp.oo),))[0] is not True
    # A coordinate's positive Symbol flag does not constrain its requested range.
    positive = sp.Symbol("positive", positive=True)
    assert (
        absolute_rule(sp.exp(-positive) / (1 + positive**2), ((positive, -sp.oo, 0),))[0]
        is not True
    )


@pytest.mark.parametrize("intrinsic", [False, True])
def test_nonlinear_disk_empty_point_and_positive_cells(intrinsic):
    radius = a**2 + b**2 - 1
    result = integrate_cells(
        1,
        parameters=(a, b),
        region=x**2 + y**2 <= radius,
        variables=(x, y),
        measure_dimension="intrinsic" if intrinsic else "ambient",
    )
    assert result.complete and result.exact
    assert [c.value for c in result.cases] == [0, 1 if intrinsic else 0, sp.pi * radius]
    assert all(child.absolute_convergent is True for child in result.diagnostics["cell_results"])
    assert result.coverage[0].complete


def test_nonlinear_ellipsoid_moment_and_parameter_permutation():
    radius = a**2 + b**2 + 1
    formula = 4 * (x - a) ** 2 + (y - b) ** 2 <= radius
    first = integrate_cells((x - a) ** 2, parameters=(a, b), region=formula, variables=(x, y))
    second = integrate_cells((x - a) ** 2, parameters=(b, a), region=formula, variables=(x, y))
    assert first.complete and second.complete
    assert sp.simplify(first.value - sp.pi * radius**2 / 32) == 0
    assert first.value == second.value


def test_strict_zero_fiber_is_empty_intrinsically():
    child = integrate_quadratic_cell(
        1, x * x + y * y < a, (x, y), assumptions=sp.Eq(a, 0), measure_dimension="intrinsic"
    )
    assert child.value == 0 and child.diagnostics["dimension"] == -1


def test_quadratic_cell_declines_indefinite_and_surface_requests():
    assert integrate_quadratic_cell(1, x * x - y * y < a, (x, y), assumptions=a > 0) is None
    assert integrate_quadratic_cell(1, sp.Eq(x * x + y * y, a), (x, y), assumptions=a > 0) is None


def test_higher_degree_parameter_events_cover_equality_fibers():
    cells = linear_quadratic_cells(region=x * x <= a**3 - a, variables=(x,), parameters=(a,))
    assert {c.condition for c in cells if isinstance(c.condition, sp.Equality)} == {
        sp.Eq(a, -1),
        sp.Eq(a, 0),
        sp.Eq(a, 1),
    }
    result = integrate_cells(1, region=x * x <= a**3 - a, variables=(x,), parameters=(a,))
    assert result.complete and result.value.subs(a, 2) == 2 * sp.sqrt(6)
    assert result.value.subs(a, 0) == 0 and result.value.subs(a, sp.Rational(1, 2)) == 0


def test_nonlinear_sign_tree_budget():
    policy = IntegrationPolicy(semialg=StrategyBudget(max_pieces=2))
    with pytest.raises(StrategyBudgetExceeded):
        linear_quadratic_cells(
            region=x * x + y * y <= a * a + b * b - 1,
            variables=(x, y),
            parameters=(a, b),
            policy=policy,
        )


@pytest.mark.parametrize(
    "m,n,power,expected",
    [
        (4, 2, sp.Rational(1, 2), sp.pi),
        (2, 4, sp.Rational(1, 2), sp.beta(sp.Rational(1, 4), sp.Rational(1, 2))),
        (3, 4, sp.Rational(1, 4), sp.pi / sp.sqrt(2)),
    ],
)
def test_weighted_horn_independent_beta_oracles(m, n, power, expected):
    formula = sp.And(x > 0, x < 1, y**n <= x**m)
    density = (x**m - y**n) ** (-power)
    value = try_cusp_horn(density, formula, (x, y))
    assert (
        sp.simplify(
            (sp.expand_func(value) - sp.expand_func(expected)).subs(
                sp.gamma(sp.Rational(1, 4)) * sp.gamma(sp.Rational(3, 4)), sp.sqrt(2) * sp.pi
            )
        )
        == 0
    )
    result = integrate_result(density, region=formula, variables=(x, y))
    assert result.absolute_convergent is True
    assert (
        sp.simplify(
            (sp.expand_func(result.value) - sp.expand_func(expected)).subs(
                sp.gamma(sp.Rational(1, 4)) * sp.gamma(sp.Rational(3, 4)), sp.sqrt(2) * sp.pi
            )
        )
        == 0
    )


@pytest.mark.parametrize("power", [sp.Rational(1, 2), 1])
def test_weighted_horn_radial_and_angular_divergence(power):
    assert (
        try_cusp_horn((x**4 - y**4) ** (-power), sp.And(x > 0, x < 1, y**4 <= x**4), (x, y))
        == sp.oo
    )


def test_weighted_horn_odd_n_and_unknown_contact_decline():
    assert (
        try_cusp_horn(
            (x**3 - y**3) ** (-sp.Rational(1, 2)), sp.And(x > 0, x < 1, y**3 <= x**3), (x, y)
        )
        is None
    )
    assert (
        try_cusp_horn(
            x**4 / (x**4 - y * y),
            sp.And(x > 0, x < 1, y * y <= a * x**4),
            (x, y),
            assumptions=sp.And(a > 0, a <= 1),
        )
        is None
    )


def test_rank_evidence_distinguishes_transverse_and_tangent_intersections():
    transverse = stratify_singular_locus(1 / (x * y), (x, y))
    tangent = stratify_singular_locus(1 / (y * (y - x * x)), (x, y))
    assert next(s for s in transverse if len(s.components) == 2).regular is True
    assert next(s for s in tangent if len(s.components) == 2).regular is False


def test_complex_geometry_with_unproved_coefficient_domain_stays_unknown():
    details = stratify_singular_locus(1 / (x + sp.I * sp.exp(1 / a)), (x,), return_details=True)
    assert not details.geometry_certified
    assert details.strata[0].dimension is None


@pytest.mark.parametrize(
    "density,ranges,expected",
    [
        ((2 * x + 3) ** -2, ((x, 0, sp.oo),), True),
        ((3 - 2 * x) ** -2, ((x, -sp.oo, 0),), True),
        ((2 * x + 3) ** -1, ((x, 0, sp.oo),), False),
        ((2 * x) ** -2, ((x, 0, sp.oo),), False),
        ((3 - 2 * x) ** -2, ((x, 0, sp.oo),), False),
    ],
)
def test_affine_tail_comparison_with_broken_prerequisites(density, ranges, expected):
    from multiple_integrate.convergence_comparison import affine_power_absolute

    assert (affine_power_absolute(density, ranges[0], None) is True) is expected


def test_complex_rate_negative_order_truncation():
    density = (x - 1) ** -1 * sp.exp(-(2 + sp.I) * (x - 1) ** 4)
    rule = selected_interval_rule(density, x, 2, 3)
    assert rule is not None
    with mp.workdps(50):
        oracle = mp.quad(sp.lambdify(x, density, "mpmath"), [2, 3])
        value = sp.N(rule[1], 45)
        observed = mp.mpc(str(sp.re(value)), str(sp.im(value)))
        assert abs(oracle - observed) < mp.mpf("1e-40")


def test_canceled_coordinate_boundary_normalizes_before_pivot():
    expression = sp.Add(x, -x, a, evaluate=False)
    cells = linear_quadratic_cells(
        region=sp.Ge(expression, 0, evaluate=False), variables=(x,), parameters=(a,)
    )
    assert {c.condition for c in cells} == {a < 0, sp.Eq(a, 0), a > 0}


@pytest.mark.parametrize("index", [0, 1, 2])
def test_extension_guide_examples(index):
    import re
    from pathlib import Path

    guide = Path(__file__).resolve().parents[1] / "docs/nonlinear-moments-and-singularities.md"
    examples = re.findall(r"```python\n(.*?)```", guide.read_text(), re.S)
    assert len(examples) == 3
    exec(compile(examples[index], str(guide), "exec"), {})


def test_rotated_nonlinear_ellipsoid_cross_moment_reuses_covariance():
    radius = a * a + b * b + 1
    formula = 2 * x * x + 2 * x * y + 2 * y * y <= radius
    result = integrate_cells(x * y, parameters=(a, b), region=formula, variables=(x, y))
    # A^-1 has off-diagonal entry -1/3; the unit-disk second moment is pi/4.
    assert result.complete and result.exact
    assert sp.simplify(result.value + sp.pi * radius**2 / (12 * sp.sqrt(3))) == 0
    permuted = integrate_cells(x * y, parameters=(b, a), region=formula, variables=(y, x))
    assert sp.simplify(permuted.value - result.value) == 0


def test_unknown_intersection_rank_does_not_certify_geometry(monkeypatch):
    import multiple_integrate.singularities as singularities

    monkeypatch.setattr(
        singularities,
        "semialg_is_satisfiable",
        lambda condition, variables: False if condition is sp.false else None,
    )
    details = singularities.stratify_singular_locus(
        1 / (y * (y - x * x)), (x, y), return_details=True
    )
    intersection = next(s for s in details.strata if len(s.components) == 2)
    assert intersection.dimension == 0 and intersection.regular is None
    assert details.complete and not details.geometry_certified


def test_single_locus_rank_query_preserves_linear_gradient_equations(monkeypatch):
    import multiple_integrate.singularities as singularities

    calls = []
    monkeypatch.setattr(singularities, "semialg_region_dimension", lambda *args: None)

    def check(condition, variables):
        calls.append(condition)
        assert condition.has(sp.Eq(2 * x, 0)) and condition.has(sp.Eq(2 * y, 0))
        return None

    monkeypatch.setattr(singularities, "semialg_is_satisfiable", check)
    details = singularities.stratify_singular_locus(
        1 / (b + x * x + y * y), (x, y), return_details=True
    )
    assert len(calls) == 1 and details.queries == 2
    assert not details.geometry_certified
