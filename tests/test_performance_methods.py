"""Independent exact, numeric and adversarial checks for the new fast portfolio."""

from dataclasses import replace

import mpmath as mp
import pytest
import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    IntegrationOutcome,
    IntegrationPolicy,
    StrategyBudget,
    StrategyBudgetExceeded,
    clear_cache,
    integrate_cells,
    integrate_result,
    regularized_result,
    verify_regularization_result,
)
from multiple_integrate.fast_rules import (
    box_polynomial_moment,
    polynomial_function_moment,
    root_free_rational,
)
from multiple_integrate.region_recognition import _recognize_ranges, region_from_ranges
from multiple_integrate.transition_cells import linear_quadratic_cells
from multiple_integrate.triangular_intersections import triangular_product_cutoff

x, y, z, a = sp.symbols("x y z a", real=True)
policy = IntegrationPolicy(max_singular_queries=0)


def test_geometry_cache_is_independent_of_integrand_and_cleared():
    clear_cache()
    first = region_from_ranges([(x, 0, 2), (y, 0, 3)], structural_order="inner-first")
    second = region_from_ranges([(x, 0, 2), (y, 0, 3)], structural_order="inner-first")
    assert first is second and _recognize_ranges.cache_info().hits == 1
    region_from_ranges([(x, 0, 2), (y, 0, 3)], structural_order="auto")
    assert _recognize_ranges.cache_info().misses == 2
    clear_cache()
    assert _recognize_ranges.cache_info().currsize == 0


@pytest.mark.parametrize("degree", range(7))
def test_gaussian_recurrence_against_high_precision_quadrature(degree):
    density = x**degree * sp.exp(-3 * (x - 1) ** 2)
    result = integrate_result(density, (x, 0, 2), policy=policy)
    assert result.method == "polynomial_function_moments"
    assert result.absolute_convergent is True
    with mp.workdps(45):
        oracle = mp.quad(lambda t: t**degree * mp.exp(-3 * (t - 1) ** 2), [0, 1, 2])
        assert abs(mp.mpf(str(sp.N(result.value, 40))) - oracle) < mp.mpf("1e-35")


@pytest.mark.parametrize("function", [sp.exp, sp.sin, sp.cos])
@pytest.mark.parametrize("degree", [0, 1, 4])
def test_affine_moment_zero_frequency_cell(function, degree):
    result = integrate_result(x**degree * function(a * x + 1), (x, -1, 2), policy=policy)
    assert result.complete is True
    assert result.value.subs(a, 0) == (
        sp.Integer(2) ** (degree + 1) - sp.Integer(-1) ** (degree + 1)
    ) * function(1) / sp.Integer(degree + 1)
    for frequency in (-2, 1):
        reference = sp.integrate(x**degree * function(frequency * x + 1), (x, -1, 2))
        assert sp.simplify(result.value.subs(a, frequency) - reference) == 0


def test_recurrence_does_not_call_general_integration(monkeypatch):
    import multiple_integrate.assumptions as assumptions
    import multiple_integrate.strategy_common as common

    def forbidden(*args, **kwargs):
        pytest.fail("general integration backend called")

    monkeypatch.setattr(assumptions, "_integrate", forbidden)
    monkeypatch.setattr(common, "_integrate", forbidden)
    clear_cache()
    assert integrate_result(x**4 * sp.exp(-3 * (x - 1) ** 2), (x, 0, 2), policy=policy).finite
    assert integrate_result(x**4 * sp.sin(2 * x), (x, 0, 2), policy=policy).finite


@pytest.mark.parametrize("bounds", [(0, 1), (1, 0), (1, 1), (-2, 3)])
def test_sparse_box_moment_preserves_orientation_and_degeneracy(bounds):
    lo, hi = bounds
    result = integrate_result((x + y) ** 6, (x, lo, hi), (y, 0, 1), policy=policy)
    oracle = sp.integrate((x + y) ** 6, (x, lo, hi), (y, 0, 1))
    assert result.value == oracle
    assert result.absolute_convergent is True


def test_polynomial_expansion_budget_declines_before_poly(monkeypatch):
    import multiple_integrate.fast_rules as rules
    from multiple_integrate.policy import _CURRENT_POLICY

    budget = IntegrationPolicy(semialg=StrategyBudget(max_pieces=2))
    token = _CURRENT_POLICY.set(budget)
    try:

        def forbidden(*args, **kwargs):
            pytest.fail("expanded an over-budget polynomial")

        monkeypatch.setattr(rules.sp, "Poly", forbidden)
        assert box_polynomial_moment((x + y) ** 8, [(x, 0, 1), (y, 0, 1)]) is None
    finally:
        _CURRENT_POLICY.reset(token)


@pytest.mark.parametrize(
    "density",
    [
        1 / (x + 2),
        x**3 / (x + 2),
        1 / (x * x + x + 1),
        x / (x * x + x + 1),
        1 / (x * x - 4),
        1 / (x - 2) ** 2,
        (x**4 + 1) / (x * x + 1),
        1 / (-x * x - 1),
    ],
)
def test_root_free_rational_matches_exact_backend(density):
    result = integrate_result(density, (x, 0, 1), policy=policy)
    assert result.method == "root_free_rational"
    assert result.absolute_convergent is True
    reference = sp.integrate(density, (x, 0, 1))
    assert sp.simplify(result.value - reference) == 0


@pytest.mark.parametrize("density", [1 / x, 1 / (x * x - 1), 1 / (x - 1) ** 2])
def test_rational_pole_cannot_enter_root_free_route(density):
    assert root_free_rational(density, x, -2, 2) is None


@pytest.mark.parametrize("function", [sp.sin, sp.cos])
@pytest.mark.parametrize("bounds", [(0, 1), (-1, 0), (-1, 1), (1, -1)])
def test_reciprocal_phase_one_sided_limits(function, bounds):
    result = integrate_result(function(1 / x) / x, (x, *bounds), policy=policy)
    if function is sp.sin:
        factor = 2 if bounds == (-1, 1) else -2 if bounds == (1, -1) else 1
        expected = factor * (sp.pi / 2 - sp.Si(1))
    else:
        expected = -sp.Ci(1) if bounds == (0, 1) else sp.Ci(1) if bounds == (-1, 0) else 0
    assert sp.simplify(result.value - expected) == 0
    assert result.method == "reciprocal_phase"
    assert result.ordinary_convergent is True and result.absolute_convergent is False


def test_reciprocal_phase_declines_unknown_or_zero_phase():
    from multiple_integrate.fast_rules import reciprocal_phase_rule

    assert reciprocal_phase_rule(sp.sin(a / x) / x, x, 0, 1) is None
    assert (
        integrate_result(sp.cos(0) / x, (x, 0, 1), policy=policy).outcome
        is IntegrationOutcome.DIVERGENT
    )


@pytest.mark.parametrize("function", [sp.exp, sp.sin, sp.cos])
def test_nonseparable_series_exact_and_uniformly_convergent(function):
    result = integrate_result(function(x * y), (x, 0, 1), (y, 0, 1), policy=policy)
    assert result.method == "entire_box_series" and result.finite is True
    assert result.absolute_convergent is True and result.complete is True
    with mp.workdps(40):
        numerical = mp.quad(
            lambda u: mp.quad(lambda v: getattr(mp, function.__name__)(u * v), [0, 1]), [0, 1]
        )
        actual = sp.N(result.value, 35)
        assert abs(mp.mpc(str(sp.re(actual)), str(sp.im(actual))) - numerical) < mp.mpf("1e-30")


def test_nonseparable_series_zero_parameter_and_bad_coefficient():
    result = integrate_result(sp.exp(a * x * y), (x, 0, 1), (y, 0, 1), policy=policy)
    assert result.finite is True
    assert result.value.subs(a, 0).doit() == 1
    from multiple_integrate.entire_box_series import entire_box_series

    assert entire_box_series(sp.exp(x * y / a), [(x, 0, 1), (y, 0, 1)], assumptions=a >= 0) is None


def test_automatic_degree_drop_cell_retains_zero_fiber():
    result = integrate_cells(
        1, region=a * x >= 0, variables=(x,), bounds=[(x, -1, 1)], parameters=(a,), policy=policy
    )
    assert result.complete is True
    assert result.value.subs(a, 0) == 2
    assert result.value.subs(a, 1) == result.value.subs(a, -1) == 1
    assert result.coverage[0].method == "automatic_transition_cells"


def test_automatic_quadratic_topology_and_boundary_coincidence():
    result = integrate_cells(
        1, region=x * x <= a, variables=(x,), bounds=[(x, -1, 1)], parameters=(a,), policy=policy
    )
    assert result.complete is True
    assert result.value.subs(a, -1) == result.value.subs(a, 0) == 0
    assert result.value.subs(a, sp.Rational(1, 4)) == 1
    assert result.value.subs(a, 2) == 2
    cells = linear_quadratic_cells(
        region=x * x <= a, variables=(x,), bounds=[(x, -1, 1)], parameters=(a,)
    )
    assert any(c.condition == sp.Eq(a, 0) for c in cells)
    assert any(c.condition == sp.Eq(a, 1) for c in cells)


def test_automatic_intrinsic_dimension_drop():
    result = integrate_cells(
        1,
        region=sp.Eq(a * x, 0),
        variables=(x,),
        bounds=[(x, -1, 1)],
        parameters=(a,),
        measure_dimension="intrinsic",
        policy=policy,
    )
    assert result.value.subs(a, 1) == 1
    assert result.value.subs(a, 0) == 2
    assert result.complete is True


def test_automatic_cells_decline_unproved_multivariate_transitions():
    with pytest.raises(NotImplementedError, match="linear or quadratic boundaries"):
        linear_quadratic_cells(region=x * y * (x + y) <= a, variables=(x, y), parameters=(a,))
    with pytest.raises(StrategyBudgetExceeded, match="degree"):
        linear_quadratic_cells(region=x <= a**9, variables=(x,), parameters=(a,))


def test_triangular_intersection_finite_part_and_replay_tampering():
    u, v, w = x, y - x * x, z - x * y
    level = u * v * w
    domain = sp.And(*(atom for f in (u, v, w) for atom in (f >= -1, f <= 1)))
    result = regularized_result(
        1 / level**2,
        region=domain,
        variables=(x, y, z),
        regularization=HadamardFinitePart(hypersurface=level),
        policy=policy,
    )
    assert result.value == -8
    assert result.provenance.details["realization"] == "triangular_polynomial_cutoff"
    assert verify_regularization_result(result).valid
    details = dict(result.provenance.details)
    details["triangular_jacobian"] = 2
    corrupted = replace(result, provenance=replace(result.provenance, details=details))
    assert not verify_regularization_result(corrupted).valid


def test_nontransverse_polynomial_intersection_is_not_certified():
    e = sp.Dummy("epsilon", positive=True)
    level = (y - x * x) * (y + x * x) * z
    domain = sp.And(x >= -1, x <= 1, y >= -1, y <= 1, z >= -1, z <= 1)
    assert triangular_product_cutoff(level**-2, domain, level, e, (x, y, z)) is None


def test_broken_moment_coefficient_never_gets_fast_certificate():
    assert polynomial_function_moment(x * sp.exp(a * x) / a, x, 0, 1, assumptions=a >= 0) is None


def test_nested_expansion_budget_precedes_polynomial_construction(monkeypatch):
    monkeypatch.setattr(sp, "Poly", lambda *a, **k: pytest.fail("expanded before budget"))
    assert box_polynomial_moment((x * (y + z)) ** 100, ((x, 0, 1), (y, 0, 1), (z, 0, 1))) is None


def test_automatic_geometry_retains_parameter_pole_fibers():
    cells = linear_quadratic_cells(parameters=(a,), region=x / a >= 0, variables=(x,))
    assert any(c.condition == sp.Eq(a, 0) for c in cells)


def test_nonseparable_trig_constant_and_multi_phase():
    from multiple_integrate.entire_box_series import entire_box_series

    value = entire_box_series(sp.sin(1 + x * y + x * y**2), ((x, 0, 1), (y, 0, 1)))
    # Independent numerical quadrature compared to an explicit series truncation.
    truncated = value.replace(
        lambda e: isinstance(e, sp.Sum),
        lambda e: sp.Sum(e.function, *((k, 0, 12) for k, _, _ in e.limits)).doit(),
    )
    with mp.workdps(35):
        oracle = mp.quad(lambda u: mp.quad(lambda v: mp.sin(1 + u * v + u * v * v), [0, 1]), [0, 1])
        assert abs(mp.mpf(str(sp.N(truncated, 30))) - oracle) < mp.mpf("1e-25")


def test_fast_method_documentation_executes():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    page = root / "docs/performance-methods.md"
    for block in re.findall(r"```python\n(.*?)```", page.read_text(), re.S):
        exec(compile(block, str(page), "exec"), {})
    assert "tests/test_performance_methods.py" in page.read_text()
    assert "performance-methods" in (root / "docs/index.rst").read_text()


@pytest.mark.parametrize("bounds", [(0, sp.oo), (-sp.oo, 0), (-sp.oo, sp.oo), (sp.oo, -sp.oo)])
def test_gaussian_improper_moments_have_vanishing_boundary_terms(bounds):
    result = integrate_result(x**4 * sp.exp(-2 * x**2), (x, *bounds), policy=policy)
    signed_half = 3 * sp.sqrt(2) * sp.sqrt(sp.pi) / 64
    expected = signed_half * (
        2 if bounds == (-sp.oo, sp.oo) else -2 if bounds == (sp.oo, -sp.oo) else 1
    )
    assert sp.simplify(result.value - expected) == 0
    assert result.absolute_convergent is True
    assert not result.value.has(sp.nan)


def test_nonseparable_series_preserves_reversed_orientation():
    result = integrate_result(sp.exp(x * y), (x, 1, 0), (y, 0, 1))
    forward = integrate_result(sp.exp(x * y), (x, 0, 1), (y, 0, 1))
    assert abs(float(sp.N(result.value + forward.value))) < 1e-14
