from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor

import pytest
import sympy as sp

import multiple_integrate.core as core
from multiple_integrate import (
    BallRegion,
    EllipsoidRegion,
    boole,
    clear_cache,
    multiple_integrate,
    region_from_ranges,
)


def test_generic_constant_nested_region_integrates_volume_not_width_product():
    x, y, z = sp.symbols("x y z", real=True)
    ranges = [(z, 0, x * y), (y, 0, x**2), (x, 0, 1)]
    assert multiple_integrate(1, *ranges) == sp.Rational(1, 12)


def test_zero_over_infinite_region_is_exact_zero():
    x = sp.symbols("x", real=True)
    assert multiple_integrate(0, (x, 0, sp.oo)) == 0


def test_odd_singular_integral_is_not_replaced_by_zero():
    x = sp.symbols("x", real=True)
    result = multiple_integrate(1 / x, (x, -1, 1))
    assert result != 0


def test_one_dimensional_principal_value_is_explicit():
    x = sp.symbols("x", real=True)
    assert multiple_integrate(1 / x, (x, -1, 1), principal_value=True) == 0


def test_multidimensional_principal_value_is_rejected_until_defined():
    x, y = sp.symbols("x y", real=True)
    with pytest.raises(NotImplementedError):
        multiple_integrate(1 / x, (x, -1, 1), (y, 0, 1), principal_value=True)


def test_reflection_invariance_checks_dependent_inner_bounds():
    x, y = sp.symbols("x y", real=True)
    ranges = [(y, 0, 1 + x), (x, -1, 1)]
    assert multiple_integrate(x, *ranges) == sp.Rational(2, 3)


def test_graph_boolean_restriction_cannot_enlarge_box():
    x, y = sp.symbols("x y", real=True)
    result = multiple_integrate(
        (x + 2) * boole(y <= 2 * x + 2),
        (x, 0, 1),
        (y, 0, 1),
    )
    assert result == sp.Rational(5, 2)


def test_disk_boolean_restriction_cannot_enlarge_box():
    x, y = sp.symbols("x y", real=True)
    result = multiple_integrate(
        (x + 2) * boole(x**2 + y**2 <= 4),
        (x, -1, 1),
        (y, -1, 1),
    )
    assert result == 8


def test_ellipsoid_axis_agreement_is_not_enough_for_radiality():
    x, y = sp.symbols("x y", real=True)
    region = EllipsoidRegion(variables_nd=(x, y), axes=(2, 3))
    assert region.radial_integral(sp.exp(x * y) - 1) is None


def test_high_dimensional_ball_axis_agreement_is_not_enough_for_radiality():
    x0, x1, x2, x3 = sp.symbols("x0:4", real=True)
    region = BallRegion(
        tuple((v, -1, 1) for v in (x0, x1, x2, x3)),
        radius=1,
        dimension=4,
    )
    assert region.radial_integral(sp.exp(x0 * x1) - 1) is None


def test_improper_exponential_uses_convergence_aware_fallback():
    x, a = sp.symbols("x a", real=True)
    result = multiple_integrate(sp.exp(a * x), (x, 0, sp.oo))
    assert not result.has(sp.exp(sp.oo * a))
    assert isinstance(result, (sp.Piecewise, sp.Integral)) or result.has(sp.Integral)


def test_multiple_integrate_enforces_inner_first_dependencies():
    x, y = sp.symbols("x y", real=True)
    with pytest.raises(ValueError, match="inner-first"):
        multiple_integrate(1, (x, 0, 1), (y, 0, x))


def test_standalone_region_classifier_can_remain_orientation_flexible():
    x, y = sp.symbols("x y", real=True)
    region = region_from_ranges([(x, 0, 1), (y, 0, 1 - x)])
    assert type(region).__name__ in {"SimplexRegion", "AffineSimplexRegion", "GraphRegion"}
    assert region.constant_volume() == sp.Rational(1, 2)


def test_duplicate_integration_variables_rejected():
    x = sp.symbols("x", real=True)
    with pytest.raises(ValueError, match="unique"):
        multiple_integrate(1, (x, 0, 1), (x, 0, 1))


def test_bound_may_not_depend_on_its_own_variable():
    x = sp.symbols("x", real=True)
    with pytest.raises(ValueError, match="inner-first"):
        multiple_integrate(1, (x, 0, x + 1))


def test_generate_conditions_removed_from_public_signature():
    assert "generate_conditions" not in inspect.signature(multiple_integrate).parameters


def test_result_cache_is_bounded_and_clearable():
    x = sp.symbols("x", real=True)
    clear_cache()
    for n in range(core._CACHE_MAXSIZE + 20):
        assert multiple_integrate(x, (x, 0, n + 1)) == sp.Rational((n + 1) ** 2, 2)
    assert len(multiple_integrate._cache) <= core._CACHE_MAXSIZE
    clear_cache()
    assert len(multiple_integrate._cache) == 0


def test_signal_timeout_skips_heuristic_outside_main_thread():
    x = sp.symbols("x", real=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(core._find_critical_points, x**2, x, -1, 1)
    assert future.result() == []
