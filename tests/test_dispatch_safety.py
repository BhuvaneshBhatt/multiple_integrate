import signal

import pytest
import sympy as sp

import multiple_integrate.analytic_strategies as strategies
import multiple_integrate.assumptions as assumptions
import multiple_integrate.recognition as recognition
import multiple_integrate.transforms as transforms
from multiple_integrate import (
    Boole,
    core,
    integrate,
    region_from_ranges,
)
from multiple_integrate.region_types import DiskRegion, EllipsoidRegion


def test_translated_polar_map_keeps_center():
    x, y = sp.symbols("x y", real=True)
    region = DiskRegion(
        ((y, -sp.sqrt(1 - (x - 1) ** 2), sp.sqrt(1 - (x - 1) ** 2)), (x, 0, 2)),
        radius=1,
        center=(0, 1),
    )
    transform = transforms._polar_disk_transform(region)
    theta, radius = transform.target_vars
    assert sp.simplify(transform.forward_map[0] - radius * sp.cos(theta)) == 0
    assert sp.simplify(transform.forward_map[1] - (1 + radius * sp.sin(theta))) == 0


def test_translated_ellipsoid_map_keeps_center():
    x, y = sp.symbols("x y", real=True)
    region = EllipsoidRegion(variables_nd=(x, y), axes=(2, 3), center=(1, -2))
    transform = transforms._affine_region_transform(region)
    assert transform is not None
    assert sp.simplify(transform.forward_map[0] - (1 + 2 * transform.target_vars[1])) == 0
    assert sp.simplify(transform.forward_map[1] - (-2 + 3 * transform.target_vars[0])) == 0


def test_disjoint_or_uses_union_without_double_counting():
    x = sp.symbols("x", real=True)
    result = integrate(
        Boole((x < sp.Rational(1, 2)) | (x > sp.Rational(3, 2))),
        (x, 0, 2),
    )
    assert result == 1


def test_overlapping_or_declines_union_shortcut_but_is_correct():
    x = sp.symbols("x", real=True)
    result = integrate(
        Boole((x < sp.Rational(3, 2)) | (x > sp.Rational(1, 2))),
        (x, 0, 2),
    )
    assert result == 2


def test_geometric_volume_respects_negative_symbolic_orientation():
    x, y, a = sp.symbols("x y a", real=True)
    ranges = [
        (y, -sp.sqrt(a**2 - x**2), sp.sqrt(a**2 - x**2)),
        (x, -a, a),
    ]
    positive = integrate(1, *ranges, assumptions={a > 0})
    negative = integrate(1, *ranges, assumptions={a < 0})
    assert sp.simplify(positive - sp.pi * a**2) == 0
    assert sp.simplify(negative + sp.pi * a**2) == 0


def test_symmetry_precedes_coordinate_transform():
    x, y = sp.symbols("x y", real=True)
    ranges = [
        (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
        (x, -1, 1),
    ]
    assert integrate(sp.sin(x), *ranges) == 0


def test_outer_first_standard_simplex_honors_requested_order():
    x, y = sp.symbols("x y", real=True)
    region = region_from_ranges([(x, 0, 1), (y, 0, 1 - x)], structural_order="outer-first")
    assert type(region).__name__ == "SimplexRegion"
    assert region.ranges == ((y, 0, 1 - x), (x, 0, 1))


def test_unknown_critical_point_membership_is_not_assumed_inside():
    x, a = sp.symbols("x a", real=True)
    assert recognition._find_critical_points((x - a) ** 2, x, 0, 1) == []


def test_nonfinite_and_nonreal_centers_rejected():
    x, y = sp.symbols("x y", real=True)
    with pytest.raises(ValueError, match="finite"):
        EllipsoidRegion(variables_nd=(x, y), axes=(1, 1), center=(sp.oo, 0))
    with pytest.raises(ValueError, match="real"):
        EllipsoidRegion(variables_nd=(x, y), axes=(1, 1), center=(sp.I, 0))


def test_signal_timeout_preserves_existing_timer():
    if not recognition._signal_timeout_ready():
        pytest.skip("POSIX interval timers unavailable")
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.setitimer(signal.ITIMER_REAL, 10.0)
        assert recognition._run_with_signal_timeout(lambda: 7, 0.5, None) is None
        assert signal.getsignal(signal.SIGALRM) is old_handler
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 8.0 < remaining <= 10.0
        assert interval == 0.0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0 or old_timer[1] > 0:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)


def test_transform_declines_unevaluated_integral():
    x, u = sp.symbols("x u", real=True)
    transform = transforms.CoordinateTransform(
        source_vars=(x,),
        target_vars=(u,),
        forward_map=(u,),
        jacobian=sp.Integer(1),
        target_ranges=((u, 0, 1),),
    )
    result = transforms._try_transform(
        transform,
        sp.exp(x**x),
        assumptions=assumptions._EMPTY_ASSUMPTIONS,
        principal_value=False,
    )
    assert result is None


def test_clear_cache_clears_decomposition_cache():
    x, y = sp.symbols("x y", real=True)
    recognition._cached_decomposition.cache_clear()
    first = recognition._cached_decomposition(sp.sin(x * y), (x, y))
    second = recognition._cached_decomposition(sp.sin(x * y), (x, y))
    assert first is second
    assert recognition._cached_decomposition.cache_info().hits >= 1
    core.clear_cache()
    assert recognition._cached_decomposition.cache_info().currsize == 0


def test_nonconstant_pushforward_density_declines_exact_shortcut():
    x, y = sp.symbols("x y", real=True)
    result = strategies._try_additive_separable(
        lambda value: value**2,
        x**2 + y**2,
        [x, y],
        [(x, 0, 1), (y, 0, 1)],
        {},
    )
    assert result is None
