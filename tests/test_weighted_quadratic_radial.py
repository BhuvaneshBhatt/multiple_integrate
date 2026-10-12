"""Polynomial moments use independent symbolic and numerical references."""

import mpmath as mp
import pytest
import semialg
import sympy as sp

from multiple_integrate import clear_cache, integrate_result
from multiple_integrate.quadratic_radial import (
    try_weighted_quadratic,
    try_weighted_radial_region,
    weighted_quadratic_data,
)
from multiple_integrate.region_types import BallRegion

x, y, a = sp.symbols("x y a", real=True)
FULL = ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
q = 2 * x * x - 2 * x * y + 2 * y * y


@pytest.mark.parametrize(
    "density,expected",
    [
        ((1 + x * y) * sp.exp(-q), 7 * sp.sqrt(3) * sp.pi / 18),
        (x * x * y * y * sp.exp(-q), sp.sqrt(3) * sp.pi / 18),
        (x * x / (1 + x * x + y * y) ** 3, sp.pi / 4),
        (x**4 / (1 + x * x + y * y) ** 4, sp.pi / 8),
        (x * x * y * y / (1 + x * x + y * y) ** 4, sp.pi / 24),
        (x * y / (1 + 2 * x * x + 2 * x * y + 2 * y * y) ** 3, -sp.sqrt(3) * sp.pi / 36),
        (x * y / (1 + (x - 1) ** 2 + (y + 2) ** 2) ** 3, -sp.pi),
    ],
)
def test_full_space_weighted_quadratic_moments(density, expected):
    r = integrate_result(density, *FULL)
    assert sp.simplify(r.value - expected) == 0
    assert r.absolute_convergent is True and r.ordinary_convergent is True
    assert any(attempt.method == "weighted_quadratic" for attempt in r.attempts)


def test_gaussian_weighted_moment_matches_direct_symbolic_integration():
    f = (1 + x * y) * sp.exp(-q)
    reference = sp.integrate(f, *FULL)
    assert sp.simplify(try_weighted_quadratic(f, FULL) - reference) == 0


def test_quadratic_parameters_require_proved_positive_definiteness():
    f = (1 + x * x) * sp.exp(-a * (x * x + y * y))
    assert weighted_quadratic_data(f, (x, y)) is None
    r = integrate_result(f, *FULL, assumptions=a > 0)
    assert sp.simplify(r.value - sp.pi / a - sp.pi / (2 * a * a)) == 0
    assert r.absolute_convergent is True


@pytest.mark.parametrize(
    "density",
    [
        x * x / (1 + x * x + y * y) ** 2,
        x / (x * x + y * y) ** 2,
        sp.exp(-((x - y) ** 2)),
        sp.exp(-x * x + y * y),
        (1 + x * x + y * y) ** x,
    ],
)
def test_broken_tail_rank_or_branch_prerequisites_decline(density):
    assert weighted_quadratic_data(density, (x, y)) is None


def test_finite_ball_gaussian_moment_and_cached_evidence():
    f = x * x * sp.exp(-x * x - y * y)
    region = semialg.Ball((0, 0), 1)
    clear_cache()
    first = integrate_result(f, region=region, variables=(x, y))
    second = integrate_result(f, region=region, variables=(x, y))
    assert first.value == second.value == sp.pi / 2 - sp.pi / sp.E
    assert first.absolute_convergent is True and second.absolute_convergent is True


def test_ball_moment_matches_high_precision_cartesian_quadrature():
    with mp.workdps(28):
        reference = mp.quad(
            lambda xx: mp.quad(
                lambda yy: xx * xx * mp.exp(-xx * xx - yy * yy),
                [-mp.sqrt(1 - xx * xx), mp.sqrt(1 - xx * xx)],
            ),
            [-1, 0, 1],
        )
        exact = mp.pi / 2 - mp.pi / mp.e
        assert abs(reference - exact) < mp.mpf("1e-24")


def test_shell_local_singularity_is_excluded_geometrically():
    r = integrate_result(
        x * x / (x * x + y * y) ** 2,
        region=semialg.SphericalShell((0, 0), (1, 2)),
        variables=(x, y),
    )
    assert r.value == sp.pi * sp.log(2) and r.absolute_convergent is True


def test_rotated_ellipsoid_weighted_radial_moment():
    shape = sp.Matrix([[2, 1], [1, 2]])
    vector = sp.Matrix([x, y])
    rho = (vector.T * shape.inv() * vector)[0]
    f = x * y * sp.exp(-rho)
    r = integrate_result(f, region=semialg.Ellipsoid((0, 0), shape), variables=(x, y))
    assert sp.simplify(r.value - sp.sqrt(3) * (sp.pi / 2 - sp.pi / sp.E)) == 0
    assert r.absolute_convergent is True


def test_radial_absolute_check_precedes_odd_cancellation():
    region = BallRegion(
        ((x, -1, 1), (y, -sp.sqrt(1 - x * x), sp.sqrt(1 - x * x))), radius=sp.S.One, dimension=2
    )
    assert try_weighted_radial_region(x / (x * x + y * y) ** 2, region) is None
    assert try_weighted_radial_region(x / (x * x + y * y), region) == 0
