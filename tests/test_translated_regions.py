import sympy as sp

from multiple_integrate import (
    BallRegion,
    DiskRegion,
    EllipsoidRegion,
    multiple_integrate,
    region_from_ranges,
)


def test_translated_disk_is_classified_and_integrated():
    x, y = sp.symbols("x y", real=True)
    ranges = [
        (y, -2 - sp.sqrt(9 - (x - 1) ** 2), -2 + sp.sqrt(9 - (x - 1) ** 2)),
        (x, -2, 4),
    ]
    region = region_from_ranges(ranges, structural_order="inner-first")
    assert isinstance(region, DiskRegion)
    assert region.center == (-2, 1)  # center follows region.variables == (y, x)
    assert region.radius == 3
    assert sp.simplify(region.constant_volume() - 9 * sp.pi) == 0
    shifted_r2 = (x - 1) ** 2 + (y + 2) ** 2
    assert sp.simplify(multiple_integrate(shifted_r2, *ranges) - sp.Rational(81, 2) * sp.pi) == 0


def test_translated_ball_is_classified_and_integrated():
    x, y, z = sp.symbols("x y z", real=True)
    ranges = [
        (
            z,
            3 - sp.sqrt(4 - (x - 1) ** 2 - (y + 2) ** 2),
            3 + sp.sqrt(4 - (x - 1) ** 2 - (y + 2) ** 2),
        ),
        (y, -2 - sp.sqrt(4 - (x - 1) ** 2), -2 + sp.sqrt(4 - (x - 1) ** 2)),
        (x, -1, 3),
    ]
    region = region_from_ranges(ranges, structural_order="inner-first")
    assert isinstance(region, BallRegion)
    assert region.center == (3, -2, 1)
    assert region.dimension == 3
    assert sp.simplify(region.constant_volume() - sp.Rational(32, 3) * sp.pi) == 0
    shifted_r2 = (x - 1) ** 2 + (y + 2) ** 2 + (z - 3) ** 2
    assert sp.simplify(region.radial_integral(shifted_r2) - sp.Rational(128, 5) * sp.pi) == 0
    assert sp.simplify(multiple_integrate(shifted_r2, *ranges) - sp.Rational(128, 5) * sp.pi) == 0


def test_translated_ellipsoid_is_classified_and_integrated():
    x, y = sp.symbols("x y", real=True)
    ranges = [
        (
            y,
            -2 - 3 * sp.sqrt(1 - (x - 1) ** 2 / 4),
            -2 + 3 * sp.sqrt(1 - (x - 1) ** 2 / 4),
        ),
        (x, -1, 3),
    ]
    region = region_from_ranges(ranges, structural_order="inner-first")
    assert isinstance(region, EllipsoidRegion)
    assert region.variables == (y, x)
    assert region.center == (-2, 1)
    assert region.axes == (3, 2)
    assert sp.simplify(region.constant_volume() - 6 * sp.pi) == 0
    normalized_r2 = (x - 1) ** 2 / 4 + (y + 2) ** 2 / 9
    assert sp.simplify(region.radial_integral(normalized_r2) - 3 * sp.pi) == 0
    assert sp.simplify(multiple_integrate(normalized_r2, *ranges) - 3 * sp.pi) == 0
    assert sp.simplify(multiple_integrate(1, *ranges) - 6 * sp.pi) == 0


def test_direct_translated_disk_polynomial_moment_uses_center():
    x, y = sp.symbols("x y", real=True)
    region = DiskRegion(
        ((y, -2 - sp.sqrt(9 - (x - 1) ** 2), -2 + sp.sqrt(9 - (x - 1) ** 2)), (x, -2, 4)),
        radius=3,
        center=(-2, 1),
    )
    assert (
        sp.simplify(
            region.polynomial_moment((x - 1) ** 2 + (y + 2) ** 2) - sp.Rational(81, 2) * sp.pi
        )
        == 0
    )


def test_translated_ellipsoid_reflection_invariance_respects_center():
    x, y = sp.symbols("x y", real=True)
    region = EllipsoidRegion(variables_nd=(x, y), axes=(2, 3), center=(1, 0))
    assert not region.is_reflection_invariant(x)
    assert region.is_reflection_invariant(y)
