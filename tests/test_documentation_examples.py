import semialg
import sympy as sp

from multiple_integrate import integrate, region_from_ranges


def test_quickstart_domains():
    x, y = sp.symbols("x y", real=True)
    triangle_ranges = [(y, 0, 1 - x), (x, 0, 1)]
    domain = region_from_ranges(triangle_ranges)
    disk = semialg.Ball((0, 0), 2)

    assert integrate(1, *triangle_ranges) == sp.Rational(1, 2)
    assert domain.constant_volume() == sp.Rational(1, 2)
    assert disk.measure() == 4 * sp.pi
    assert integrate(1, region=disk, variables=(x, y)) == 4 * sp.pi


def test_region_guide_values():
    x, y = sp.symbols("x y", real=True)
    box = semialg.Box(((0, 2), (-1, 3)))
    triangle = semialg.Simplex(((0, 0), (1, 0), (0, 1)))
    ellipse = semialg.Ellipsoid((0, 0), ((4, 0), (0, 9)))

    assert box.measure() == 8
    assert box.centroid() == (1, 1)
    assert integrate(x + y, region=box, variables=(x, y)) == 16
    assert triangle.measure() == sp.Rational(1, 2)
    assert integrate(x + y, region=triangle, variables=(x, y)) == sp.Rational(1, 3)
    assert ellipse.measure() == 6 * sp.pi
    assert integrate(1, region=ellipse, variables=(x, y)) == 6 * sp.pi


def test_intrinsic_guide_values():
    x, y = sp.symbols("x y", real=True)
    circle = semialg.Sphere((0, 0), 2)

    assert circle.measure() == 4 * sp.pi
    assert integrate(1, region=circle, variables=(x, y)) == 0
    assert (
        integrate(
            1,
            region=circle,
            variables=(x, y),
            measure_dimension="intrinsic",
        )
        == 4 * sp.pi
    )
