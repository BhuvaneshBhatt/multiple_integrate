import pytest
import sympy as sp

from multiple_integrate import (
    AffineSimplexRegion,
    AnnulusRegion,
    BallRegion,
    BoxRegion,
    DiskRegion,
    EllipsoidRegion,
    GraphRegion,
    IteratedRegion,
    Region,
    SimplexRegion,
    SphericalShellRegion,
    UnionRegion,
    multiple_integrate,
)

x, y, z = sp.symbols("x y z", real=True)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: Region(((x, 0, 1), (x, 0, 2))), "distinct"),
        (lambda: Region(((x, 0, sp.nan),)), "nan or zoo"),
        (lambda: BoxRegion(((x, 0, y), (y, 0, 1))), "independent"),
        (lambda: IteratedRegion(((x, 0, y), (y, 0, x))), "consistent nested"),
        (lambda: GraphRegion(((x, 0, 1),), outer_var=x, inner_var=y), "exactly two"),
        (lambda: SimplexRegion(((x, 0, 1),), dimension=2), "dimension"),
        (
            lambda: AffineSimplexRegion(
                ((y, 0, 1 - x), (x, 0, 1)),
                shifts=(0, 0),
                scales=(1, 0),
                dimension=2,
            ),
            "nonzero",
        ),
        (
            lambda: DiskRegion(
                ((y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)),
                radius=-1,
            ),
            "nonnegative",
        ),
        (lambda: AnnulusRegion(variables_xy=(x, y), inner_radius=2, outer_radius=1), "smaller"),
        (
            lambda: BallRegion(
                ((z, -1, 1), (y, -1, 1), (x, -1, 1)),
                radius=1,
                dimension=2,
            ),
            "dimension",
        ),
        (lambda: EllipsoidRegion(variables_nd=(x, y), axes=(2, 0)), "positive"),
        (
            lambda: SphericalShellRegion(variables_nd=(x, y, z), inner_radius=3, outer_radius=2),
            "smaller",
        ),
        (lambda: UnionRegion(pieces=()), "at least one"),
    ],
)
def test_region_constructor_invariants(factory, match):
    with pytest.raises((TypeError, ValueError), match=match):
        factory()


def test_symbolic_region_parameters_are_allowed_when_not_provably_invalid():
    r, R = sp.symbols("r R", positive=True)
    region = AnnulusRegion(variables_xy=(x, y), inner_radius=r, outer_radius=R)
    assert region.inner_radius == r
    assert region.outer_radius == R


def test_union_region_accepts_touching_measure_zero_boundaries():
    inner = AnnulusRegion(variables_xy=(x, y), inner_radius=0, outer_radius=1)
    outer = AnnulusRegion(variables_xy=(x, y), inner_radius=1, outer_radius=2)
    union = UnionRegion(pieces=(inner, outer))
    assert union.constant_volume() == 4 * sp.pi


def test_union_region_rejects_provably_overlapping_pieces():
    first = AnnulusRegion(variables_xy=(x, y), inner_radius=0, outer_radius=2)
    second = AnnulusRegion(variables_xy=(x, y), inner_radius=1, outer_radius=3)
    with pytest.raises(ValueError, match="disjoint"):
        UnionRegion(pieces=(first, second))


def test_union_region_rejects_duplicate_piece():
    disk = DiskRegion(
        ((y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)),
        radius=1,
    )
    with pytest.raises(ValueError, match="disjoint"):
        UnionRegion(pieces=(disk, disk))


def test_union_region_rejects_mismatched_ordered_variables():
    xy = AnnulusRegion(variables_xy=(x, y), inner_radius=0, outer_radius=1)
    xz = AnnulusRegion(variables_xy=(x, z), inner_radius=1, outer_radius=2)
    with pytest.raises(ValueError, match="same ordered variables"):
        UnionRegion(pieces=(xy, xz))


@pytest.mark.parametrize(
    ("region", "method", "expr", "expected"),
    [
        (
            AnnulusRegion(variables_xy=(x, y), inner_radius=1, outer_radius=2),
            "constant_volume",
            None,
            3 * sp.pi,
        ),
        (
            AnnulusRegion(variables_xy=(x, y), inner_radius=1, outer_radius=2),
            "polynomial_moment",
            x**2,
            sp.Rational(15, 4) * sp.pi,
        ),
        (
            AnnulusRegion(variables_xy=(x, y), inner_radius=1, outer_radius=2),
            "radial_integral",
            x**2 + y**2,
            sp.Rational(15, 2) * sp.pi,
        ),
        (
            EllipsoidRegion(variables_nd=(x, y), axes=(2, 3)),
            "constant_volume",
            None,
            6 * sp.pi,
        ),
        (
            EllipsoidRegion(variables_nd=(x, y), axes=(2, 3)),
            "polynomial_moment",
            x**2 + y**2,
            sp.Rational(39, 2) * sp.pi,
        ),
        (
            EllipsoidRegion(variables_nd=(x, y), axes=(2, 3)),
            "radial_integral",
            x**2 / 4 + y**2 / 9,
            3 * sp.pi,
        ),
        (
            SphericalShellRegion(variables_nd=(x, y, z), inner_radius=1, outer_radius=2),
            "constant_volume",
            None,
            sp.Rational(28, 3) * sp.pi,
        ),
        (
            SphericalShellRegion(variables_nd=(x, y, z), inner_radius=1, outer_radius=2),
            "polynomial_moment",
            x**2,
            sp.Rational(124, 15) * sp.pi,
        ),
        (
            SphericalShellRegion(variables_nd=(x, y, z), inner_radius=1, outer_radius=2),
            "radial_integral",
            x**2 + y**2 + z**2,
            sp.Rational(124, 5) * sp.pi,
        ),
    ],
)
def test_region_method_coverage_matrix(region, method, expr, expected):
    fn = getattr(region, method)
    result = fn() if expr is None else fn(expr)
    assert sp.simplify(result - expected) == 0


def test_union_region_method_coverage_matrix():
    first = AnnulusRegion(variables_xy=(x, y), inner_radius=0, outer_radius=1)
    second = AnnulusRegion(variables_xy=(x, y), inner_radius=1, outer_radius=2)
    union = UnionRegion(pieces=(first, second))
    assert sp.simplify(union.constant_volume() - 4 * sp.pi) == 0
    assert sp.simplify(union.polynomial_moment(x**2) - 4 * sp.pi) == 0
    assert sp.simplify(union.radial_integral(x**2 + y**2) - 8 * sp.pi) == 0
    assert union.is_reflection_invariant(x)
    assert union.is_reflection_invariant(y)


def test_region_matrix_rejects_nonradial_ellipsoid_integrand():
    region = EllipsoidRegion(variables_nd=(x, y), axes=(2, 3))
    assert region.radial_integral(sp.exp(x * y) - 1) is None


def test_region_methods_are_used_by_multiple_integrate():
    shell = SphericalShellRegion(variables_nd=(x, y, z), inner_radius=1, outer_radius=2)
    assert (
        sp.simplify(multiple_integrate(x**2 + y**2 + z**2, shell) - sp.Rational(124, 5) * sp.pi)
        == 0
    )
