"""Exact reference corpus; rows record measure and provenance, not numerical guesses."""

import mpmath as mp
import pytest
import semialg
import sympy as sp

from multiple_integrate import IntegrationStatus, integrate, integrate_result
from multiple_integrate.adapters.semialg import reduce_domain, reduce_intrinsic_domain

x, y, z = sp.symbols("x y z", real=True)

# name, region, integrand, coordinates, measure, oracle, route
REFERENCE_CASES = [
    (
        "box",
        semialg.Box(((0, 1), (0, 2))),
        x + y,
        (x, y),
        "ambient",
        3,
        "semialgebraic_domain_reduction",
    ),
    ("point", semialg.Point((2, 3)), x + y, (x, y), 0, 5, "semialg_intrinsic_reduction"),
    ("degenerate", semialg.Box(((0, 0), (0, 1))), 1, (x, y), "ambient", 0, "ambient_null_set"),
    (
        "shell",
        semialg.SphericalShell((0, 0), (1, 2)),
        1,
        (x, y),
        "ambient",
        3 * sp.pi,
        "semialgebraic_domain_reduction",
    ),
    (
        "algebraic_boundary",
        sp.And(x >= 0, x**2 <= 2),
        1,
        (x,),
        "ambient",
        sp.sqrt(2),
        "semialgebraic_domain_reduction",
    ),
    ("unbounded", x >= 0, sp.exp(-x), (x,), "ambient", 1, "semialgebraic_domain_reduction"),
    (
        "disconnected",
        sp.Or(sp.And(x >= -2, x <= -1), sp.And(x >= 1, x <= 2)),
        x**2,
        (x,),
        "ambient",
        sp.Rational(14, 3),
        "semialgebraic_domain_reduction",
    ),
    ("empty", sp.false, 1, (x,), "ambient", 0, "semialgebraic_domain_reduction"),
    (
        "intrinsic_cover",
        sp.And(sp.Eq(y, x), x >= 0, x <= 1),
        1,
        (x, y),
        "intrinsic",
        sp.sqrt(2),
        "semialg_intrinsic_reduction",
    ),
    (
        "mixed_dimensions",
        sp.Or(sp.And(sp.Eq(y, 0), x >= 0, x <= 1), sp.And(sp.Eq(x, 2), sp.Eq(y, 2))),
        1,
        (x, y),
        1,
        1,
        "semialg_intrinsic_reduction",
    ),
]


@pytest.mark.parametrize(
    "name,region,f,variables,measure,expected,route",
    REFERENCE_CASES,
    ids=[row[0] for row in REFERENCE_CASES],
)
def test_reference_region(name, region, f, variables, measure, expected, route):
    result = integrate_result(f, region=region, variables=variables, measure_dimension=measure)
    assert sp.simplify(result.value - expected) == 0
    assert result.status is IntegrationStatus.EXACT
    assert result.method == route


@pytest.mark.parametrize(
    "a,expected_dimension,expected_length", [(-1, None, 0), (0, 0, 0), (1, 1, 2), (4, 1, 4)]
)
@pytest.mark.parametrize("rewrite", [False, True])
@pytest.mark.parametrize("rename", [False, True])
def test_parameter_transition_cover(a, expected_dimension, expected_length, rewrite, rename):
    """x²≤a on y=0: empty, point, segment; no transition is omitted."""
    u, v = sp.symbols("u v", real=True) if rename else (x, y)
    formula = sp.And(sp.Eq(v, 0), u**2 <= a)
    if rewrite:
        formula = sp.And(sp.Eq(2 * v, 0), 2 * u**2 <= 2 * a)
    cover = reduce_intrinsic_domain(1, formula, variables=(u, v))
    if expected_dimension is None:
        assert cover is None  # no verified nonempty intrinsic cover
        assert integrate(1, region=formula, variables=(u, v)) == 0
        return
    assert cover is not None and cover.certified
    assert cover.dimension == expected_dimension
    assert not cover.singular_strata
    if a > 0:
        # Check actual chart coverage, not just the sum of chart integrals.
        assert len(cover.pieces) == 1
        piece = cover.pieces[0]
        assert piece.limits == ((u, -sp.sqrt(a), sp.sqrt(a)),)
        assert piece.metric_factor == 1
        assert (
            integrate(1, region=formula, variables=(u, v), measure_dimension=1) == expected_length
        )
    else:
        assert integrate(1, region=formula, variables=(u, v), measure_dimension=0) == 1
    assert integrate(1, region=formula, variables=(u, v), measure_dimension="ambient") == 0


@pytest.mark.parametrize(
    "a,expected", [(-1, 0), (0, 0), (sp.Rational(1, 4), 2 - sp.sqrt(3)), (1, 2), (4, 2)]
)
def test_topology_and_boundary_coincidence(a, expected):
    # Two components for 0<a<1; their inner boundaries coincide at a=1.
    formula = sp.And(x**2 >= 1 - a, x**2 <= 1)
    assert integrate(1, region=formula, variables=(x,)) == expected


@pytest.mark.parametrize("p,q", [(0, 0), (1, 0), (0, 2), (2, 3)])
def test_differential_orders_decomposition_and_numeric(p, q):
    f = x**p * y**q
    formula = sp.And(x >= 0, x <= 2, y >= 0, y <= 3)
    geometric = integrate(f, region=formula, variables=(x, y))
    direct = integrate(f, (x, 0, 2), (y, 0, 3))
    reverse = integrate(f, (y, 0, 3), (x, 0, 2))
    reduction = reduce_domain(f, formula, variables=(x, y))
    assert reduction is not None
    explicit = sum(integrate(piece.integrand, *piece.limits) for piece in reduction.pieces)
    oracle = sp.Rational(2 ** (p + 1) * 3 ** (q + 1), (p + 1) * (q + 1))
    assert geometric == direct == reverse == explicit == oracle
    with mp.workdps(60):
        numeric = mp.quad(lambda t: t**p, [0, 2]) * mp.quad(lambda t: t**q, [0, 3])
        assert abs(numeric - mp.mpf(str(sp.N(oracle, 60)))) < mp.mpf("1e-55")


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_transformed_vs_geometric(radius):
    r, theta = sp.symbols("r theta", real=True)
    # r is the Jacobian, r² the transformed integrand.
    polar = integrate(r**3, (r, 0, radius), (theta, 0, 2 * sp.pi))
    geometric = integrate(x**2 + y**2, region=semialg.Ball((0, 0), radius), variables=(x, y))
    assert polar == geometric == sp.pi * radius**4 / 2


@pytest.mark.parametrize("shift,scale", [(0, 1), (2, 3), (-3, 2)])
def test_metamorphic_translation_scaling_linearity(shift, scale):
    u, v = sp.symbols("u v", real=True)
    f = x + 2 * y
    baseline = integrate(f, (x, 0, 1), (y, 0, 2))
    transformed = f.subs({x: (u - shift) / scale, y: (v - shift) / scale})
    value = integrate(transformed, (u, shift, shift + scale), (v, shift, shift + 2 * scale))
    assert value == scale**2 * baseline
    assert integrate(3 * f + 2, (x, 0, 1), (y, 0, 2)) == 3 * baseline + 4
    assert integrate(f, (x, 0, 1), (y, 0, 1)) + integrate(f, (x, 0, 1), (y, 1, 2)) == baseline
    assert integrate(f.subs({x: v, y: u}), (v, 0, 1), (u, 0, 2)) == baseline


def test_symmetry_prerequisite_pair():
    assert integrate(x**3, (x, -1, 1)) == 0
    assert integrate(x**3, (x, -1, 2)) == sp.Rational(15, 4)


def test_geometry_unknown_is_not_empty(monkeypatch):
    import multiple_integrate.adapters.semialg as adapter

    monkeypatch.setattr(adapter, "_intrinsic_parametric_cover", lambda *args, **kwargs: None)
    assert reduce_intrinsic_domain(1, sp.Eq(y, x), variables=(x, y)) is None


@pytest.mark.parametrize("a,b", [(1, 2), (0, 2), (2, 0)])
def test_parameter_permutation(a, b):
    p, q = sp.symbols("p q", real=True)
    family = sp.And(x >= 0, x <= p, y >= 0, y <= q)
    first = family.subs({p: a, q: b})
    permuted = family.xreplace({x: y, y: x, p: q, q: p}).subs({p: b, q: a})
    assert integrate(1, region=first, variables=(x, y)) == a * b
    assert integrate(1, region=permuted, variables=(y, x)) == a * b


@pytest.mark.parametrize("a,dimension,value", [(0, 0, 1), (1, 1, 2)])
def test_rank_drop_in_native_geometry(a, dimension, value):
    geometry = semialg.Simplex(((0, 0), (2 * a, 0))) if a else semialg.Point((0, 0))
    assert integrate(1, region=geometry, variables=(x, y), measure_dimension=dimension) == value
    assert integrate(1, region=geometry, variables=(x, y), measure_dimension="ambient") == 0


def test_intrinsic_surface_vs_ambient_and_explicit_chart():
    formula = sp.And(sp.Eq(z, x + y), x >= 0, x <= 1, y >= 0, y <= 1)
    cover = reduce_intrinsic_domain(1, formula, variables=(x, y, z))
    assert cover is not None and cover.dimension == 2
    assert all(sp.simplify(piece.metric_factor - sp.sqrt(3)) == 0 for piece in cover.pieces)
    assert integrate(1, region=formula, variables=(x, y, z), measure_dimension=2) == sp.sqrt(3)
    assert integrate(1, region=formula, variables=(x, y, z), measure_dimension="ambient") == 0


def test_finite_fiber_prerequisite_breaks_at_transition():
    # a*x=0: counting measure of one point becomes an infinite line at a=0.
    a = sp.Symbol("a", real=True)
    formula = sp.Eq(a * x, 0)
    assert integrate(1, region=formula.subs(a, 1), variables=(x,), measure_dimension=0) == 1
    # The transitioned fiber must not be treated as a single point.
    assert integrate(1, region=formula.subs(a, 0), variables=(x,), measure_dimension=1) == sp.oo


def test_noninvertible_change_requires_both_branches():
    u = sp.Symbol("u", real=True)
    original = integrate(x**2, (x, -1, 1))
    one_branch = integrate(sp.sqrt(u) / 2, (u, 0, 1))
    assert original == 2 * one_branch == sp.Rational(2, 3)
    assert original != one_branch


def test_nonsemialgebraic_intrinsic_input_is_unsupported():
    with pytest.raises(NotImplementedError):
        integrate(1, region=sp.Eq(y, sp.sin(x)), variables=(x, y), measure_dimension=1)


def test_unverified_dimension_does_not_produce_zero(monkeypatch):
    import multiple_integrate.adapters.semialg as adapter
    import multiple_integrate.core as core

    monkeypatch.setattr(core, "reduce_domain", lambda *args, **kwargs: None)
    monkeypatch.setattr(adapter, "region_dimension", lambda *args, **kwargs: None)
    with pytest.raises(NotImplementedError, match="certified cylindrical"):
        integrate(1, region=sp.Eq(z, x + y), variables=(x, y, z))


def test_symbolic_parameter_box_and_exhaustive_cells():
    a, b = sp.symbols("a b", positive=True)
    value = integrate(1, region=semialg.Box(((0, a), (0, b))), variables=(x, y))
    assert value == a * b
    t = sp.Symbol("t", real=True)
    cells = (t < 0, sp.Eq(t, 0), t > 0)
    assert sp.simplify(sp.Not(sp.Or(*cells))) is sp.false
    for i, cell in enumerate(cells):
        for other in cells[i + 1 :]:
            assert sp.simplify(sp.And(cell, other)) is sp.false
