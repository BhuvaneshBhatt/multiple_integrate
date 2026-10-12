"""Boundary exponentials, whole parameter cells, oriented loops and facet recursion."""

import pytest
import sympy as sp

from multiple_integrate import (
    IntegrationPolicy,
    ParameterCell,
    StrategyBudget,
    clear_cache,
    integrate_cells,
    integrate_result,
)
from multiple_integrate.boundary_reduction import (
    _convex_facets,
    _polygon_moment,
    integrate_boundary,
)

x, y, z, a, b = sp.symbols("x y z a b", real=True)
triangle = sp.And(x >= 0, y >= 0, x + y <= 1)
L = sp.Or(sp.And(x >= 0, x <= 2, y >= 0, y <= 1), sp.And(x >= 0, x <= 1, y >= 1, y <= 2))
hole = sp.And(x >= 0, x <= 3, y >= 0, y <= 3, sp.Not(sp.And(x > 1, x < 2, y > 1, y < 2)))


@pytest.mark.parametrize(
    "density",
    [sp.exp(x + y), sp.exp(x - y), x * sp.exp(x), (1 + x * y) * sp.exp(2 * x - y), sp.exp(y + 3)],
)
def test_affine_exponential_edge_recurrences(density):
    oracle = sp.integrate(density, (y, 0, 1 - x), (x, 0, 1))
    result = integrate_boundary(density, triangle, (x, y))
    assert result is not None and sp.simplify(result.value - oracle) == 0
    assert result.method == "affine_exponential_boundary"
    assert result.absolute_convergent is True and result.complete


def test_zero_frequency_and_parameter_cell_edges():
    result = integrate_cells(
        sp.exp(a * x),
        region=triangle,
        variables=(x, y),
        parameters=(a,),
        cells=(ParameterCell(a < 0), ParameterCell(sp.Eq(a, 0)), ParameterCell(a > 0)),
    )
    assert result.complete
    assert result.value.subs(a, 0) == sp.Rational(1, 2)
    assert sp.simplify(result.value.subs(a, 1) - (sp.E - 2)) == 0
    assert all(child.absolute_convergent is True for child in result.diagnostics["cell_results"])


def test_unproved_frequency_declines_without_dividing_by_zero():
    assert integrate_boundary(sp.exp(a * x), triangle, (x, y)) is None
    result = integrate_boundary(sp.exp(a), triangle, (x, y))
    assert result.value == sp.exp(a) / 2


@pytest.mark.parametrize("density", [1, x * y, x**3 + y**2])
def test_symbolic_fixed_combinatorics(density):
    region = sp.And(x >= 0, y >= 0, x + y <= a, y <= b)
    assumptions = sp.And(a > b, b > 0)
    result = integrate_boundary(density, region, (x, y), assumptions=assumptions)
    oracle = sp.integrate(density, (x, 0, a - y), (y, 0, b))
    assert result is not None and sp.simplify(result.value - oracle) == 0
    assert result.complete and result.absolute_convergent is True


def test_topology_threshold_and_dimension_drop_cells():
    region = sp.And(x >= 0, y >= 0, x + y <= a, y <= b)
    result = integrate_cells(
        1,
        region=region,
        variables=(x, y),
        parameters=(a, b),
        assumptions=sp.And(a > 0, b >= 0),
        cells=(
            ParameterCell(sp.Eq(b, 0)),
            ParameterCell(sp.And(b > 0, b < a)),
            ParameterCell(sp.Eq(b, a)),
            ParameterCell(b > a),
        ),
    )
    assert result.complete
    for substitutions, expected in [
        ({a: 2, b: 0}, 0),
        ({a: 2, b: 1}, sp.Rational(3, 2)),
        ({a: 2, b: 2}, 2),
        ({a: 2, b: 3}, 2),
    ]:
        assert result.value.subs(substitutions) == expected


def test_geometry_and_moments_are_reused_without_evidence_leakage():
    clear_cache()
    first = integrate_boundary(x * y, triangle, (x, y))
    info = _polygon_moment.cache_info()
    second = integrate_boundary(7 * x * y, triangle, (x, y))
    assert second.value == 7 * first.value
    assert _polygon_moment.cache_info().hits > info.hits
    assert _convex_facets.cache_info().hits > 0
    strict = integrate_boundary(x * y, sp.And(x > 0, y > 0, x + y < 1), (x, y))
    assert (
        dict(strict.coverage[0].details)["source_region"]
        != dict(first.coverage[0].details)["source_region"]
    )
    clear_cache()
    assert _polygon_moment.cache_info().currsize == _convex_facets.cache_info().currsize == 0


def test_cached_high_degree_does_not_bypass_a_later_budget():
    integrate_boundary(x**6, triangle, (x, y))
    policy = IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=2))
    assert integrate_boundary(x**6, triangle, (x, y), policy=policy) is None


@pytest.mark.parametrize(
    "region,ranges,subtract",
    [
        (L, ((y, 0, 2), (x, 0, 2)), ((y, 1, 2), (x, 1, 2))),
        (hole, ((y, 0, 3), (x, 0, 3)), ((y, 1, 2), (x, 1, 2))),
    ],
)
@pytest.mark.parametrize("density", [1, x * y, sp.exp(x + y), (x + 1) * sp.exp(x - y)])
def test_nonconvex_and_hole_boundary_loops(region, ranges, subtract, density):
    oracle = sp.integrate(density, *ranges) - sp.integrate(density, *subtract)
    result = integrate_boundary(density, region, (x, y))
    assert result is not None and sp.simplify(result.value - oracle) == 0
    assert result.complete and result.absolute_convergent is True


def test_overlapping_union_counts_overlap_once():
    region = sp.Or(sp.And(x >= 0, x <= 2, y >= 0, y <= 2), sp.And(x >= 1, x <= 3, y >= 0, y <= 2))
    assert integrate_boundary(1, region, (x, y)).value == 6


def test_disconnected_loops_and_unbounded_exterior():
    region = sp.Or(sp.And(x >= 0, x <= 1, y >= 0, y <= 1), sp.And(x >= 2, x <= 3, y >= 0, y <= 1))
    assert integrate_boundary(1, region, (x, y)).value == 2
    assert integrate_boundary(1, sp.Not(hole), (x, y)) is None
    assert integrate_boundary(1, sp.Or(triangle, x > 2), (x, y)) is None


@pytest.mark.parametrize(
    "density,expected",
    [
        (1, sp.Rational(1, 6)),
        (x * y * z, sp.Rational(1, 720)),
        (x * x, sp.Rational(1, 60)),
        (1 + x - y + z, sp.Rational(5, 24)),
    ],
)
def test_recursive_tetrahedron_matches_dirichlet(density, expected):
    region = sp.And(x >= 0, y >= 0, z >= 0, x + y + z <= 1)
    result = integrate_boundary(density, region, (x, y, z))
    assert result is not None and result.value == expected
    assert result.diagnostics["boundary_facets"] == 4
    assert result.absolute_convergent is True


def test_translated_scaled_tetrahedron_and_redundant_facets():
    mapping = {x: 2 * x + 3, y: y - 2, z: 3 * z + 1}
    region = sp.And(x >= 0, y >= 0, z >= 0, x + y + z <= 1, x + y + z <= 2)
    result = integrate_boundary(
        6 * (x * y * z).subs(mapping, simultaneous=True),
        region.subs(mapping, simultaneous=True),
        (z, x, y),
    )
    assert result is not None and result.value == sp.Rational(1, 720)


def test_prism_with_a_nontriangular_face():
    region = sp.And(x >= 0, y >= 0, x <= 2, y <= 2, x + y <= 3, z >= 0, z <= 2)
    assert integrate_boundary(x * y * z, region, (x, y, z)).value == sp.Rational(21, 4)


def test_unbounded_and_flat_polyhedra_decline():
    assert integrate_boundary(1, sp.And(x >= 0, y >= 0, z >= 0, x + y <= 1), (x, y, z)) is None
    assert (
        integrate_boundary(1, sp.And(x >= 0, y >= 0, z >= 0, z <= 0, x + y <= 1), (x, y, z)) is None
    )


def test_public_polytope_and_nonconvex_routes_avoid_backend(monkeypatch):
    monkeypatch.setattr(
        sp,
        "integrate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
    )
    for density, region, variables, expected in [
        (sp.exp(x + y), triangle, (x, y), 1),
        (1, L, (x, y), 3),
        (x * y * z, sp.And(x >= 0, y >= 0, z >= 0, x + y + z <= 1), (x, y, z), sp.Rational(1, 720)),
    ]:
        result = integrate_result(density, region=region, variables=variables)
        assert result.value == expected and result.complete


def test_loop_arrangement_budget_declines_before_geometry():
    policy = IntegrationPolicy(semialg=StrategyBudget(max_pieces=2))
    assert integrate_boundary(1, L, (x, y), policy=policy) is None


def test_collapsed_symbolic_polygon_has_only_ambient_zero():
    region = sp.And(x >= 0, y >= 0, x + y <= a, y <= b)
    assumptions = sp.And(a > 0, sp.Eq(b, 0))
    result = integrate_boundary(sp.exp(x), region, (x, y), assumptions=assumptions)
    assert result.value == 0 and result.complete
    assert dict(result.coverage[0].details)["basis"] == "certified zero ambient area"
    for measure in ("intrinsic", "top", 1):
        assert (
            integrate_boundary(
                1, region, (x, y), assumptions=assumptions, measure_dimension=measure
            )
            is None
        )


def test_parameter_cell_geometry_cache_keeps_incidence_conditions():
    clear_cache()
    region = sp.And(x >= 0, y >= 0, x + y <= a, y <= b)
    clipped = integrate_boundary(1, region, (x, y), assumptions=sp.And(a > b, b > 0))
    triangle_result = integrate_boundary(1, region, (x, y), assumptions=sp.And(b > a, a > 0))
    assert sp.simplify(clipped.value - (a * b - b * b / 2)) == 0
    assert triangle_result.value == a * a / 2
    assert integrate_boundary(1, region, (x, y)) is None
    assert clipped.assumptions != triangle_result.assumptions


def test_recursive_face_edge_budget_cannot_be_hidden_by_four_faces():
    region = sp.And(x >= 0, y >= 0, z >= 0, x + y + z <= 1)
    policy = IntegrationPolicy(semialg=StrategyBudget(max_pieces=5))
    assert integrate_boundary(1, region, (x, y, z), policy=policy) is None


def test_api_and_boundary_guide_examples_execute():
    import re
    from pathlib import Path

    root = Path(__file__).parents[1]
    for page in ("docs/api.md", "docs/boundary-reduction.md"):
        namespace = {}
        text = (root / page).read_text()
        for code in re.findall(r"```python\n(.*?)```", text, re.S):
            exec(compile(code, page, "exec"), namespace)
