"""Boundary flux identities, independent area integrals and failed prerequisites."""

import pytest
import sympy as sp

from multiple_integrate import IntegrationPolicy, StrategyBudget, integrate_result
from multiple_integrate.boundary_reduction import integrate_boundary

x, y = sp.symbols("x y", real=True)
polygon = sp.And(x >= 0, y >= 0, x <= 2, y <= 2, x + y <= 3)


@pytest.mark.parametrize("density", [1, x, y, x * y, x**2 + y**2, 2 + x**3 - 3 * y**2 + x * y, 0])
def test_flux_matches_independent_rectangle_minus_triangle(density):
    density = sp.sympify(density)
    oracle = sp.integrate(density, (y, 0, 2), (x, 0, 2)) - sp.integrate(
        density, (y, 3 - x, 2), (x, 1, 2)
    )
    result = integrate_boundary(density, polygon, (x, y))
    assert sp.simplify(result.value - oracle) == 0
    assert result.complete and result.absolute_convergent is True
    assert result.ordinary_convergent is True
    assert result.diagnostics["boundary_facets"] == 5
    assert all(c.complete for c in result.coverage)


def test_public_route_does_not_invoke_general_integrator(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("boundary moments must not invoke a general integrator")

    monkeypatch.setattr(sp, "integrate", forbidden)
    result = integrate_result(x * y, region=polygon, variables=(x, y))
    assert result.value == sp.Rational(21, 8)
    assert result.method == "polynomial_boundary"
    assert result.absolute_convergent is True and result.complete


@pytest.mark.parametrize("mapping", [{x: x + 3, y: y - 2}, {x: 2 * x, y: 3 * y}, {x: y, y: x}])
def test_affine_coordinate_variants(mapping):
    mapped_region = polygon.subs(mapping, simultaneous=True)
    mapped_density = (x * y).subs(mapping, simultaneous=True)
    jacobian = abs(sp.Matrix([mapping[x], mapping[y]]).jacobian([x, y]).det())
    result = integrate_boundary(mapped_density * jacobian, mapped_region, (x, y))
    assert result.value == sp.Rational(21, 8)


def test_redundancy_strict_boundaries_and_variable_order():
    formula = sp.And(polygon, 2 * x + 2 * y <= 6, x + y < 4, x < 10, x + y > 0)
    result = integrate_boundary(x * y, formula, (y, x))
    assert result.value == sp.Rational(21, 8)
    assert result.diagnostics["boundary_facets"] == 5


@pytest.mark.parametrize(
    "formula",
    [
        sp.And(x >= 0, y >= 0),
        sp.And(x >= 0, x <= 1, y >= 0),
        sp.And(x >= 0, x <= 0, y >= 0, y <= 1),
        sp.And(x >= 1, x <= 0, y >= 0, y <= 1),
        sp.Or(polygon, x > 5),
        sp.And(x * x + y * y <= 1, x >= 0),
        sp.And(x >= 0, y >= 0, x + y <= sp.Symbol("a", real=True)),
    ],
)
def test_broken_geometry_prerequisites_decline(formula):
    assert integrate_boundary(x * y, formula, (x, y)) is None


@pytest.mark.parametrize(
    "density", [1 / x, sp.exp(x * y), sp.Symbol("a") * x, x / sp.Symbol("a", real=True)]
)
def test_broken_density_prerequisites_decline(density):
    assert integrate_boundary(density, polygon, (x, y)) is None


def test_finite_coefficients_and_linearity():
    a = sp.Symbol("a", real=True)
    result = integrate_boundary(a * x * y + 1, polygon, (x, y))
    assert sp.simplify(result.value - (sp.Rational(21, 8) * a + sp.Rational(7, 2))) == 0


@pytest.mark.parametrize("dimension", [0, 1, 3])
def test_intrinsic_dimension_mismatch_declines(dimension):
    assert integrate_boundary(1, polygon, (x, y), measure_dimension=dimension) is None


def test_budget_precedes_large_expansion():
    policy = IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=2))
    assert integrate_boundary((x + y) ** 20, polygon, (x, y), policy=policy) is None
    policy = IntegrationPolicy(semialg=StrategyBudget(max_pieces=3))
    assert integrate_boundary(1, polygon, (x, y), policy=policy) is None


def test_axis_aligned_boxes_keep_cartesian_dispatch():
    box = sp.And(x >= 0, x <= 2, y >= 0, y <= 3)
    assert integrate_boundary(x * y, box, (x, y)) is None
    result = integrate_result(x * y, region=box, variables=(x, y))
    assert result.value == 9
    assert result.method != "polynomial_boundary"


def test_boundary_guide_example_executes():
    import re
    from pathlib import Path

    text = (Path(__file__).parents[1] / "docs/boundary-reduction.md").read_text()
    namespace = {}
    for code in re.findall(r"```python\n(.*?)```", text, re.S):
        exec(compile(code, "boundary-reduction.md", "exec"), namespace)
    assert namespace["result"].value == sp.Rational(21, 8)


def test_unevaluated_false_constant_strict_constraint_cannot_certify_area():
    formula = sp.And(sp.Lt(0, 0, evaluate=False), polygon, evaluate=False)
    assert integrate_boundary(1, formula, (x, y)) is None
