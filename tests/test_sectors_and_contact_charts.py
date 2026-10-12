"""Exact sector, graph-cell and logarithmic contact regression oracles."""

import re
from pathlib import Path

import mpmath as mp
import pytest
import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    ParameterCell,
    integrate_cells,
    integrate_result,
    regularized_result,
    verify_regularization_result,
)
from multiple_integrate.assumptions import _normalize_assumptions
from multiple_integrate.cell_graphs import integrate_partitioned_cell_graph
from multiple_integrate.cell_proofs import nonnegative
from multiple_integrate.quadratic_sectors import integrate_quadratic_sector
from multiple_integrate.tangent_cutoff import tangent_graph_cutoff
from multiple_integrate.transition_cells import linear_quadratic_cells

x, y, a = sp.symbols("x y a", real=True)


@pytest.mark.parametrize(
    "density,expected",
    [(1, sp.pi / 4), (x * y, sp.Rational(1, 8)), (x**2, sp.pi / 16), (x, sp.Rational(1, 3))],
)
@pytest.mark.parametrize("variables", [(x, y), (y, x)])
def test_quadrant_moments(density, expected, variables):
    r = integrate_quadratic_sector(density, sp.And(x * x + y * y <= 1, x >= 0, y >= 0), variables)
    assert sp.simplify(r.value - expected) == 0
    assert r.complete and r.absolute_convergent and r.coverage


@pytest.mark.parametrize("slope", [1, sp.sqrt(3), 2])
def test_nonparallel_wedge_area(slope):
    r = integrate_quadratic_sector(1, sp.And(x * x + y * y <= 1, y >= 0, y <= slope * x), (x, y))
    assert abs((r.value - sp.atan(slope) / 2).evalf(45)) < sp.Float("1e-35")


def test_shear_and_translation_covariance():
    u = x - y - 1
    v = y - 2
    r = integrate_quadratic_sector(u * v, sp.And(u * u + v * v <= 4, u >= 0, v >= 0), (x, y))
    assert sp.simplify(r.value - 2) == 0


def test_sector_public_dispatch():
    r = integrate_result(x * y, region=sp.And(x * x + y * y <= 1, x >= 0, y >= 0), variables=(x, y))
    assert r.value == sp.Rational(1, 8) and r.complete


@pytest.mark.parametrize("extra", [x >= sp.Rational(1, 2), x * y >= 0, x * x - y * y >= 0])
def test_unproved_sector_constraints_decline(extra):
    assert (
        integrate_quadratic_sector(1, sp.And(x * x + y * y <= 1, x >= 0, y >= 0, extra), (x, y))
        is None
    )


def test_zero_angle_ambient_vs_intrinsic():
    region = sp.And(x * x + y * y <= 1, x >= 0, y >= 0, y <= 0)
    assert integrate_quadratic_sector(1, region, (x, y)).value == 0
    assert integrate_quadratic_sector(1, region, (x, y), measure_dimension="intrinsic") is None


@pytest.mark.parametrize("m", [2, 3, 4])
def test_logarithmic_tangent_order(m):
    g = y * (y - x**m)
    e = sp.Symbol("epsilon", positive=True)
    region = sp.And(x > 0, x < 1, y > 0, y < x**m)
    density = x ** (m - 1) / g
    result = regularized_result(
        density, region=region, variables=(x, y), regularization=HadamardFinitePart(hypersurface=g)
    )
    assert sp.simplify(result.value - sp.pi**2 / (6 * m)) == 0
    assert verify_regularization_result(result).valid
    cutoff, _ = tangent_graph_cutoff(density, region, g, e, (x, y))
    with mp.workdps(50):
        eps = mp.mpf("0.001")
        floor = mp.root(4 * eps, 2 * m)

        def inner(t):
            h = t**m
            d = mp.sqrt(max(mp.mpf(0), h * h - 4 * eps))
            low = (h - d) / 2
            high = (h + d) / 2
            return 2 / t * mp.log(low / high)

        numerical = mp.quad(inner, [floor, 1])
        assert abs(
            mp.mpf(str(cutoff.subs(e, sp.Rational(1, 1000)).evalf(45))) - numerical
        ) < mp.mpf("1e-35")


@pytest.mark.parametrize("m", [2, 3, 4])
@pytest.mark.parametrize("variables", [(x, y), (y, x)])
def test_polynomial_shear_translated_contact(m, variables):
    p = x**3 + x
    gap = (x - 1) ** m
    g = (y - p) * (y - p - gap)
    result = regularized_result(
        (x - 1) ** (m - 1) / g,
        region=sp.And(x > 1, x < 2, y > p, y < p + gap),
        variables=variables,
        regularization=HadamardFinitePart(hypersurface=g),
    )
    assert sp.simplify(result.value - sp.pi**2 / (6 * m)) == 0
    assert verify_regularization_result(result).valid
    d = result.provenance.details
    assert d["realization"] == "polynomial_contact_cutoff"
    mapping = dict(zip(d["source_variables"], d["forward_map"], strict=True))
    assert sp.simplify(g.subs(mapping, simultaneous=True) - d["transformed_level"]) == 0
    assert (
        tuple(sp.simplify(v.subs(mapping, simultaneous=True)) for v in d["inverse_map"])
        == d["target_variables"]
    )
    assert (
        sp.simplify(
            sp.Abs(sp.Matrix(d["forward_map"]).jacobian(d["target_variables"]).det())
            - d["chart_jacobian"]
        )
        == 0
    )


def test_crossing_parameter_graph_and_exceptional_fibers():
    region = sp.And(x > -1, x < 1, y > 0, y < x * x - a)
    r = integrate_partitioned_cell_graph(1, region, (x, y), assumptions=(a > 0, a < 1))
    assert (
        sp.simplify(
            r.value - (sp.Rational(2, 3) - 2 * a + sp.Rational(4, 3) * a ** sp.Rational(3, 2))
        )
        == 0
    )
    assert r.complete and r.absolute_convergent
    assert integrate_partitioned_cell_graph(1, region, (x, y)) is None
    family = integrate_cells(
        1,
        region=region,
        variables=(x, y),
        parameters=(a,),
        cells=[ParameterCell(sp.And(a > 0, a < 1))],
        assumptions=sp.And(a > 0, a < 1),
    )
    assert family.complete
    assert integrate_result(1, region=region.subs(a, 1), variables=(x, y)).value == 0
    assert integrate_result(1, region=region.subs(a, 0), variables=(x, y)).value == sp.Rational(
        2, 3
    )


@pytest.mark.parametrize(
    "expr,condition,strict,expected",
    [
        (1 - sp.sqrt(a), sp.And(a >= 0, a <= 1), False, True),
        (1 - sp.sqrt(a), sp.And(a >= 0, a <= 1), True, False),
        (1 - sp.sqrt(a), sp.And(a > 0, a < 1), True, True),
        (sp.sqrt(a) - 1, a >= 1, False, True),
        (sp.sqrt(a) - 1, a >= 1, True, False),
    ],
)
def test_radical_comparison_preserves_equality(expr, condition, strict, expected):
    assert nonnegative(expr, _normalize_assumptions(condition), strict=strict) == expected


def test_higher_graph_projection_tracks_endpoint_events():
    cells = linear_quadratic_cells(
        region=sp.And(x >= -1, x <= 1, y >= 0, y <= x**4 - a), variables=(x, y), parameters=(a,)
    )
    conditions = [c.condition for c in cells]
    assert any(c.subs(a, 0) == sp.true for c in conditions)
    assert any(c.subs(a, 1) == sp.true for c in conditions)
    assert len(cells) >= 5


def test_guide_examples_execute():
    path = Path(__file__).parents[1] / "docs/sectors-and-contact-charts.md"
    snippets = re.findall(r"```python\n(.*?)```", path.read_text(), flags=re.S)
    assert len(snippets) == 3
    for snippet in snippets:
        exec(compile(snippet, str(path), "exec"), {})


@pytest.mark.parametrize("scale", [2, -3])
def test_logarithmic_regulator_scale_changes_finite_part(scale):
    g = y * (y - x * x)
    r = regularized_result(
        x / g,
        region=sp.And(x > 0, x < 1, y > 0, y < x * x),
        variables=(x, y),
        regularization=HadamardFinitePart(hypersurface=scale * g),
    )
    assert sp.simplify(r.value - sp.pi**2 / 12 + sp.log(abs(scale)) ** 2 / 4) == 0
    assert verify_regularization_result(r).valid


def test_unknown_parameter_radius_does_not_receive_sector_proof():
    region = sp.And(x * x + y * y <= a, x >= 0, y >= 0)
    assert integrate_quadratic_sector(1, region, (x, y)) is None
    r = integrate_quadratic_sector(1, region, (x, y), assumptions=a > 0)
    assert sp.simplify(r.value - sp.pi * a / 4) == 0
