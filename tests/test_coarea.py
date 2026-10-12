"""Coarea identities, whole-chart certification and unsafe Fubini counterexamples."""

from dataclasses import replace

import pytest
import sympy as sp

from multiple_integrate import IntegrationOutcome, IntegrationPolicy, StrategyBudget
from multiple_integrate.coarea import CoareaChart, integrate_coarea, reduce_coarea
from multiple_integrate.results import UnevaluatedRegionIntegral, UnresolvedIntegration

x, y, z, t, u, v, a = sp.symbols("x y z t u v a", real=True)
box = ((y, 0, 1), (x, 0, 1))


@pytest.mark.parametrize(
    "level,density,expected",
    [
        (x + y, sp.exp(-x - y), (1 - sp.exp(-1)) ** 2),
        (2 * x - y + 3, x * y, sp.Rational(1, 4)),
        (-x - y, 1, 1),
        (x + 2 * y, x + y, 1),
        (3 * x + 2, x * y, sp.Rational(1, 4)),
    ],
)
def test_affine_boxes(level, density, expected):
    reduction = reduce_coarea(density, level_map=level, variables=(x, y), ranges=box)
    result = reduction.evaluate()
    assert sp.simplify(result.value - expected) == 0
    assert result.complete and result.absolute_convergent is True
    assert result.method == "coarea_affine"
    assert all(p.ranges[-1][0] == p.chart.coordinates[0] for p in reduction.pieces)
    assert all(c.complete for c in reduction.coverage)


def test_affine_nonunit_gradient_uses_fiber_metric_over_jacobian():
    reduction = reduce_coarea(1, level_map=3 * x + 4 * y, variables=(x, y), ranges=box)
    for piece in reduction.pieces:
        assert piece.jacobian == 5
        assert piece.fiber_metric == sp.Rational(5, 3)
        assert sp.simplify(piece.density - piece.fiber_metric / piece.jacobian) == 0
    assert reduction.evaluate().value == 1


def test_affine_symbolic_cell_and_permutation():
    result = integrate_coarea(
        x, level_map=a * x + y, variables=(y, x), ranges=box, assumptions=a > 1
    )
    assert result.value == sp.Rational(1, 2)
    assert result.absolute_convergent is True
    unresolved = integrate_coarea(x, level_map=a * x + y, variables=(x, y), ranges=box)
    assert unresolved.outcome is IntegrationOutcome.UNKNOWN


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_radial_squared_gaussians(dimension):
    coords = (x, y, z, v)[:dimension]
    q = sum(c * c for c in coords)
    reduction = reduce_coarea(sp.exp(-q), level_map=q, variables=coords, level_bounds=(0, sp.oo))
    result = reduction.evaluate()
    assert sp.simplify(result.value - sp.pi ** sp.Rational(dimension, 2)) == 0
    assert result.complete and result.absolute_convergent is True
    assert "center" in str(reduction.coverage[0].details)


@pytest.mark.parametrize(
    "level,density,bounds,expected",
    [
        (sp.sqrt(x * x + y * y), 1, (0, 2), 4 * sp.pi),
        (x * x + y * y, 1, (1, 4), 3 * sp.pi),
        ((x - 2) ** 2 + (y + 3) ** 2, ((x - 2) ** 2 + (y + 3) ** 2), (0, 2), 2 * sp.pi),
        (x * x + y * y, sp.log(x * x + y * y), (0, 1), -sp.pi),
        (x * x + y * y, 1 / (x * x + y * y), (1, 4), sp.pi * sp.log(4)),
    ],
)
def test_radial_shells_centers_and_integrable_singularities(level, density, bounds, expected):
    result = integrate_coarea(density, level_map=level, variables=(x, y), level_bounds=bounds)
    assert sp.simplify(result.value - expected) == 0
    assert result.absolute_convergent is True


def test_radial_positive_divergence_and_critical_map_rejection():
    result = integrate_coarea(
        1 / (x * x + y * y), level_map=x * x + y * y, variables=(x, y), level_bounds=(0, 1)
    )
    assert result.outcome is IntegrationOutcome.DIVERGENT and result.value == sp.oo
    assert (
        integrate_coarea(1, level_map=0, variables=(x, y), ranges=box).outcome
        is IntegrationOutcome.UNKNOWN
    )


@pytest.mark.parametrize("density", [sp.exp(x)])
def test_nonradial_densities_decline(density):
    result = integrate_coarea(
        density, level_map=x * x + y * y, variables=(x, y), level_bounds=(0, 1)
    )
    assert result.outcome is IntegrationOutcome.UNKNOWN


def test_supplied_polynomial_shear_chart():
    chart = CoareaChart((t, u), (t + u * u, u), (x - y * y, y), ((u, 0, 1), (t, 0, 1)))
    region = sp.And(y >= 0, y <= 1, x - y * y >= 0, x - y * y <= 1)
    reduction = reduce_coarea(
        x, level_map=x - y * y, variables=(x, y), region=region, charts=(chart,)
    )
    assert reduction.evaluate().value == sp.Rational(5, 6)
    assert sp.simplify(reduction.pieces[0].jacobian - reduction.pieces[0].fiber_metric) == 0


def test_supplied_rational_chart_checks_denominator_rank_and_metric():
    chart = CoareaChart((t, u), (t * u, u), (x / y, y), ((u, 1, 2), (t, 0, 1)))
    region = sp.And(y >= 1, y <= 2, x >= 0, x <= y)
    reduction = reduce_coarea(1, level_map=x / y, variables=(x, y), region=region, charts=(chart,))
    assert reduction.evaluate().value == sp.Rational(3, 2)
    assert sp.simplify(reduction.pieces[0].density - u) == 0
    bad = replace(chart, ranges=((u, 0, 2), (t, 0, 1)))
    result = integrate_coarea(
        1,
        level_map=x / y,
        variables=(x, y),
        region=sp.And(y >= 0, y <= 2, x >= 0, x <= y),
        charts=(bad,),
    )
    assert result.outcome is IntegrationOutcome.UNKNOWN


@pytest.mark.parametrize("error", ["inverse", "map", "hole", "overlap", "extra"])
def test_unverified_chart_claims_cannot_certify_domain(error):
    chart = CoareaChart((t, u), (t, u), (x, y), ((u, 0, 1), (t, 0, 1)))
    region = sp.And(x >= 0, x <= 1, y >= 0, y <= 1)
    charts = (chart,)
    level = x
    if error == "inverse":
        charts = (replace(chart, inverse=(x + 1, y)),)
    if error == "map":
        level = x + y
    if error == "hole":
        charts = (replace(chart, ranges=((u, 0, sp.Rational(1, 2)), (t, 0, 1))),)
    if error == "overlap":
        charts = (chart, chart)
    if error == "extra":
        region = sp.And(x >= 0, x <= sp.Rational(1, 2), y >= 0, y <= 1)
    assert (
        integrate_coarea(1, level_map=level, variables=(x, y), region=region, charts=charts).outcome
        is IntegrationOutcome.UNKNOWN
    )


def test_disjoint_charts_share_only_null_boundaries():
    region = sp.And(x >= 0, x <= 2, y >= 0, y <= 1)
    charts = tuple(CoareaChart((t, u), (t, u), (x, y), ((u, 0, 1), (t, k, k + 1))) for k in (0, 1))
    result = integrate_coarea(x, level_map=x, variables=(x, y), region=region, charts=charts)
    assert result.value == 2 and result.complete


def test_supplied_three_dimensional_chart():
    chart = CoareaChart(
        (t, u, v), (t + u * v, u, v), (x - y * z, y, z), ((v, 0, 1), (u, 0, 1), (t, 0, 1))
    )
    region = sp.And(y >= 0, y <= 1, z >= 0, z <= 1, x - y * z >= 0, x - y * z <= 1)
    result = integrate_coarea(
        x, level_map=x - y * z, variables=(x, y, z), region=region, charts=(chart,)
    )
    assert result.value == sp.Rational(3, 4) and result.absolute_convergent is True


def test_conditional_iterated_integral_is_not_promoted_to_coarea_value():
    density = (x * x - y * y) / (x * x + y * y) ** 2
    result = integrate_coarea(density, level_map=x, variables=(x, y), ranges=box)
    assert result.outcome is IntegrationOutcome.UNKNOWN and result.complete is False
    assert result.coverage[0].complete is True
    assert isinstance(result.value, UnevaluatedRegionIntegral)
    assert result.absolute_convergent is None
    assert result.diagnostics["fiber_results"][0].absolute_convergent is not True


def test_chart_budget_and_malformed_coordinate_order():
    policy = IntegrationPolicy(semialg=StrategyBudget(max_pieces=1))
    assert (
        integrate_coarea(1, level_map=x + y, variables=(x, y), ranges=box, policy=policy).outcome
        is IntegrationOutcome.UNKNOWN
    )
    chart = CoareaChart((t, u), (t, u), (x, y), ((t, 0, 1), (u, 0, 1)))
    with pytest.raises(ValueError, match="inner-first"):
        reduce_coarea(1, level_map=x, variables=(x, y), region=True, charts=(chart,))
    with pytest.raises(UnresolvedIntegration):
        reduce_coarea(1, level_map=a * x, variables=(x, y), ranges=box)


def test_chart_coordinates_cannot_capture_parameters():
    chart = CoareaChart((a, u), (a, u), (x, y), ((u, 0, 1), (a, 0, 1)))
    with pytest.raises(ValueError, match="capture"):
        reduce_coarea(
            a * x,
            level_map=x,
            variables=(x, y),
            region=sp.And(x >= 0, x <= 1, y >= 0, y <= 1),
            charts=(chart,),
        )


def test_proved_existence_survives_an_unknown_child_value(monkeypatch):
    from multiple_integrate import ConvergenceEvidence
    from multiple_integrate.results import IntegrationResult, IntegrationStatus

    monkeypatch.setattr(
        "multiple_integrate.api.integration.integrate_result",
        lambda f, *ranges, **kw: IntegrationResult(
            sp.Integral(f, *ranges),
            IntegrationStatus.UNKNOWN,
            "test",
            complete=False,
            convergence_evidence=(ConvergenceEvidence("absolute", True, "test"),),
        ),
    )
    result = integrate_coarea(1, level_map=x, variables=(x, y), ranges=box)
    assert result.outcome is IntegrationOutcome.UNKNOWN and result.complete is False
    assert result.absolute_convergent is True and result.ordinary_convergent is True


def test_external_pole_and_chart_density_not_misclassified_by_compact_enclosure():
    # The chart is regular, but the integrand has an actual interior pole.
    result = integrate_coarea(
        1 / (x - 2), level_map=x, variables=(x, y), ranges=((y, 0, 1), (x, 1, 3))
    )
    assert result.outcome is not IntegrationOutcome.FINITE
    assert result.absolute_convergent is not True


def test_geometry_degree_budget_precedes_native_equivalence():
    chart = CoareaChart((t, u), (t, u), (x, y), ((u, 0, t * t + 1), (t, 0, 1)))
    region = sp.And(x >= 0, x <= 1, y >= 0, y <= x * x + 1)
    policy = IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=1))
    result = integrate_coarea(
        1, level_map=x, variables=(x, y), region=region, charts=(chart,), policy=policy
    )
    assert result.outcome is IntegrationOutcome.UNKNOWN and "degree" in result.limitations[0]
    result = integrate_coarea(
        1, level_map=x * x + y * y, variables=(x, y), level_bounds=(0, 1), policy=policy
    )
    assert result.outcome is IntegrationOutcome.UNKNOWN


def test_coordinate_assumptions_cannot_hide_missing_source_coverage():
    with pytest.raises(ValueError, match="parameters"):
        reduce_coarea(1, level_map=x, variables=(x, y), ranges=box, assumptions=x > 0)


def test_negative_radial_divergence_preserves_sign():
    result = integrate_coarea(
        -1 / (x * x + y * y), level_map=x * x + y * y, variables=(x, y), level_bounds=(0, 1)
    )
    assert result.outcome is IntegrationOutcome.DIVERGENT and result.value == -sp.oo


def test_Boole_radial_fiber_has_finite_Tonelli_evidence():
    from multiple_integrate import Boole

    q = x * x + y * y
    result = integrate_coarea(Boole(q <= 1), level_map=q, variables=(x, y), level_bounds=(0, sp.oo))
    assert result.value == sp.pi and result.absolute_convergent is True


def test_coarea_documentation_examples_execute():
    import re
    from pathlib import Path

    text = (Path(__file__).parents[1] / "docs/coarea.md").read_text()
    namespace = {}
    for code in re.findall(r"```python\n(.*?)```", text, re.S):
        exec(compile(code, "coarea.md", "exec"), namespace)
