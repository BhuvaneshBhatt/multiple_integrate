"""Independent coarea identities, exceptional fibers and deterministic dispatch limits."""

from dataclasses import replace

import pytest
import sympy as sp

from multiple_integrate import IntegrationOutcome, IntegrationPolicy, StrategyBudget
from multiple_integrate.adapters import semialg as geometry_adapter
from multiple_integrate.coarea import (
    CoareaChart,
    clear_coarea_cache,
    integrate_coarea,
    reduce_coarea,
)

x, y, z, t, s, u, a = sp.symbols("x y z t s u a", real=True)


def test_vector_affine_projections_use_gram_jacobian():
    r = reduce_coarea(
        x + y + z,
        level_map=(2 * x, -3 * y),
        variables=(x, y, z),
        ranges=((z, 0, 1), (y, 0, 1), (x, 0, 1)),
    )
    p = r.pieces[0]
    assert p.jacobian == 6 and p.fiber_metric == 1
    assert r.evaluate().value == sp.Rational(3, 2)
    assert r.evaluate().absolute_convergent


def test_coupled_vector_chart_metric_and_source_coverage():
    chart = CoareaChart((t, s, u), (t + s, s, u), (x - y, y, z), ((u, 0, 1), (s, 0, 1), (t, 0, 1)))
    region = sp.And(x - y >= 0, x - y <= 1, y >= 0, y <= 1, z >= 0, z <= 1)
    r = reduce_coarea(x, level_map=(x - y, y), variables=(x, y, z), region=region, charts=(chart,))
    assert r.pieces[0].jacobian == 1 and r.evaluate().value == 1
    wrong = replace(chart, inverse=(x, y, z))
    assert (
        integrate_coarea(
            x, level_map=(x - y, y), variables=(x, y, z), region=region, charts=(wrong,)
        ).outcome
        is IntegrationOutcome.UNKNOWN
    )


def test_vector_rank_loss_cannot_certify_chart():
    chart = CoareaChart((t, s, u), (t, s, u), (x, y, z), ((u, 0, 1), (s, 0, 1), (t, 0, 1)))
    region = sp.And(x >= 0, x <= 1, y >= 0, y <= 1, z >= 0, z <= 1)
    r = integrate_coarea(1, level_map=(x, x), variables=(x, y, z), region=region, charts=(chart,))
    assert r.outcome is IntegrationOutcome.UNKNOWN


def test_geometry_certificate_reuse_does_not_replay_proofs(monkeypatch):
    from semialg import chart_certificates as gc

    clear_coarea_cache()
    chart = CoareaChart((t, u), (t, u), (x, y), ((u, 0, 1), (t, 0, 1)))
    calls = 0
    original = gc._prove

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gc, "_prove", counted)
    first = geometry_adapter.chart_geometry(chart, x, (x, y))
    cold = calls
    assert cold > 0
    for _ in range(10):
        assert geometry_adapter.chart_geometry(chart, x, (x, y)) is first
    assert calls == cold
    gc.clear_chart_cache()
    geometry_adapter.chart_geometry(chart, x, (x, y))
    assert calls > cold


def test_budget_rejects_before_inverse_and_rank_work(monkeypatch):
    from semialg import chart_certificates as gc

    clear_coarea_cache()

    def forbidden(*args, **kwargs):
        pytest.fail("over-budget geometry reached a proof")

    monkeypatch.setattr(gc, "_identity", forbidden)
    chart = CoareaChart((t, u), (t + u**9, u), (x - y**9, y), ((u, 0, 1), (t, 0, 1)))
    policy = IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=2))
    r = integrate_coarea(
        1, level_map=x - y**9, variables=(x, y), region=sp.true, charts=(chart,), policy=policy
    )
    assert r.outcome is IntegrationOutcome.UNKNOWN


@pytest.mark.parametrize(
    "density,expected",
    [
        (x * x, sp.pi / 4),
        (1 / (1 + x * x + y * y) ** 2, sp.pi / 2),
        (sp.Piecewise((1, x * x + y * y < sp.Rational(1, 4)), (0, True)), sp.pi / 4),
    ],
)
def test_compact_radial_profiles_dispatch_once(monkeypatch, density, expected):
    from multiple_integrate import coarea
    from multiple_integrate.region_types import DiskRegion

    clear_coarea_cache()
    calls = 0
    original = coarea.integrate_coarea

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(coarea, "integrate_coarea", counted)
    region = DiskRegion(((y, -1, 1), (x, -1, 1)), radius=1, center=(0, 0))
    answer = coarea.try_automatic_coarea(density, region, (), (), IntegrationPolicy())
    assert answer is not None and sp.simplify(answer[1] - expected) == 0
    assert calls == 1


def test_unsuccessful_dispatch_is_cached_and_does_not_evaluate(monkeypatch):
    from multiple_integrate import coarea
    from multiple_integrate.region_types import DiskRegion

    clear_coarea_cache()

    def forbidden(*args, **kwargs):
        pytest.fail("unsuitable profile reached child integration")

    monkeypatch.setattr(coarea, "integrate_coarea", forbidden)
    region = DiskRegion(((y, -1, 1), (x, -1, 1)), radius=1, center=(0, 0))
    for _ in range(4):
        assert (
            coarea.try_automatic_coarea(sp.sin(x * y), region, (), (), IntegrationPolicy()) is None
        )
    info = coarea._automatic_geometry.cache_info()
    assert info.misses == 1 and info.hits == 3


def test_operation_budget_precedes_profile_recognition(monkeypatch):
    from multiple_integrate import coarea

    clear_coarea_cache()

    def forbidden(*args, **kwargs):
        pytest.fail("operation budget reached recognizer")

    monkeypatch.setattr(coarea, "_cheap_radial_profile", forbidden)
    expr = sp.Add(*(sp.sin(k * x + y) for k in range(100)), evaluate=False)
    assert (
        coarea._automatic_geometry(expr, (x, y), (), ((0, 0), 0, 1), (), IntegrationPolicy())
        is None
    )


def test_shifted_gaussian_automatic_center():
    from multiple_integrate import coarea
    from multiple_integrate.region_types import BoxRegion

    region = BoxRegion(((y, -sp.oo, sp.oo), (x, -sp.oo, sp.oo)))
    answer = coarea.try_automatic_coarea(
        sp.exp(-2 * ((x - 1) ** 2 + (y + 2) ** 2)),
        region,
        ((y, -sp.oo, sp.oo), (x, -sp.oo, sp.oo)),
        (),
        IntegrationPolicy(),
    )
    assert answer is not None and answer[1] == sp.pi / 2


@pytest.mark.parametrize(
    "density,expected",
    [
        (sp.cos(sp.atan2(y, x)) ** 2, sp.pi / 2),
        (sp.cos(3 * sp.atan2(y, x)), 0),
        (sp.sin(2 * sp.atan2(y, x)) ** 2, sp.pi / 2),
        (1 / (3 + x / sp.sqrt(x * x + y * y)), sp.pi / sp.sqrt(8)),
    ],
)
def test_fourier_and_bounded_reciprocal_angular_moments(density, expected):
    r = integrate_coarea(density, level_map=x * x + y * y, variables=(x, y), level_bounds=(0, 1))
    assert r.outcome is IntegrationOutcome.FINITE
    assert sp.simplify(r.value - expected) == 0 and r.absolute_convergent


def test_angular_denominator_and_origin_prerequisites():
    angle = sp.atan2(y, x)
    q = x * x + y * y
    for f in [1 / (1 + x / sp.sqrt(q)), sp.cos(angle) / q, sp.cos(sp.Rational(1, 2) * angle)]:
        r = integrate_coarea(f, level_map=q, variables=(x, y), level_bounds=(0, 1))
        assert r.outcome is IntegrationOutcome.UNKNOWN


def test_separable_endpoint_order_evidence_on_null_excluded_chart():
    from multiple_integrate.coarea import _endpoint_product_absolute

    chart = CoareaChart(
        (t, u),
        (t * u, u),
        (x / y, y),
        ((u, 0, 1), (t, 0, 1)),
        chart_exclusions=sp.Eq(u, 0),
        source_exclusions=sp.Eq(y, 0),
    )
    region = sp.And(x >= 0, x <= y, y >= 0, y <= 1)
    r = reduce_coarea(
        y ** sp.Rational(-3, 2), level_map=x / y, variables=(x, y), region=region, charts=(chart,)
    )
    assert _endpoint_product_absolute(r.pieces[0], (), r.policy)
    answer = r.evaluate()
    assert answer.value == 2 and answer.absolute_convergent
    broken = integrate_coarea(
        y**-2, level_map=x / y, variables=(x, y), region=region, charts=(chart,)
    )
    assert broken.outcome is IntegrationOutcome.DIVERGENT


def test_caller_coarea_cells_evaluate_zero_rank_fiber_as_source_integral():
    from multiple_integrate.coarea import integrate_coarea_cells
    from multiple_integrate.parameter_cells import ParameterCell

    cells = (ParameterCell(a < 0), ParameterCell(sp.Eq(a, 0)), ParameterCell(a > 0))
    r = integrate_coarea_cells(
        x + y,
        cells=cells,
        parameters=(a,),
        level_map=a * x,
        variables=(x, y),
        ranges=((y, 0, 1), (x, 0, 1)),
    )
    assert r.complete and all(c.value == 1 for c in r.cases)
    assert r.diagnostics["geometry_events"] == (a,)
    assert len(r.diagnostics["cell_results"]) == 3
    assert r.diagnostics["cell_results"][1].method != "coarea_affine"


def test_coarea_cell_uncovered_values_and_overlap_are_rejected():
    from multiple_integrate.coarea import integrate_coarea_cells
    from multiple_integrate.parameter_cells import ParameterCell

    request = dict(
        parameters=(a,), level_map=a * x, variables=(x, y), ranges=((y, 0, 1), (x, 0, 1))
    )
    r = integrate_coarea_cells(1, cells=(ParameterCell(a > 0),), **request)
    assert not r.complete and r.cases[-1].status.value == "unknown"
    with pytest.raises(ValueError, match="overlap"):
        integrate_coarea_cells(1, cells=(ParameterCell(a >= 0), ParameterCell(a <= 0)), **request)


def test_coarea_cell_collapsed_shell_and_coefficient_pole():
    from multiple_integrate.coarea import integrate_coarea_cells
    from multiple_integrate.parameter_cells import ParameterCell

    r = integrate_coarea_cells(
        1,
        cells=(ParameterCell(sp.Eq(a, 0)), ParameterCell(a > 0)),
        parameters=(a,),
        assumptions=a >= 0,
        level_map=x * x + y * y,
        variables=(x, y),
        level_bounds=(0, a),
    )
    assert r.complete and r.cases[0].value == 0 and sp.simplify(r.cases[1].value - sp.pi * a) == 0
    assert r.diagnostics["geometry_events"] == (a,)
    r = integrate_coarea_cells(
        1 / a,
        cells=(ParameterCell(sp.Eq(a, 0)),),
        parameters=(a,),
        level_map=x,
        variables=(x, y),
        ranges=((y, 0, 1), (x, 0, 1)),
    )
    assert r.diagnostics["cell_results"][0].outcome is IntegrationOutcome.UNKNOWN


def test_parameter_vector_rank_drop_keeps_nonzero_source_integral():
    from multiple_integrate.coarea import integrate_coarea_cells
    from multiple_integrate.parameter_cells import ParameterCell

    cells = (ParameterCell(a > 0), ParameterCell(sp.Eq(a, 0)))
    r = integrate_coarea_cells(
        1,
        cells=cells,
        parameters=(a,),
        assumptions=a >= 0,
        level_map=(a * x, y),
        variables=(x, y, z),
        ranges=((z, 0, 1), (y, 0, 1), (x, 0, 1)),
    )
    assert r.complete and all(c.value == 1 for c in r.cases)
    assert r.diagnostics["cell_results"][1].value == 1


def test_parameter_chart_collapse_has_ambient_zero_and_event_evidence():
    from multiple_integrate.coarea import integrate_coarea_cells
    from multiple_integrate.parameter_cells import ParameterCell

    chart = CoareaChart((t, u), (t, u), (x, y), ((u, 0, 1), (t, 0, a)))
    region = sp.And(x >= 0, x <= a, y >= 0, y <= 1)
    r = integrate_coarea_cells(
        1,
        cells=(ParameterCell(a > 0), ParameterCell(sp.Eq(a, 0))),
        parameters=(a,),
        assumptions=a >= 0,
        level_map=x,
        variables=(x, y),
        region=region,
        charts=(chart,),
    )
    assert r.complete and r.cases[0].value == a and r.cases[1].value == 0
    assert r.diagnostics["geometry_events"] == (a,)


def test_angular_negative_denominator_and_gaussian_profile():
    q = x * x + y * y
    r = integrate_coarea(
        sp.exp(-q) / (3 + x / sp.sqrt(q)), level_map=q, variables=(x, y), level_bounds=(0, sp.oo)
    )
    assert sp.simplify(r.value - sp.pi / sp.sqrt(8)) == 0 and r.absolute_convergent
    r = integrate_coarea(
        1 / (-3 + x / sp.sqrt(q)), level_map=q, variables=(x, y), level_bounds=(0, 1)
    )
    assert sp.simplify(r.value + sp.pi / sp.sqrt(8)) == 0 and r.absolute_convergent


def test_independent_numeric_angular_oracle_precision_escalation():
    import mpmath as mp

    q = x * x + y * y
    r = integrate_coarea(
        x * x / (3 + x / sp.sqrt(q)), level_map=q, variables=(x, y), level_bounds=(0, 1)
    )
    for precision in (35, 65):
        with mp.workdps(precision):
            oracle = (
                mp.quad(lambda angle: mp.cos(angle) ** 2 / (3 + mp.cos(angle)), [-mp.pi, 0, mp.pi])
                / 4
            )
            actual = mp.mpf(str(sp.N(r.value, precision)))
            assert abs(oracle - actual) < mp.mpf(10) ** (-precision + 5)
    assert r.absolute_convergent


def test_coarea_cells_none_region_and_malformed_vector_dimensions():
    from multiple_integrate.coarea import integrate_coarea_cells
    from multiple_integrate.parameter_cells import ParameterCell

    r = integrate_coarea_cells(
        1,
        cells=(ParameterCell(sp.true),),
        parameters=(a,),
        level_map=x,
        variables=(x, y),
        ranges=((y, 0, 1), (x, 0, 1)),
        region=None,
    )
    assert r.complete and r.cases[0].value == 1
    with pytest.raises(ValueError, match="Level dimension"):
        integrate_coarea_cells(
            1,
            cells=(ParameterCell(sp.true),),
            parameters=(a,),
            level_map=(x, y),
            variables=(x, y),
            ranges=((y, 0, 1), (x, 0, 1)),
        )

    with pytest.raises(ValueError, match="source ranges"):
        integrate_coarea_cells(
            1, cells=(ParameterCell(sp.true),), parameters=(a,), level_map=0, variables=(x, y)
        )


def test_coincident_affine_corner_event_is_evaluated_on_its_equality_cell():
    from multiple_integrate.coarea import integrate_coarea_cells
    from multiple_integrate.parameter_cells import ParameterCell

    r = integrate_coarea_cells(
        1,
        cells=(
            ParameterCell(sp.And(a > 0, a < 1)),
            ParameterCell(sp.Eq(a, 1)),
            ParameterCell(a > 1),
        ),
        parameters=(a,),
        assumptions=a > 0,
        level_map=a * x + y,
        variables=(x, y),
        ranges=((y, 0, 1), (x, 0, 1)),
    )
    assert r.complete and all(c.value == 1 for c in r.cases)
    assert a - 1 in r.diagnostics["geometry_events"]
    assert len(r.diagnostics["cell_results"][1].diagnostics["coarea_reduction"].pieces) == 2


def test_radial_geometry_degree_budget_precedes_child_dispatch(monkeypatch):
    from multiple_integrate import coarea
    from multiple_integrate.region_types import DiskRegion

    clear_coarea_cache()

    def forbidden(*args, **kwargs):
        pytest.fail("over-budget radial map dispatched")

    monkeypatch.setattr(coarea, "integrate_coarea", forbidden)
    policy = IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=1))
    region = DiskRegion(((y, -1, 1), (x, -1, 1)), radius=1, center=(0, 0))
    assert coarea.try_automatic_coarea(1, region, (), (), policy) is None


def test_sign_flag_on_radial_coordinate_cannot_hide_nonintegrable_cancellation():
    positive_x = sp.Symbol("positive_x", positive=True)
    q = positive_x**2 + y * y
    result = integrate_coarea(
        positive_x / q**2, level_map=q, variables=(positive_x, y), level_bounds=(0, 1)
    )
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert "plain real" in result.limitations[0]


def test_chart_coordinate_flags_cannot_turn_unbounded_source_into_finite_zero():
    positive_y = sp.Symbol("positive_y", positive=True)
    positive_u = sp.Symbol("positive_u", positive=True)
    chart = CoareaChart(
        (t, positive_u),
        (t * positive_u, positive_u),
        (x / positive_y, positive_y),
        ((positive_u, -1, 1), (t, 0, 1)),
    )
    source = sp.And(positive_y <= 1, x / positive_y >= 0, x / positive_y <= 1)
    result = integrate_coarea(
        1, level_map=x / positive_y, variables=(x, positive_y), region=source, charts=(chart,)
    )
    assert result.outcome is IntegrationOutcome.UNKNOWN


def test_separate_coordinate_rational_density_keeps_faster_iterated_route(monkeypatch):
    from multiple_integrate import coarea
    from multiple_integrate.region_types import BoxRegion

    clear_coarea_cache()

    def forbidden(*args, **kwargs):
        pytest.fail("separate amplitude took speculative affine reduction")

    monkeypatch.setattr(coarea, "integrate_coarea", forbidden)
    ranges = ((x, 0, 3), (y, 1, 2))
    region = BoxRegion(ranges)
    assert (
        coarea.try_automatic_coarea(x * x * y / (x + y), region, ranges, (), IntegrationPolicy())
        is None
    )


def test_compact_true_affine_rational_profile_still_dispatches():
    from multiple_integrate import coarea
    from multiple_integrate.region_types import BoxRegion

    ranges = ((x, 0, 1), (y, 0, 1))
    region = BoxRegion(ranges)
    answer = coarea.try_automatic_coarea(1 / (1 + x + y), region, ranges, (), IntegrationPolicy())
    assert answer is not None and sp.simplify(answer[1] - (3 * sp.log(3) - 4 * sp.log(2))) == 0
