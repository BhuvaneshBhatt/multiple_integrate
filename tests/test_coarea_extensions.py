"""Exceptional fibers, chart branches, angular L1 prerequisites and dispatch budgets."""

from dataclasses import replace

import pytest
import sympy as sp

from multiple_integrate import (
    IntegrationOutcome,
    IntegrationPolicy,
    StrategyBudget,
    clear_cache,
    integrate,
    integrate_result,
)
from multiple_integrate.coarea import (
    CoareaChart,
    _automatic_geometry,
    integrate_coarea,
    reduce_coarea,
)

x, y, t, u, a = sp.symbols("x y t u a", real=True)
box = ((y, 0, 1), (x, 0, 1))
q = x * x + y * y


@pytest.mark.parametrize("rate", [-2, 0, 2])
def test_affine_coefficient_equality_and_orientation(rate):
    r = integrate_coarea(x + y, level_map=rate * x + y, variables=(x, y), ranges=box)
    assert r.value == 1 and r.absolute_convergent
    assert (
        integrate_coarea(1, level_map=0 * x + 0 * y, variables=(x, y), ranges=box).outcome
        is IntegrationOutcome.UNKNOWN
    )


@pytest.mark.parametrize("condition", [a > 0, a < 0])
def test_chart_rank_and_orientation_in_parameter_cells(condition):
    chart = CoareaChart((t, u), (a * t, u), (x / a, y), ((u, 0, 1), (t, 0, 1)))
    region = (
        sp.And(y >= 0, y <= 1, x >= 0, x <= a)
        if condition == (a > 0)
        else sp.And(y >= 0, y <= 1, x >= a, x <= 0)
    )
    r = integrate_coarea(
        1, level_map=x / a, variables=(x, y), region=region, charts=(chart,), assumptions=condition
    )
    assert sp.simplify(r.value - (a if condition == (a > 0) else -a)) == 0
    lost = replace(chart, forward=(0 * t, u), inverse=(x, y))
    assert (
        integrate_coarea(1, level_map=x, variables=(x, y), region=region, charts=(lost,)).outcome
        is IntegrationOutcome.UNKNOWN
    )


@pytest.mark.parametrize("bounds", [(0, 0), (1, 1)])
def test_collapsed_radial_levels_are_ambient_null(bounds):
    r = integrate_coarea(1, level_map=q, variables=(x, y), level_bounds=bounds)
    assert r.value == 0 and r.absolute_convergent and r.complete
    assert r.diagnostics["coarea_reduction"].pieces[0].geometry_certificate is not None


@pytest.mark.parametrize("rate", [1, -1])
def test_coincident_affine_corner_levels(rate):
    r = reduce_coarea(1, level_map=x + rate * y, variables=(x, y), ranges=box)
    assert len(r.pieces) == 2 and r.evaluate().value == 1


@pytest.mark.parametrize(
    "density,expected",
    [
        (x, 0),
        (x * y, 0),
        (x * x, sp.pi / 4),
        (x * x * y * y, sp.pi / 24),
        ((x + 1) ** 2, 5 * sp.pi / 4),
    ],
)
def test_polynomial_angular_unit_disk_moments(density, expected):
    r = integrate_coarea(density, level_map=q, variables=(x, y), level_bounds=(0, 1))
    assert sp.simplify(r.value - expected) == 0 and r.absolute_convergent


@pytest.mark.parametrize(
    "density,expected",
    [(x * x * sp.exp(-q), sp.pi / 2), (x**4 * sp.exp(-q), 3 * sp.pi / 4), (x * y * sp.exp(-q), 0)],
)
def test_polynomial_angular_gaussian_moments(density, expected):
    r = integrate_coarea(density, level_map=q, variables=(x, y), level_bounds=(0, sp.oo))
    assert r.value == expected and r.absolute_convergent


def test_angular_zero_cannot_hide_nonintegrable_origin_or_tail():
    for bounds in [(0, 1), (1, sp.oo)]:
        r = integrate_coarea(
            x / q**2 if bounds == (0, 1) else x / q,
            level_map=q,
            variables=(x, y),
            level_bounds=bounds,
        )
        assert r.outcome is IntegrationOutcome.UNKNOWN and r.absolute_convergent is not True
        assert r.diagnostics["absolute_majorant_results"][0].outcome is IntegrationOutcome.DIVERGENT


def test_rational_chart_regular_away_from_null_boundary():
    chart = CoareaChart(
        (t, u),
        (t * u, u),
        (x / y, y),
        ((u, 0, 1), (t, 0, 1)),
        chart_exclusions=sp.Eq(u, 0),
        source_exclusions=sp.Eq(y, 0),
    )
    region = sp.And(y >= 0, y <= 1, x >= 0, x <= y)
    r = integrate_coarea(1, level_map=x / y, variables=(x, y), region=region, charts=(chart,))
    assert r.value == sp.Rational(1, 2) and r.absolute_convergent
    # Removing a positive-volume source part must fail even with a claimed exclusion.
    bad = replace(chart, source_exclusions=y < sp.Rational(1, 2))
    assert (
        integrate_coarea(1, level_map=x / y, variables=(x, y), region=region, charts=(bad,)).outcome
        is IntegrationOutcome.UNKNOWN
    )


def test_null_boundary_does_not_certify_a_singular_density():
    chart = CoareaChart(
        (t, u),
        (t * u, u),
        (x / y, y),
        ((u, 0, 1), (t, 0, 1)),
        chart_exclusions=sp.Eq(u, 0),
        source_exclusions=sp.Eq(y, 0),
    )
    region = sp.And(y >= 0, y <= 1, x >= 0, x <= y)
    r = integrate_coarea(
        1 / y**2, level_map=x / y, variables=(x, y), region=region, charts=(chart,)
    )
    assert r.outcome is IntegrationOutcome.DIVERGENT and r.value == sp.oo


def test_positive_algebraic_chart_branch():
    chart = CoareaChart(
        (t, u),
        (t, sp.sqrt(u)),
        (x, y * y),
        ((u, 0, 1), (t, 0, 1)),
        chart_exclusions=sp.Eq(u, 0),
        source_exclusions=sp.Eq(y, 0),
        source_domain=y >= 0,
        kind="algebraic",
    )
    r = integrate_coarea(
        1,
        level_map=x,
        variables=(x, y),
        region=sp.And(x >= 0, x <= 1, y >= 0, y <= 1),
        charts=(chart,),
    )
    assert r.value == 1 and r.absolute_convergent
    bad = replace(chart, source_domain=sp.true)
    assert (
        integrate_coarea(
            1,
            level_map=x,
            variables=(x, y),
            region=sp.And(x >= 0, x <= 1, y >= -1, y <= 1),
            charts=(bad,),
        ).outcome
        is IntegrationOutcome.UNKNOWN
    )


@pytest.mark.parametrize("squared", [False, True])
def test_canonical_supplied_polar_chart(squared):
    from semialg.chart_certificates import polar_chart

    chart = polar_chart((x, y), (t, u), (0, 1), squared=squared)
    candidate = CoareaChart(**vars(chart))
    r = integrate_coarea(
        x * x,
        level_map=q if squared else sp.sqrt(q),
        variables=(x, y),
        region=q <= 1,
        charts=(candidate,),
    )
    assert sp.simplify(r.value - sp.pi / 4) == 0 and r.absolute_convergent
    bad = replace(candidate, forward=(t * sp.cos(u), t * sp.cos(u)))
    assert (
        integrate_coarea(
            1,
            level_map=q if squared else sp.sqrt(q),
            variables=(x, y),
            region=q <= 1,
            charts=(bad,),
        ).outcome
        is IntegrationOutcome.UNKNOWN
    )


def test_affine_automatic_dispatch_precedes_general_fallback(monkeypatch):
    import multiple_integrate.core as core
    from multiple_integrate.planner import fresh_strategy_trace, strategy_trace

    clear_cache()
    original = core._iterated_integrate

    def guarded(expr, ranges, opts):
        if len(ranges) > 1 and any(r[0] in (x, y) for r in ranges):
            pytest.fail("multidimensional slow fallback")
        return original(expr, ranges, opts)

    monkeypatch.setattr(core, "_iterated_integrate", guarded)
    # The outer one-dimensional symbolic integral is permitted; the original multidimensional fallback is forbidden.
    f = sp.exp(-((x + y) ** 3))
    with fresh_strategy_trace():
        value = integrate(f, *box)
        assert any(s.method == "coarea_affine" and s.succeeded for s in strategy_trace())
    assert not value.has(sp.Integral)


def test_automatic_recognition_cache_and_budget():
    clear_cache()
    f = sp.exp(-((x + y) ** 3))
    r = integrate_result(f, *box)
    assert r.absolute_convergent and any(
        c.scope == "coarea_fibers" and c.complete for c in r.coverage
    )
    before = _automatic_geometry.cache_info()
    integrate(f, *box)
    assert _automatic_geometry.cache_info() == before
    low = IntegrationPolicy(semialg=StrategyBudget(max_variables=1))
    from multiple_integrate.coarea import try_automatic_coarea
    from multiple_integrate.policy import _with_policy
    from multiple_integrate.region_types import BoxRegion

    assert _with_policy(try_automatic_coarea)(f, BoxRegion(box), box, None, policy=low) is None
    clear_cache()
    assert _automatic_geometry.cache_info().currsize == 0


def test_non_gaussian_radial_dispatch_with_polynomial_amplitude():
    clear_cache()
    r = integrate_result(x * x * sp.exp(-q * q), (y, -sp.oo, sp.oo), (x, -sp.oo, sp.oo))
    assert r.method == "coarea_radial" and r.value == sp.pi / 4 and r.absolute_convergent


def test_cached_coarea_result_replays_separate_evidence():
    clear_cache()
    f = sp.exp(-((x + y) ** 3))
    fresh = integrate_result(f, *box)
    cached = integrate_result(f, *box)
    assert cached.value == fresh.value and cached.absolute_convergent
    assert any(c.scope == "coarea_fibers" and c.complete for c in cached.coverage)


def test_fast_separable_affine_exponential_does_not_enter_chart_certifier(monkeypatch):
    import multiple_integrate.coarea as module

    clear_cache()
    monkeypatch.setattr(
        module, "integrate_coarea", lambda *a, **k: pytest.fail("unnecessary coarea proof")
    )
    assert sp.simplify(integrate(sp.exp(-x - y), *box) - (1 - sp.exp(-1)) ** 2) == 0


@pytest.mark.parametrize("dimension", [2, 3])
def test_shell_dispatch_uses_origin_and_finite_angular_majorant(dimension):
    from multiple_integrate.coarea import try_automatic_coarea
    from multiple_integrate.region_types import AnnulusRegion, SphericalShellRegion

    z = sp.Symbol("z", real=True)
    variables = (x, y, z)[:dimension]
    region = (
        AnnulusRegion(variables_xy=variables, inner_radius=1, outer_radius=2)
        if dimension == 2
        else SphericalShellRegion(variables_nd=variables, inner_radius=1, outer_radius=2)
    )
    density = x * sp.exp(-(sum(v * v for v in variables) ** 2))
    result = try_automatic_coarea(density, region, (), None)
    assert result == ("coarea_radial", 0)


def test_unsuitable_exponential_profile_declines_before_any_child_integration(monkeypatch):
    import multiple_integrate.coarea as module
    from multiple_integrate.region_types import BoxRegion

    clear_cache()
    monkeypatch.setattr(
        module, "integrate_coarea", lambda *a, **k: pytest.fail("slow speculative child")
    )
    for f in [
        sp.log(x + y),
        sp.exp(sp.I * (x + y) ** 3),
        sp.exp(-((x + y) ** 3)) * sp.exp(-x * x),
        sp.exp(-((x + y) ** 3) + (x + y) ** 2),
    ]:
        assert module.try_automatic_coarea(f, BoxRegion(box), box, None) is None
