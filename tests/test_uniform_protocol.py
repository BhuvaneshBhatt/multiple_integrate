"""Public outcome semantics, coverage accounting, and serious-bug regressions."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sympy as sp

import multiple_integrate.api.integration as api
from multiple_integrate import (
    GeometricPrincipalValue,
    HadamardFinitePart,
    IntegrationOutcome,
    IntegrationStatus,
    clear_cache,
    integrate,
    integrate_result,
    regularized_result,
)
from multiple_integrate.conditional import conditional_status
from multiple_integrate.hypersurface_atlas import HypersurfaceAtlas, HypersurfaceChart
from multiple_integrate.hypersurface_principal_value import _atlas_cutoff_integral
from multiple_integrate.planner import coverage_trace, fresh_strategy_trace
from multiple_integrate.region_types import (
    AffineSimplexRegion,
    AnnulusRegion,
    DiskRegion,
    EllipsoidRegion,
    SphericalShellRegion,
)
from multiple_integrate.results import UnresolvedIntegration

x, y, z, a = sp.symbols("x y z a", real=True)


@pytest.mark.parametrize(
    "factory,volume",
    [
        (lambda vs: EllipsoidRegion(variables_nd=vs, axes=(1, 1)), sp.pi),
        (lambda vs: AnnulusRegion(variables_xy=vs, inner_radius=1, outer_radius=2), 3 * sp.pi),
        (
            lambda vs: SphericalShellRegion(variables_nd=vs, inner_radius=1, outer_radius=2),
            3 * sp.pi,
        ),
    ],
)
def test_cache_preserves_region_coordinates(factory, volume):
    clear_cache()
    assert integrate(x, region=factory((x, y))) == 0
    assert integrate(x, region=factory((y, z))) == volume * x
    clear_cache()


def test_geometric_pv_keeps_user_parameter():
    p = sp.Symbol("_pv_u0", real=True)
    disk = DiskRegion(((y, -sp.sqrt(1 - x * x), sp.sqrt(1 - x * x)), (x, -1, 1)), radius=1)
    r = regularized_result(p, region=disk, principal_value=GeometricPrincipalValue(point=(0, 0)))
    assert r.value == sp.pi * p and r.outcome is IntegrationOutcome.FINITE
    assert any(e.scope == "angular_domain" and e.complete for e in r.coverage)


@pytest.mark.parametrize(
    "value,outcome",
    [
        (sp.Integer(2), IntegrationOutcome.FINITE),
        (sp.oo, IntegrationOutcome.DIVERGENT),
        (sp.zoo, IntegrationOutcome.DIVERGENT),
        (sp.nan, IntegrationOutcome.UNKNOWN),
        (sp.Integral(sp.sin(x * x), (x, 0, 1)), IntegrationOutcome.UNKNOWN),
        (sp.Limit(sp.sin(x), x, sp.oo), IntegrationOutcome.UNKNOWN),
        (sp.Piecewise((1, a > 0), (sp.oo, True)), IntegrationOutcome.CONDITIONAL),
    ],
)
def test_both_result_apis_classify_same_outcomes(monkeypatch, value, outcome):
    monkeypatch.setattr(api, "integrate", lambda *args, **kwargs: value)
    ordinary = integrate_result(x, (x, 0, 1), principal_value=True)
    regularized = regularized_result(x, (x, 0, 1), principal_value=True)
    assert ordinary.outcome is regularized.outcome is outcome
    assert ordinary.exact == regularized.exact
    assert ordinary.complete == regularized.complete
    assert ordinary.cases == regularized.cases


@pytest.mark.parametrize("api_function", [integrate_result, regularized_result])
def test_divergent_pv_is_not_successful_regularization(api_function):
    result = api_function(x**-2, (x, -1, 1), principal_value=True)
    assert result.outcome is IntegrationOutcome.DIVERGENT
    assert not result.exact and result.finite is False


@pytest.mark.parametrize("api_function", [integrate_result, regularized_result])
def test_unsupported_prescription_retains_request(api_function):
    result = api_function(sp.sin(x), (x, 1, sp.oo), regularization=HadamardFinitePart())
    assert result.outcome is IntegrationOutcome.UNSUPPORTED
    assert result.value == sp.Integral(sp.sin(x), (x, 1, sp.oo))
    assert result.finite is None and not result.complete and result.limitations


def test_flat_api_still_raises_for_unsupported_request():
    with pytest.raises(NotImplementedError):
        integrate(sp.sin(x), (x, 1, sp.oo), regularization=HadamardFinitePart())


@pytest.mark.parametrize("api_function", [integrate_result, regularized_result])
def test_backend_uncertainty_is_unknown(monkeypatch, api_function):
    def unresolved(*args, **kwargs):
        raise UnresolvedIntegration("uncertified domain")

    monkeypatch.setattr(api, "integrate", unresolved)
    result = api_function(x, (x, 0, 1), principal_value=True)
    assert result.outcome is IntegrationOutcome.UNKNOWN and not result.complete


@pytest.mark.parametrize("api_function", [integrate_result, regularized_result])
def test_invalid_inputs_still_raise(api_function):
    with pytest.raises(ValueError):
        api_function(x, (x, 0, 1), assumptions=False, principal_value=True)


def test_nested_piecewise_is_conditional():
    value = 2 * sp.Piecewise((1, a > 0), (sp.oo, True), evaluate=False)
    status, cases, _, _ = conditional_status(value)
    assert status is IntegrationStatus.CONDITIONAL and len(cases) == 2


@pytest.fixture
def atlas_data():
    t = sp.Dummy("level", real=True)
    e = sp.Dummy("epsilon", positive=True)
    g = x * x - 1
    charts = tuple(
        HypersurfaceChart(
            x, t, g, 2 * x, sign * sp.sqrt(t + 1), 1 / (2 * sp.sqrt(t + 1)), sign * x > 0, sign, i
        )
        for i, sign in enumerate((1, -1))
    )
    atlas = HypersurfaceAtlas(g, (x,), charts, True)
    reduction = SimpleNamespace(
        pieces=(SimpleNamespace(integrand=sp.Integer(1), limits=((x, 0, 2),)),)
    )
    return atlas, e, reduction


@pytest.mark.parametrize("missing_kind", ["unavailable", "empty", "unrealized"])
def test_atlas_cannot_silently_drop_patch(atlas_data, missing_kind):
    from dataclasses import replace

    atlas, e, reduced = atlas_data
    if missing_kind == "unrealized":
        atlas = replace(
            atlas, charts=(atlas.charts[0], replace(atlas.charts[1], inverse=None, jacobian=None))
        )
    missing = None if missing_kind == "unavailable" else SimpleNamespace(pieces=())
    with (
        fresh_strategy_trace(),
        patch(
            "multiple_integrate.hypersurface_principal_value.reduce_domain",
            side_effect=[reduced, missing],
        ),
        patch("multiple_integrate.hypersurface_principal_value.semialg_implies", return_value=None),
    ):
        with pytest.raises(NotImplementedError):
            _atlas_cutoff_integral(sp.Integer(1), sp.And(x >= -2, x <= 2), (x,), atlas, e)
        assert any(c.scope == "atlas_charts" and c.complete is False for c in coverage_trace())


def test_certified_empty_patch_may_be_skipped(atlas_data):
    atlas, e, reduced = atlas_data
    with (
        fresh_strategy_trace(),
        patch(
            "multiple_integrate.hypersurface_principal_value.reduce_domain",
            side_effect=[reduced, None],
        ),
        patch("multiple_integrate.hypersurface_principal_value.semialg_implies", return_value=True),
    ):
        cutoff, details = _atlas_cutoff_integral(
            sp.Integer(1), sp.And(x >= 0, x <= 2), (x,), atlas, e
        )
        assert sp.limit(cutoff, e, 0, dir="+") == 2
        assert details["coverage_complete"] and details["atlas_empty_chart_count"] == 1
        assert coverage_trace()[-1].complete is True


def test_parameter_coverage_does_not_claim_known_values():
    r = integrate_result(sp.exp(-a * x), (x, 0, sp.oo))
    assert any(e.scope == "parameter_space" and e.complete for e in r.coverage)
    assert r.outcome is IntegrationOutcome.CONDITIONAL


def test_native_and_cache_coverage_are_distinct():
    clear_cache()
    first = integrate_result(x, (x, 0, 1))
    cached = integrate_result(x, (x, 0, 1))
    assert any(e.scope == "integration_domain" for e in first.coverage)
    assert any(e.scope == "problem_identity" for e in cached.coverage)


def test_affine_simplex_cache_coordinates():

    def region(coords):
        return AffineSimplexRegion(
            tuple((v, 0, 1) for v in coords), shifts=(0, 0), scales=(1, 1), dimension=2
        )

    clear_cache()
    assert integrate(x, region=region((x, y))) == sp.Rational(1, 6)
    assert integrate(x, region=region((y, z))) == x / 2


def test_unsupported_region_placeholder_preserves_bounds(monkeypatch):
    def refuse(*args, **kwargs):
        raise NotImplementedError("no route")

    monkeypatch.setattr(api, "integrate", refuse)
    r = integrate_result(1, region=x > 0, variables=(x,), bounds={x: (0, 1)})
    assert r.outcome is IntegrationOutcome.UNSUPPORTED
    assert r.value.domain.has(x <= 1)


def test_coverage_trace_does_not_leak_between_calls():
    clear_cache()
    native = integrate_result(x, (x, 0, 1))
    unresolved = integrate_result(sp.nan, (x, 0, 1))
    assert native.coverage
    assert not any(e.scope == "problem_identity" for e in unresolved.coverage)


def test_actual_geometry_backend_refusal_is_unknown(monkeypatch):
    import multiple_integrate.core as core

    monkeypatch.setattr(core, "reduce_domain", lambda *args, **kwargs: None)
    monkeypatch.setattr("multiple_integrate.adapters.semialg.region_dimension", lambda *args: None)
    r = integrate_result(1, region=sp.And(x > 0, y > 0), variables=(x, y))
    assert r.outcome is IntegrationOutcome.UNKNOWN and not r.complete


def test_certified_interior_divergence_survives_nan_sum():
    r = integrate_result(1 / x, (x, -1, 1))
    assert r.outcome is IntegrationOutcome.DIVERGENT
    assert any(attempt.method == "divergent_interval_piece" for attempt in r.attempts)
    assert any(e.scope == "interval_partition" for e in r.coverage)


def test_uncertified_regularity_is_unknown(monkeypatch):
    from multiple_integrate import HypersurfacePrincipalValue
    from multiple_integrate.region_types import BoxRegion

    monkeypatch.setattr(
        "multiple_integrate.hypersurface_atlas.semialg_is_satisfiable", lambda *args: None
    )
    r = integrate_result(
        1 / (x * x + y * y - 1),
        region=BoxRegion(((x, -2, 2), (y, -2, 2))),
        principal_value=HypersurfacePrincipalValue(x * x + y * y - 1),
    )
    assert r.outcome is IntegrationOutcome.UNKNOWN


def test_regularized_path_does_not_compute_unused_singular_diagnostics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unneeded singularity diagnostic")

    monkeypatch.setattr(api, "stratify_singular_locus", forbidden)
    assert (
        regularized_result(1 / x, (x, -1, 1), principal_value=True).outcome
        is IntegrationOutcome.FINITE
    )


def test_nonfinite_result_cannot_verify_as_successful_regularization():
    from multiple_integrate import verify_regularization_result

    r = regularized_result(x**-2, (x, -1, 1), principal_value=True)
    assert not verify_regularization_result(r).valid


def test_divergence_proof_is_preserved_from_flat_cache():
    clear_cache()
    assert integrate(1 / x, (x, -1, 1)) is sp.nan
    r = integrate_result(1 / x, (x, -1, 1))
    assert r.method == "cache" and r.outcome is IntegrationOutcome.DIVERGENT
    assert any("cached one-sided" in (attempt.reason or "") for attempt in r.attempts)
    assert integrate_result(sp.nan, (x, 0, 1)).outcome is IntegrationOutcome.UNKNOWN


def test_public_result_retains_failed_atlas_coverage(atlas_data, monkeypatch):
    from multiple_integrate import HypersurfacePrincipalValue
    from multiple_integrate.region_types import BoxRegion

    atlas, e, reduced = atlas_data
    monkeypatch.setattr(
        "multiple_integrate.hypersurface_principal_value._hypersurface_graph_chart",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "multiple_integrate.hypersurface_principal_value.build_hypersurface_atlas",
        lambda *args: atlas,
    )
    with (
        patch(
            "multiple_integrate.hypersurface_principal_value.reduce_domain",
            side_effect=[reduced, None],
        ),
        patch("multiple_integrate.hypersurface_principal_value.semialg_implies", return_value=None),
    ):
        r = integrate_result(
            1 / (x * x - 1) ** 2,
            region=BoxRegion(((x, -2, 2),)),
            principal_value=HypersurfacePrincipalValue(x * x - 1),
        )
    assert r.outcome is IntegrationOutcome.UNKNOWN and not r.complete
    assert any(c.scope == "atlas_charts" and c.complete is False for c in r.coverage)


def test_all_certified_empty_atlas_charts_are_zero(atlas_data):
    atlas, e, reduced = atlas_data
    with (
        fresh_strategy_trace(),
        patch("multiple_integrate.hypersurface_principal_value.reduce_domain", return_value=None),
        patch("multiple_integrate.hypersurface_principal_value.semialg_implies", return_value=True),
    ):
        cutoff, details = _atlas_cutoff_integral(sp.Integer(1), sp.false, (x,), atlas, e)
        assert cutoff == 0 and details["atlas_empty_chart_count"] == 2
        assert coverage_trace()[-1].complete is True


def test_verified_intrinsic_cover_has_coverage_evidence():
    import semialg

    region = semialg.Simplex(((0, 0), (3, 4)))
    r = integrate_result(1, region=region, variables=(x, y), measure_dimension=1)
    assert r.value == 5
    assert any(e.scope == "intrinsic_cover" and e.complete for e in r.coverage)


def test_canonical_ambient_null_set_has_dimension_evidence():
    import semialg

    r = integrate_result(1, region=semialg.Point((0, 0)), variables=(x, y))
    assert r.value == 0
    assert any(e.scope == "ambient_null_set" and e.complete for e in r.coverage)
