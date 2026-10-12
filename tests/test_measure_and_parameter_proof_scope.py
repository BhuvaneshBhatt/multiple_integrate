"""Domain and measure scope for convergence certificates and parameter fibers."""

import pytest
import semialg
import sympy as sp

from multiple_integrate import IntegrationOutcome, integrate_result
from multiple_integrate.api.integration import _evaluate_result
from multiple_integrate.core import integrate
from multiple_integrate.planner import coverage_trace, fresh_strategy_trace
from multiple_integrate.quadratic_radial import radial_region_data, weighted_quadratic_data
from multiple_integrate.region_types import AnnulusRegion, DiskRegion, UnionRegion
from multiple_integrate.singularities import stratify_singular_locus

x, y, b = sp.symbols("x y b", real=True)
FULL = ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))


def regions():
    shell = AnnulusRegion(variables_xy=(x, y), inner_radius=1, outer_radius=2)
    disk = DiskRegion(
        (
            (x, -sp.sqrt(sp.Rational(1, 4) - y * y), sp.sqrt(sp.Rational(1, 4) - y * y)),
            (y, -sp.Rational(1, 2), sp.Rational(1, 2)),
        ),
        radius=sp.Rational(1, 2),
    )
    return shell, disk


def test_union_piece_proof_cannot_certify_divergent_domain():
    shell, disk = regions()
    r = _evaluate_result(
        (x * x + y * y) ** -2,
        region=UnionRegion(pieces=(shell, disk)),
        _include_singular_strata=False,
    )
    assert r.outcome is IntegrationOutcome.DIVERGENT
    assert r.absolute_convergent is not True and r.ordinary_convergent is not True
    assert not any(c.scope == "absolute_convergence" and c.complete is True for c in r.coverage)


def test_all_union_pieces_can_still_prove_convergence():
    shell, disk = regions()
    r = integrate_result(sp.exp(-x * x - y * y), region=UnionRegion(pieces=(shell, disk)))
    assert r.absolute_convergent is True


@pytest.mark.parametrize("dimension", [0, 1])
def test_ambient_proof_does_not_certify_another_measure(dimension):
    r = integrate_result(
        sp.exp(-x * x - y * y),
        region=semialg.Ball((0, 0), 1),
        variables=(x, y),
        measure_dimension=dimension,
    )
    assert r.outcome is IntegrationOutcome.UNKNOWN
    assert r.absolute_convergent is not True and r.ordinary_convergent is not True


@pytest.mark.parametrize("dimension", ["ambient", "intrinsic", 2])
def test_matching_measure_retains_radial_proof(dimension):
    r = integrate_result(
        sp.exp(-x * x - y * y),
        region=semialg.Ball((0, 0), 1),
        variables=(x, y),
        measure_dimension=dimension,
    )
    assert r.absolute_convergent is True


@pytest.mark.parametrize("shift", [b, b * b])
def test_nonnegative_shift_retains_zero_fiber(shift):
    f = (shift + x * x + y * y) ** -2
    assumptions = shift >= 0
    assert weighted_quadratic_data(f, (x, y), assumptions=assumptions) is None
    r = integrate_result(f, *FULL, assumptions=assumptions)
    assert r.outcome is IntegrationOutcome.CONDITIONAL
    assert r.value.subs(b, 0) == sp.oo
    assert r.value.subs(b, 1) == sp.pi
    assert r.absolute_convergent is not True


def test_zero_shift_and_positive_shift_have_distinct_results():
    f = (b + x * x + y * y) ** -2
    zero = integrate_result(f, *FULL, assumptions=sp.Eq(b, 0))
    positive = integrate_result(f, *FULL, assumptions=b > 0)
    assert zero.outcome is IntegrationOutcome.DIVERGENT
    assert positive.absolute_convergent is True and positive.value == sp.pi / b


def test_polynomial_shift_keeps_unresolved_zero_fiber():
    r = integrate_result((1 + x * x) / (b + x * x + y * y) ** 3, *FULL, assumptions=b >= 0)
    assert r.value.subs(b, 0).has(sp.Integral)
    assert r.value.subs(b, 1) == 3 * sp.pi / 4
    assert r.absolute_convergent is not True


def test_shell_shift_formula_requires_resolved_zero_branch():
    shell, _ = regions()
    assert radial_region_data((b + x * x + y * y) ** -2, shell, assumptions=b >= 0) is None
    assert radial_region_data((b + x * x + y * y) ** -2, shell, assumptions=sp.Eq(b, 0)) is not None


@pytest.mark.parametrize("query", ["semialg_region_dimension", "semialg_is_satisfiable"])
def test_failed_auxiliary_query_retains_unknown_stratum(monkeypatch, query):
    def fail(*args, **kwargs):
        raise sp.polys.polyerrors.NotAlgebraic("unresolved symbolic coefficient")

    monkeypatch.setattr("multiple_integrate.singularities." + query, fail)
    details = stratify_singular_locus(1 / (b + x * x + y * y), (x, y), return_details=True)
    assert details.strata
    if query == "semialg_region_dimension":
        assert details.strata[0].dimension is None
    else:
        assert details.strata[0].regular is None


def test_nested_integration_cannot_export_child_certificate(monkeypatch):
    import multiple_integrate.core as core

    original = core._evaluate_integral

    def outer(f, *ranges, **kwargs):
        if f == sp.Symbol("outer"):
            integrate(sp.exp(-x * x - y * y), *FULL)
            return sp.Integral(1 / x, (x, 0, 1))
        return original(f, *ranges, **kwargs)

    monkeypatch.setattr(core, "_evaluate_integral", outer)
    with fresh_strategy_trace():
        integrate(sp.Symbol("outer"), (x, 0, 1))
        assert not any(c.scope == "absolute_convergence" for c in coverage_trace())
    r = integrate_result(sp.exp(-x * x - y * y), *FULL)
    assert r.absolute_convergent is True


def test_ball_shift_preserves_divergent_zero_fiber():
    r = integrate_result(
        (b + x * x + y * y) ** -2,
        region=semialg.Ball((0, 0), 1),
        variables=(x, y),
        assumptions=b >= 0,
    )
    assert r.outcome is IntegrationOutcome.CONDITIONAL
    assert r.value.subs(b, 0) == sp.oo
    assert abs(sp.N(r.value.subs(b, 1) - sp.pi / 2, 40)) < sp.Rational(1, 10**35)
    assert r.absolute_convergent is not True


def test_shell_shift_has_two_finite_fibers():
    shell, _ = regions()
    r = integrate_result((b + x * x + y * y) ** -2, region=shell, assumptions=b >= 0)
    assert r.value.subs(b, 0) == 3 * sp.pi / 4
    assert abs(sp.N(r.value.subs(b, 1) - 3 * sp.pi / 10, 40)) < sp.Rational(1, 10**35)


def test_signed_zero_fiber_keeps_divergence_direction():
    r = integrate_result(-((b + x * x + y * y) ** -2), *FULL, assumptions=b >= 0)
    assert r.value.subs(b, 0) == -sp.oo
    assert r.value.subs(b, 1) == -sp.pi


def test_native_box_proof_does_not_certify_counting_measure():
    from multiple_integrate.region_types import BoxRegion

    r = integrate_result(sp.exp(x), region=BoxRegion(((x, 0, 1),)), measure_dimension=0)
    assert r.outcome is IntegrationOutcome.UNKNOWN
    assert r.absolute_convergent is not True and r.ordinary_convergent is not True


@pytest.mark.parametrize("prescription", ["finite_part", "hypersurface_pv"])
def test_ambient_prescription_cannot_ignore_requested_measure(prescription):
    from multiple_integrate import HadamardFinitePart, HypersurfacePrincipalValue
    from multiple_integrate.region_types import BoxRegion

    kwargs = (
        {"regularization": HadamardFinitePart(hypersurface=x * y)}
        if prescription == "finite_part"
        else {"principal_value": HypersurfacePrincipalValue(x * y)}
    )
    r = integrate_result(
        1 / (x * y) ** 2, region=BoxRegion(((x, -1, 1), (y, -1, 1))), measure_dimension=0, **kwargs
    )
    assert r.outcome is IntegrationOutcome.UNSUPPORTED
    assert r.absolute_convergent is not True
