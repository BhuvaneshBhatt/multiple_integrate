"""Independent regressions for exceptional fibers and directional endpoints."""

import importlib
from dataclasses import replace

import pytest
import sympy as sp

from multiple_integrate import (
    ConvergenceEvidence,
    IntegrationOutcome,
    IntegrationResult,
    IntegrationStatus,
    integrate_result,
    verify_finite_result_numerically,
)
from multiple_integrate.convergence_evidence import compact_continuity
from multiple_integrate.quadratic_radial import _evaluate

x, a = sp.symbols("x a", real=True)


@pytest.mark.parametrize("lower,upper,factor", [(-1, 0, 1), (0, 1, 1), (-1, 1, 2), (0, -1, -1)])
def test_directional_oscillatory_endpoint(lower, upper, factor):
    result = integrate_result(sp.sin(1 / x) / x, (x, lower, upper))
    expected = factor * (sp.pi / 2 - sp.Si(1))
    assert sp.simplify(result.value - expected) == 0
    assert result.outcome is IntegrationOutcome.FINITE


@pytest.mark.parametrize("center", [-2, 1, 3])
def test_translated_oscillatory_split(center):
    t = x - center
    result = integrate_result(sp.sin(1 / t) / t, (x, center - 1, center + 1))
    assert sp.simplify(result.value - sp.pi + 2 * sp.Si(1)) == 0


@pytest.mark.parametrize(
    "density", [1 / a, sp.log(a), sp.gamma(a), sp.exp(1 / a), sp.loggamma(a), sp.polygamma(0, a)]
)
@pytest.mark.parametrize("assumptions", [None, a >= 0])
def test_unproved_parameter_fiber_never_receives_finite_certificate(density, assumptions):
    result = integrate_result(density, (x, 0, 1), assumptions=assumptions)
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert result.complete is False and result.convergent is None
    assert result.absolute_convergent is not True and result.ordinary_convergent is not True
    assert not compact_continuity(density, [(x, 0, 1)], assumptions=assumptions)


@pytest.mark.parametrize(
    "density", [1 / a, sp.log(a), sp.gamma(a), sp.exp(1 / a), sp.loggamma(a), sp.polygamma(0, a)]
)
def test_valid_positive_parameter_cell_remains_supported(density):
    result = integrate_result(density, (x, 0, 1), assumptions=a > 0)
    assert result.outcome is IntegrationOutcome.FINITE
    assert result.absolute_convergent is True


def test_nonzero_and_shifted_special_function_domains():
    assert integrate_result(1 / a, (x, 0, 1), assumptions=sp.Ne(a, 0)).finite is True
    assert integrate_result(sp.log(a), (x, 0, 1), assumptions=a < 0).finite is True
    assert integrate_result(sp.gamma(a + 1), (x, 0, 1), assumptions=a >= 0).finite is True
    nonzero = sp.Symbol("nonzero", real=True, nonzero=True)
    assert integrate_result(1 / nonzero, (x, 0, 1)).finite is True


@pytest.mark.parametrize("module_name", ["multiple_integrate", "multiple_integrate.api"])
def test_all_declared_facade_exports_resolve(module_name):
    module = importlib.import_module(module_name)
    assert len(module.__all__) == len(set(module.__all__))
    assert all(hasattr(module, name) for name in module.__all__)
    namespace = {}
    exec(f"from {module_name} import *", namespace)
    assert all(namespace[name] is getattr(module, name) for name in module.__all__)


def test_custom_function_numerical_verification_is_inconclusive():
    class Custom(sp.Function):
        is_real = True
        is_finite = True

    value = Custom(1)
    result = IntegrationResult(value, IntegrationStatus.EXACT, "custom")
    check = verify_finite_result_numerically(result, value, (x, 0, 1))
    assert check.consistent is None and check.certified is False


@pytest.mark.parametrize(
    "claims,expected",
    [
        ([], None),
        ([True], True),
        ([False], False),
        ([None], None),
        ([True, True], True),
        ([True, False], None),
    ],
)
def test_shared_accessor_merges_consistent_claims(claims, expected):
    result = IntegrationResult(
        0,
        IntegrationStatus.EXACT,
        "test",
        convergence_evidence=tuple(
            ConvergenceEvidence("absolute", claim, "test") for claim in claims
        ),
    )
    assert result.absolute_convergent is expected
    scoped = replace(
        result,
        convergence_evidence=(
            *result.convergence_evidence,
            ConvergenceEvidence("absolute", True, "scoped", a > 0),
        ),
    )
    assert scoped.absolute_convergent is (None if expected is False else expected)


def test_equal_radial_indices_reuse_one_moment(monkeypatch):
    import multiple_integrate.quadratic_radial as radial

    y, z = sp.symbols("y z", real=True)
    polynomial = sp.Poly(x * x + y * y + z * z, x, y, z)
    original = radial._moment
    calls = []

    def counted(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(radial, "_moment", counted)
    value = _evaluate(sp.eye(3), polynomial, 1, 1, None, 0, sp.oo)
    assert sp.simplify(value - 3 * sp.pi ** sp.Rational(3, 2) / 2) == 0
    assert calls == [sp.Rational(5, 2)]


@pytest.mark.parametrize("density", [1 / a, sp.log(a)])
def test_conditional_branch_keeps_its_exceptional_fiber(density):
    result = integrate_result(sp.Piecewise((0, a < 0), (density, True)), (x, 0, 1))
    assert result.outcome is IntegrationOutcome.CONDITIONAL
    assert result.complete is False and result.convergent is None
    assert result.cases[0].status is IntegrationStatus.EXACT
    assert result.cases[1].status is IntegrationStatus.UNKNOWN
    assert result.absolute_convergent is not True


def test_conditional_branch_scopes_nonzero_prerequisite():
    result = integrate_result(sp.Piecewise((1 / a, a > 0), (0, True)), (x, 0, 1))
    assert result.complete is True
    assert all(case.status is IntegrationStatus.EXACT for case in result.cases)


@pytest.mark.parametrize("warm_before", [False, True])
def test_temporary_global_facts_cannot_escape_into_convergence_evidence(warm_before, monkeypatch):
    from sympy.assumptions import assuming

    from multiple_integrate import clear_cache

    y = sp.Symbol("y", real=True)
    density = sp.exp(-a * (x**2 + y**2))
    ranges = ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
    # Isolate evidence from an upstream symbolic Gaussian search: the generic
    # formula does not prove its own convergence prerequisites.
    api = importlib.import_module("multiple_integrate.api.integration")
    monkeypatch.setattr(api, "integrate", lambda *args, **kwargs: sp.pi / sp.Abs(a))
    clear_cache()
    if warm_before:
        integrate_result(density, *ranges)
    with assuming(sp.Q.positive(a)):
        temporary = integrate_result(density, *ranges)
        assert temporary.absolute_convergent is not True
        assert temporary.ordinary_convergent is not True
    after = integrate_result(density, *ranges)
    assert after.absolute_convergent is not True
    assert after.ordinary_convergent is not True
    explicit = integrate_result(density, *ranges, assumptions=a > 0)
    assert explicit.absolute_convergent is True and explicit.complete
    assert explicit.value == sp.pi / sp.Abs(a)
    clear_cache()


def test_explicit_cell_queries_ignore_conflicting_global_facts():
    from sympy.assumptions import assuming

    from multiple_integrate.assumptions import _ask, _normalize_assumptions

    with assuming(sp.Q.positive(a)):
        context = _normalize_assumptions(a < 0)
        assert _ask(sp.Q.negative(a), context) is True
        assert _ask(sp.Q.positive(a), context) is False
        assert _ask(sp.Q.positive(2 * a), _normalize_assumptions(None)) is None


@pytest.mark.parametrize("density", [sp.exp(-a * x**2), sp.sin(1 / (x - 1)) / (x - 1)])
def test_backend_recursion_retains_original_request(density, monkeypatch):
    from multiple_integrate.assumptions import _integrate

    def fail(*args, **kwargs):
        raise RecursionError("upstream symbolic search")

    monkeypatch.setattr(sp, "integrate", fail)
    assert _integrate(density, (x, 0, 1)) == sp.Integral(density, (x, 0, 1))


def test_refinement_cannot_cache_temporary_global_signs():
    from sympy.assumptions import assuming

    from multiple_integrate.assumptions import _normalize_assumptions, _refine_with_assumptions

    with assuming(sp.Q.positive(a)):
        assert _refine_with_assumptions(sp.Abs(a), _normalize_assumptions(a >= -1)) == sp.Abs(a)
        assert _refine_with_assumptions(sp.Abs(a), _normalize_assumptions(a < 0)) == -a
