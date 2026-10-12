"""Divergence theorems require the source density's parameter domain too."""

import pytest
import sympy as sp

from multiple_integrate import IntegrationPolicy, IntegrationStatus, integrate_result

x, a, b, p = sp.symbols("x a b p", real=True)
policy = IntegrationPolicy(max_singular_queries=0)


@pytest.mark.parametrize("coefficient", [1 / b, sp.gamma(b), sp.log(b), sp.tan(b)])
def test_endpoint_divergence_cannot_include_undefined_parameter_fibers(coefficient):
    f = coefficient * x**p
    r = integrate_result(f, (x, 0, 1), assumptions=p < -1, policy=policy)
    assert r.status is IntegrationStatus.UNKNOWN and r.ordinary_convergent is None
    assert not r.complete


@pytest.mark.parametrize(
    "f,assumptions",
    [
        (x**p / b, sp.And(p < -1, sp.Ne(b, 0))),
        (sp.gamma(b) * x**p, sp.And(p < -1, b > 0, sp.Ne(sp.gamma(b), 0))),
        (sp.log(b) * x**p, sp.And(p < -1, b > 1, sp.Ne(sp.log(b), 0))),
    ],
)
def test_valid_external_domains_preserve_endpoint_divergence(f, assumptions):
    r = integrate_result(f, (x, 0, 1), assumptions=assumptions, policy=policy)
    assert r.status is IntegrationStatus.DIVERGENT and r.ordinary_convergent is False


@pytest.mark.parametrize("phase", [a * x + 1 / b, a * x * x + 1 / b])
def test_exponential_divergence_requires_a_defined_constant_phase(phase):
    limits = (x, 0, sp.oo) if sp.degree(phase, x) == 1 else (x, -sp.oo, sp.oo)
    f = sp.exp(phase)
    r = integrate_result(f, limits, assumptions=a > 0, policy=policy)
    assert r.status is IntegrationStatus.UNKNOWN and r.ordinary_convergent is None
    good = integrate_result(f, limits, assumptions=sp.And(a > 0, sp.Ne(b, 0)), policy=policy)
    assert good.status is IntegrationStatus.DIVERGENT and good.ordinary_convergent is False


def test_gamma_pole_and_finite_piecewise_extension_remain_distinct():
    f = sp.Piecewise((1, sp.Eq(b, 0)), (1 / b, True)) * x**p
    r = integrate_result(f, (x, 0, 1), assumptions=p < -1, policy=policy)
    assert r.ordinary_convergent is False


@pytest.mark.parametrize(
    "density,limits",
    [
        (1 / (b * x * x), (x, 0, 1)),
        (sp.gamma(b) * sp.exp(x), (x, 0, sp.oo)),
        (sp.exp(x + 1 / b), (x, 0, sp.oo)),
        (sp.exp(x * x + 1 / b), (x, -sp.oo, sp.oo)),
    ],
)
def test_fallback_divergence_has_the_same_external_domain_guard(density, limits):
    r = integrate_result(density, limits, policy=policy)
    assert r.status is IntegrationStatus.UNKNOWN and r.ordinary_convergent is not False
    assert any("defined external parameter domain" in s for s in r.limitations)
    good = integrate_result(density.subs(b, 1), limits, policy=policy)
    assert good.status is IntegrationStatus.DIVERGENT


@pytest.mark.parametrize(
    "kernel,limits",
    [
        (x**-2, (x, 0, 1)),
        (sp.exp(x), (x, 0, sp.oo)),
        (sp.exp(x * x), (x, -sp.oo, sp.oo)),
    ],
)
def test_elementary_divergence_keeps_zero_amplitude_fiber(kernel, limits):
    r = integrate_result(b * kernel, limits, policy=policy)
    assert r.status is IntegrationStatus.CONDITIONAL
    assert any(
        c.value == 0 and c.status is IntegrationStatus.EXACT and c.condition == sp.Eq(b, 0)
        for c in r.cases
    )
    assert any(
        c.status is IntegrationStatus.DIVERGENT and c.condition == sp.Ne(b, 0) for c in r.cases
    )
    assert r.ordinary_convergent is None
    assert r.value.subs(b, 0) == 0
    zero = integrate_result(b * kernel, limits, assumptions=sp.Eq(b, 0), policy=policy)
    assert zero.status is IntegrationStatus.EXACT and zero.cases[0].value == 0
    assert zero.value == 0 and zero.absolute_convergent is True
    good = integrate_result(b * kernel, limits, assumptions=sp.Ne(b, 0), policy=policy)
    assert good.status is IntegrationStatus.DIVERGENT and good.ordinary_convergent is False


def test_unregistered_fallback_declines_unproved_amplitude_nonzero():
    # This family is outside the elementary theorem registry.
    r = integrate_result(b * sp.exp(x**4), (x, 0, sp.oo), policy=policy)
    assert r.status is not IntegrationStatus.DIVERGENT
    assert r.ordinary_convergent is not False
