import sympy as sp

from multiple_integrate import integrate_result
from multiple_integrate.convergence import CONVERGENCE_RULES
from multiple_integrate.results import IntegrationStatus


def test_registry_contains_supported_theorem_families():
    assert tuple(rule.__name__ for rule in CONVERGENCE_RULES) == (
        "_linear_exponential",
        "_power_endpoint",
        "_gamma_tail",
        "_beta_interval",
        "_gaussian",
    )


def test_gamma_tail_parameter_condition():
    x = sp.Symbol("x", positive=True)
    a, p = sp.symbols("a p", real=True)
    result = integrate_result(x**p * sp.exp(-a * x), (x, 0, sp.oo))
    assert result.status is IntegrationStatus.CONDITIONAL
    assert any(case.condition.has(a, p) for case in result.cases)
    assert any(case.value.has(sp.gamma(p + 1)) for case in result.cases if case.convergent)


def test_beta_endpoint_parameter_condition():
    x = sp.Symbol("x", real=True)
    a, b = sp.symbols("a b", real=True)
    result = integrate_result(x**a * (1 - x) ** b, (x, 0, 1))
    assert result.status is IntegrationStatus.CONDITIONAL
    assert any(case.condition.has(a, b) for case in result.cases)
