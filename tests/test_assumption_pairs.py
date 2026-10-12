import sympy as sp

from multiple_integrate import IntegrationStatus, integrate_result


def test_exponential_tail_assumption_pair():
    x, a = sp.symbols("x a", real=True)
    unconstrained = integrate_result(sp.exp(-a * x), (x, 0, sp.oo))
    constrained = integrate_result(sp.exp(-a * x), (x, 0, sp.oo), assumptions=a > 0)
    assert unconstrained.status is IntegrationStatus.CONDITIONAL
    assert constrained.status is IntegrationStatus.EXACT
    assert constrained.value == 1 / a


def test_gaussian_assumption_pair():
    x, a = sp.symbols("x a", real=True)
    unconstrained = integrate_result(sp.exp(-a * x**2), (x, -sp.oo, sp.oo))
    constrained = integrate_result(sp.exp(-a * x**2), (x, -sp.oo, sp.oo), assumptions=a > 0)
    assert unconstrained.status in {IntegrationStatus.CONDITIONAL, IntegrationStatus.UNKNOWN}
    assert constrained.status is IntegrationStatus.EXACT
    assert constrained.value == sp.sqrt(sp.pi) / sp.sqrt(a)


def test_gamma_tail_assumption_pair():
    x, a = sp.symbols("x a", real=True)
    unconstrained = integrate_result(x**a * sp.exp(-x), (x, 0, sp.oo))
    constrained = integrate_result(
        x**a * sp.exp(-x),
        (x, 0, sp.oo),
        assumptions=a > -1,
    )
    assert unconstrained.status is IntegrationStatus.CONDITIONAL
    assert constrained.status is IntegrationStatus.EXACT
    assert constrained.value == sp.gamma(a + 1)


def test_power_tail_assumption_pair():
    x, a = sp.symbols("x a", real=True)
    unconstrained = integrate_result(x ** (-a), (x, 1, sp.oo))
    constrained = integrate_result(x ** (-a), (x, 1, sp.oo), assumptions=a > 1)
    assert unconstrained.status is IntegrationStatus.CONDITIONAL
    assert constrained.status is IntegrationStatus.EXACT
    assert constrained.value == 1 / (a - 1)


def test_beta_assumption_pair():
    x, a, b = sp.symbols("x a b", real=True)
    expression = x ** (a - 1) * (1 - x) ** (b - 1)
    unconstrained = integrate_result(expression, (x, 0, 1))
    constrained = integrate_result(
        expression,
        (x, 0, 1),
        assumptions=sp.And(a > 0, b > 0),
    )
    assert unconstrained.status is IntegrationStatus.CONDITIONAL
    assert constrained.status is IntegrationStatus.EXACT
    assert constrained.value == sp.beta(a, b)
