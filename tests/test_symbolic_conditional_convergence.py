import sympy as sp

from multiple_integrate import clear_cache, multiple_integrate


def test_exponential_tail_uses_positive_parameter_assumption():
    x, a = sp.symbols("x a", real=True)
    assert multiple_integrate(sp.exp(-a * x), (x, 0, sp.oo), assumptions={a > 0}) == 1 / a


def test_exponential_tail_does_not_claim_convergence_for_negative_parameter():
    x, a = sp.symbols("x a", real=True)
    result = multiple_integrate(sp.exp(-a * x), (x, 0, sp.oo), assumptions={a < 0})
    assert isinstance(result, sp.Integral) or result.has(sp.Integral)


def test_power_tail_uses_a_greater_than_one_condition():
    x, a = sp.symbols("x a", real=True)
    result = multiple_integrate(x ** (-a), (x, 1, sp.oo), assumptions={a > 1})
    assert sp.simplify(result - 1 / (a - 1)) == 0


def test_power_tail_does_not_claim_convergence_when_a_less_than_one():
    x, a = sp.symbols("x a", real=True)
    result = multiple_integrate(x ** (-a), (x, 1, sp.oo), assumptions={a < 1})
    assert isinstance(result, sp.Integral) or result.has(sp.Integral)


def test_symbolic_gaussian_uses_positive_assumption():
    x, a = sp.symbols("x a", real=True)
    result = multiple_integrate(sp.exp(-a * x**2), (x, -sp.oo, sp.oo), assumptions={a > 0})
    assert result == sp.sqrt(sp.pi) / sp.sqrt(a)


def test_unit_interval_power_uses_positive_assumption():
    x, a = sp.symbols("x a", real=True)
    result = multiple_integrate(x ** (a - 1), (x, 0, 1), assumptions={a > 0})
    assert sp.simplify(result - 1 / a) == 0


def test_conditional_convergence_cache_isolated_by_assumptions():
    x, a = sp.symbols("x a", real=True)
    clear_cache()
    convergent = multiple_integrate(x ** (-a), (x, 1, sp.oo), assumptions={a > 1})
    divergent = multiple_integrate(x ** (-a), (x, 1, sp.oo), assumptions={a < 1})
    assert sp.simplify(convergent - 1 / (a - 1)) == 0
    assert isinstance(divergent, sp.Integral) or divergent.has(sp.Integral)
