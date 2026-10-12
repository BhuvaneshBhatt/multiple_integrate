"""Endpoint factors, shifted tails and special-function exceptional fibers."""

import mpmath as mp
import pytest
import sympy as sp

from multiple_integrate import IntegrationOutcome, integrate_result
from multiple_integrate.assumptions import _normalize_assumptions
from multiple_integrate.cell_proofs import nonzero
from multiple_integrate.selected_families import selected_interval_rule

x, b = sp.symbols("x b", real=True)


@pytest.mark.parametrize(
    "function,condition",
    [
        (sp.gamma(b), b > 0),
        (sp.gamma(b), b < 0),
        (sp.log(b), b > 1),
        (sp.log(b), sp.And(b > 0, b < 1)),
        (sp.cosh(b), sp.true),
        (sp.exp(b), sp.true),
    ],
)
def test_nonzero_domain_facts(function, condition):
    expected = function != sp.gamma(b) or condition != (b < 0)
    assert nonzero(function, _normalize_assumptions(condition)) is expected


@pytest.mark.parametrize("function", [sp.log(b), sp.gamma(b)])
def test_special_function_divergence_prunes_zero_branch(function):
    r = integrate_result(function * sp.exp(x), (x, 0, sp.oo), assumptions=b > 1)
    assert r.outcome is IntegrationOutcome.DIVERGENT
    assert len(r.cases) == 1


@pytest.mark.parametrize("width", [1, 2, sp.Rational(1, 3)])
@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_exact_endpoint_orders(width, side):
    amplitude = {
        "left": x + x**2,
        "right": (width - x) * (1 + x),
        "both": x * (width - x) * (1 + x),
    }[side]
    p = -sp.Rational(3, 2) if side in ("left", "both") else 0
    q = -sp.Rational(3, 2) if side in ("right", "both") else 0
    expr = sp.Mul(amplitude, x**p, (width - x) ** q, evaluate=False)
    rule = selected_interval_rule(expr, x, 0, width)
    assert rule is not None
    with mp.workdps(40):
        oracle = mp.quad(
            sp.lambdify(x, expr, "mpmath"),
            [0, mp.mpf(str(sp.N(width, 40))) / 2, mp.mpf(str(sp.N(width, 40)))],
        )
        assert abs(complex(rule[1].evalf(30)) - complex(oracle)) < 1e-12


def test_endpoint_threshold_not_relaxed_without_factors():
    assert selected_interval_rule((1 + x) * x ** -sp.Rational(3, 2), x, 0, 1) is None
    assert selected_interval_rule(sp.Mul(x, x**-2, evaluate=False), x, 0, 1) is None


@pytest.mark.parametrize("degree", [3, 4, 6])
@pytest.mark.parametrize("bounds", [(2, 3), (3, sp.oo), (3, 2)])
def test_shifted_truncated_laplace(degree, bounds):
    expr = (1 + x + x**2) * sp.exp(-2 * (x - 2) ** degree)
    rule = selected_interval_rule(expr, x, *bounds)
    assert rule is not None
    with mp.workdps(35):
        oracle = mp.quad(
            sp.lambdify(x, expr, "mpmath"), [mp.inf if v == sp.oo else v for v in bounds]
        )
        assert abs(complex(rule[1].evalf(30)) - complex(oracle)) < 1e-12


def test_shifted_full_line_even_moment():
    _, value = selected_interval_rule(x * sp.exp(-((x - 2) ** 4)), x, -sp.oo, sp.oo)
    assert sp.simplify(value - sp.gamma(sp.Rational(1, 4))) == 0


@pytest.mark.parametrize(
    "expr,lo,hi",
    [
        (sp.exp(-((x - 2) ** 3)), 1, 3),
        (sp.exp(-((x - 2) ** 4)), 1, 3),
        (sp.exp(-(x**4) - x), 0, sp.oo),
        (sp.exp(sp.I * x**4), 0, 1),
    ],
)
def test_shifted_family_declines_unproved_or_out_of_scope(expr, lo, hi):
    assert selected_interval_rule(expr, x, lo, hi) is None


def test_layer_cake_readme_example():
    import re
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    section = readme.split("### Layer-cake identity", 1)[1].split(
        "<!-- geometry-casebook-start -->", 1
    )[0]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    assert len(blocks) == 1
    exec(compile(blocks[0], "README layer cake", "exec"), {})


@pytest.mark.parametrize("density,condition", [(sp.gamma(b), b > 0), (sp.log(b), b > 1)])
def test_power_divergence_special_amplitude(density, condition):
    p = sp.Symbol("p", real=True)
    result = integrate_result(density * x**p, (x, 0, 1), assumptions=sp.And(condition, p < -1))
    assert result.outcome is IntegrationOutcome.DIVERGENT


@pytest.mark.parametrize("value,expected", [(0, False), (1, False), (2, True), (-1, True)])
def test_log_nonzero_exceptional_points(value, expected):
    assert nonzero(sp.log(value), _normalize_assumptions(None)) is expected


def test_shifted_complex_truncation():
    expr = (1 + x) * sp.exp(-(2 + sp.I) * (x - 1) ** 4)
    _, value = selected_interval_rule(expr, x, 2, 3)
    with mp.workdps(35):
        oracle = mp.quad(sp.lambdify(x, expr, "mpmath"), [2, 3])
        assert abs(complex(value.evalf(30)) - complex(oracle)) < 1e-14


def test_symbolic_endpoint_factor_proves_public_convergence():
    p = sp.Symbol("p", real=True)
    expr = (x + x**2) * x**p
    result = integrate_result(expr, (x, 0, 1), assumptions=sp.And(p > -2, p < -1))
    assert result.absolute_convergent is True
    assert sp.simplify(result.value - (1 / (p + 2) + 1 / (p + 3))) == 0


def test_exponential_nonzero_keeps_argument_parameter_domain():
    assert not nonzero(sp.exp(1 / b), _normalize_assumptions(None))
    assert nonzero(sp.exp(1 / b), _normalize_assumptions(b > 0))
