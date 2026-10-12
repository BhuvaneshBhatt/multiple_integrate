"""Independent full-space improper identities and prerequisite refusals."""

import re
from pathlib import Path

import mpmath as mp
import pytest
import sympy as sp

from multiple_integrate import IntegrationPolicy, StrategyBudget, integrate_result
from multiple_integrate.nonseparable_improper import improper_data

x, y, z, a, s = sp.symbols("x y z a s", real=True)
full = ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))


@pytest.mark.parametrize("ranges", [full, tuple(reversed(full))])
@pytest.mark.parametrize(
    "density,expected",
    [
        (sp.exp(-(x**4) - (y - x * x) ** 2), sp.sqrt(sp.pi) * sp.gamma(sp.Rational(1, 4)) / 2),
        (
            (1 + x**4 + (y - x * x) ** 2) ** -2,
            sp.sqrt(sp.pi) * sp.gamma(sp.Rational(1, 4)) ** 2 / 8,
        ),
        (
            x * x / (1 + x**4 + (y - x * x) ** 2) ** 2,
            sp.sqrt(sp.pi) * sp.gamma(sp.Rational(3, 4)) ** 2 / 2,
        ),
        ((y - x * x) * sp.exp(-(x**4) - (y - x * x) ** 2), 0),
    ],
)
def test_shear_identities_and_order(density, expected, ranges):
    result = integrate_result(density, *ranges)
    assert sp.simplify(result.value - expected) == 0
    assert result.complete and result.absolute_convergent
    assert result.coverage and result.convergence_evidence


@pytest.mark.parametrize("power", [sp.Rational(1, 4), sp.Rational(1, 2)])
def test_integrable_degenerate_origin_with_exponential_tail(power):
    q = x**4 + (y - x * x) ** 2
    result = integrate_result(sp.exp(-q) / q**power, *full)
    expected = (
        sp.sqrt(sp.pi)
        * sp.gamma(sp.Rational(1, 4))
        * sp.gamma(sp.Rational(3, 4) - power)
        / (2 * sp.gamma(sp.Rational(3, 4)))
    )
    assert sp.simplify(result.value - expected) == 0
    assert result.absolute_convergent


@pytest.mark.parametrize("power", [sp.Rational(3, 4), 1, 2])
def test_origin_threshold_never_hidden_by_odd_cancellation(power):
    q = x**4 + y * y
    assert improper_data(sp.exp(-q) / q**power, full) is None
    assert improper_data(x * sp.exp(-q) / q ** (power + sp.Rational(1, 4)), full) is None


def test_polynomial_shear_parameter_pole_requires_exclusion():
    q = x**4 + (y - x * x / a) ** 2
    assert improper_data(sp.exp(-q), full) is None
    assert improper_data(sp.exp(-q), full, assumptions=sp.Ne(a, 0)) is not None


def test_parameter_tail_prerequisites():
    f = (1 + x**4 + (y - x * x) ** 2) ** (-s)
    assert improper_data(f, full) is None
    result = integrate_result(f, *full, assumptions=s > sp.Rational(3, 4))
    expected = (
        sp.sqrt(sp.pi)
        * sp.gamma(sp.Rational(1, 4))
        * sp.gamma(s - sp.Rational(3, 4))
        / (2 * sp.gamma(s))
    )
    assert sp.simplify(result.value - expected) == 0 and result.absolute_convergent


@pytest.mark.parametrize(
    "density",
    [
        sp.exp(x**4 - y * y),
        sp.exp(-(x**4) - (y - x**3) ** 2 - y**2),
        sp.exp(-(x**4) - y * y) / (1 + x * x + y * y),
        1 / (x**4 + y * y) ** 2,
    ],
)
def test_unmatched_or_nonintegrable_profiles_decline(density):
    assert improper_data(density, full) is None


def test_finite_or_halfspace_domain_does_not_use_fullspace_chart():
    f = sp.exp(-(x**4) - (y - x * x) ** 2)
    assert improper_data(f, ((x, 0, sp.oo), full[1])) is None
    assert improper_data(f, ((x, -1, 1), full[1])) is None


def test_three_dimensional_triangular_chart():
    q = x * x + (y - x * x) ** 4 + (z - x * y) ** 2
    r = integrate_result(sp.exp(-q), (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo))
    assert sp.simplify(r.value - sp.pi * sp.gamma(sp.Rational(1, 4)) / 2) == 0
    assert r.complete and r.absolute_convergent


def test_mixed_profile_independent_laplace_oracle():
    q = x**4 + (y - x * x) ** 2
    r = integrate_result(sp.exp(-q) / (1 + q), *full)
    # 1/(1+q)=integral_0^infinity exp(-t*(1+q)) dt, justified by Tonelli.
    with mp.workdps(60):
        oracle = (
            mp.sqrt(mp.pi)
            * mp.gamma(mp.mpf("0.25"))
            / 2
            * mp.quad(lambda t: mp.exp(-t) / (1 + t) ** mp.mpf("0.75"), [0, 1, mp.inf])
        )
        assert abs(mp.mpf(str(r.value.evalf(55))) - oracle) < mp.mpf("1e-48")
    assert r.absolute_convergent


def test_improper_budget_declines_before_expansion():
    tiny = IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=2))
    assert improper_data(sp.exp(-(x**4) - (y - x * x) ** 2), full, policy=tiny) is None


def test_guide_examples_execute():
    p = Path(__file__).parents[1] / "docs/nonseparable-improper.md"
    blocks = re.findall(r"```python\n(.*?)```", p.read_text(), re.S)
    assert len(blocks) == 2
    for block in blocks:
        exec(compile(block, str(p), "exec"), {})


def test_completed_square_is_valid_shear():
    f = sp.exp(-(x**4) - y * y - (y - x * x) ** 2)
    r = integrate_result(f, *full)
    expected = (
        sp.sqrt(sp.pi / 2)
        * sp.gamma(sp.Rational(1, 4))
        / (2 * sp.Rational(3, 2) ** sp.Rational(1, 4))
    )
    assert sp.simplify(r.value - expected) == 0 and r.absolute_convergent


def test_released_funcprops_names_are_queried(monkeypatch):
    from types import SimpleNamespace

    from multiple_integrate.adapters import funcprops as adapter

    backend = SimpleNamespace(
        symmetry=lambda *args, **kwargs: "even",
        singularities=lambda *args, **kwargs: sp.FiniteSet(0),
    )
    monkeypatch.setattr(adapter, "_module", lambda: backend)
    assert adapter.function_symmetry(x * x, x) == "even"
    assert adapter.function_singularities(1 / x, x) == sp.FiniteSet(0)


def test_funcprops_submodule_name_is_not_called(monkeypatch):
    from types import SimpleNamespace

    from multiple_integrate.adapters import funcprops as adapter

    backend = SimpleNamespace(
        sign=SimpleNamespace(), function_sign=lambda *args, **kwargs: "positive"
    )
    monkeypatch.setattr(adapter, "_module", lambda: backend)
    assert adapter.function_sign(x * x + 1, x) == "positive"


@pytest.mark.parametrize("inner", [1 + x + y, (1 + x + y) ** 3])
def test_quadratic_halfspace_miss_never_queries_symmetry(inner, monkeypatch):
    from multiple_integrate import quadratic_gaussian as route

    monkeypatch.setattr(
        route, "_is_even_function", lambda *args: pytest.fail("unnecessary symmetry query")
    )
    assert (
        route._try_halfspace_quadratic(
            lambda v: sp.exp(-v), (inner), (x, y), ((x, 0, sp.oo), (y, 0, sp.oo)), {}
        )
        is None
    )
