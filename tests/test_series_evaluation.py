"""Exact Taylor moments and analytic error bounds against independent quadrature."""

import mpmath as mp
import pytest
import sympy as sp

from multiple_integrate.series_evaluation import evaluate_entire_box

x, y, a = sp.symbols("x y a", real=True)


@pytest.mark.parametrize("function", [sp.exp, sp.sin, sp.cos])
@pytest.mark.parametrize("coefficient", [1, 1 + sp.I / 2])
def test_certified_series_error_bounds_cover_independent_quadrature(function, coefficient):
    f = (x * x + y) * function(sp.Rational(1, 3) + coefficient * x * y)
    r = evaluate_entire_box(f, (x, 0, 1), (y, 0, 1), tolerance=sp.Rational(1, 10**16))
    assert r.converged and r.certified and r.error_bound <= sp.Rational(1, 10**16)
    assert not r.value.has(sp.Sum, sp.Integral, sp.Float)
    with mp.workdps(40):
        fn = sp.lambdify((x, y), f, "mpmath")
        oracle = mp.quad(lambda u: mp.quad(lambda v: fn(u, v), [0, 1]), [0, 1])
        approximate = mp.mpc(str(sp.re(r.value).evalf(35)), str(sp.im(r.value).evalf(35)))
        assert abs(approximate - oracle) <= mp.mpf(str(r.error_bound.evalf(35))) + mp.mpf("1e-33")


def test_series_parameter_resolution_and_coordinate_permutation():
    f = sp.exp(a * x * y)
    first = evaluate_entire_box(f, (x, -1, 1), (y, 0, 2), substitutions={a: sp.Rational(1, 2)})
    second = evaluate_entire_box(f, (y, 0, 2), (x, -1, 1), substitutions={a: sp.Rational(1, 2)})
    assert first == second


def test_term_and_sparse_expansion_budgets_preserve_honest_remainder():
    r = evaluate_entire_box(sp.exp(x * y), (x, 0, 1), (y, 0, 1), max_terms=1)
    assert r.value == 1 and r.terms == 1 and not r.converged and r.error_bound == 3
    r = evaluate_entire_box(sp.exp(x * y + x * x * y * y), (x, 0, 1), (y, 0, 1), max_monomials=1)
    assert r.terms == 1 and not r.converged and r.error_bound > 0


@pytest.mark.parametrize(
    "density",
    [
        sp.exp(a * x * y),
        sp.exp(sp.Float("0.5") * x * y),
        sp.exp(10000 * x * y),
        sp.exp(x * y) / (1 + x * x),
    ],
)
def test_series_evaluation_declines_unresolved_inexact_or_over_budget_inputs(density):
    with pytest.raises(NotImplementedError):
        evaluate_entire_box(density, (x, 0, 1), (y, 0, 1))


@pytest.mark.parametrize(
    "options",
    [
        {"max_terms": 0},
        {"max_monomials": True},
        {"tolerance": sp.Float("0.1")},
        {"substitutions": {x: 0}},
    ],
)
def test_series_validation(options):
    with pytest.raises(ValueError):
        evaluate_entire_box(sp.exp(x * y), (x, 0, 1), (y, 0, 1), **options)


def test_zero_kernel_is_exact_and_reversed_endpoints_are_rejected():
    r = evaluate_entire_box(x * x * sp.sin(sp.Rational(1, 3)), (x, 0, 1))
    # SymPy factors the constant sine out, so no nonconstant special factor is retained.
    assert r.value == sp.sin(sp.Rational(1, 3)) / 3 and r.error_bound == 0 and r.terms == 1
    with pytest.raises(ValueError):
        evaluate_entire_box(sp.exp(x * y), (x, 1, 0), (y, 0, 1))


def test_series_input_polynomial_policy_is_applied_before_expansion():
    from multiple_integrate import IntegrationPolicy, StrategyBudget

    with pytest.raises(NotImplementedError):
        evaluate_entire_box(
            sp.exp(x * y),
            (x, 0, 1),
            (y, 0, 1),
            policy=IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=1)),
        )


@pytest.mark.parametrize(
    "density,expected", [(0, 0), (x * x, sp.Rational(1, 3)), (sp.exp(0 * x * y), 1)]
)
def test_degenerate_entire_phases_retain_exact_polynomial_moments(density, expected):
    r = evaluate_entire_box(density, (x, 0, 1), (y, 0, 1))
    assert r.converged and r.value == expected and r.error_bound == 0 and r.terms == 1


def test_public_proof_portfolio_example_executes():
    import runpy
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / "examples/proof_portfolio.py"))
    assert len(namespace["examples"]()) == 5
