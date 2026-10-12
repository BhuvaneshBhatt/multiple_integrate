import pytest
import sympy as sp

from multiple_integrate import (
    HypersurfacePrincipalValue,
    IntegrationPolicy,
    StrategyBudget,
    StrategyBudgetExceeded,
    integrate,
)
from multiple_integrate.hypersurface_atlas import build_hypersurface_atlas
from multiple_integrate.policy import check_semialg_budget
from multiple_integrate.region_types import BoxRegion
from multiple_integrate.regularization import finite_part_from_cutoff


def test_semialg_variable_boundary():
    x, y = sp.symbols("x y", real=True)
    formula = sp.And(x >= 0, x <= 1, y >= 0, y <= 1)
    policy = IntegrationPolicy(semialg=StrategyBudget(max_variables=2))
    assert check_semialg_budget(formula, (x, y), policy) == (True, None)
    policy = IntegrationPolicy(semialg=StrategyBudget(max_variables=1))
    allowed, reason = check_semialg_budget(formula, (x, y), policy)
    assert not allowed and "variable count 2 exceeds 1" in reason


def test_semialg_degree_boundary():
    x = sp.Symbol("x", real=True)
    formula = x**4 <= 1
    policy = IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=4))
    assert check_semialg_budget(formula, (x,), policy) == (True, None)
    policy = IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=3))
    allowed, reason = check_semialg_budget(formula, (x,), policy)
    assert not allowed and "polynomial degree 4 exceeds 3" in reason


def test_semialg_boolean_atom_boundary():
    x = sp.Symbol("x", real=True)
    formula = sp.And(x >= -1, x <= 1, sp.Ne(x, 0), sp.Ne(x, sp.Rational(1, 2)))
    policy = IntegrationPolicy(semialg=StrategyBudget(max_boolean_atoms=4))
    assert check_semialg_budget(formula, (x,), policy) == (True, None)
    policy = IntegrationPolicy(semialg=StrategyBudget(max_boolean_atoms=3))
    allowed, reason = check_semialg_budget(formula, (x,), policy)
    assert not allowed and "Boolean atom count 4 exceeds 3" in reason


def test_semialg_piece_boundary():
    x = sp.Symbol("x", real=True)
    formula = sp.Or(sp.And(x >= 0, x <= 1), sp.And(x >= 2, x <= 3))
    policy = IntegrationPolicy(semialg=StrategyBudget(max_pieces=2))
    assert integrate(1, region=formula, variables=(x,), policy=policy) == 2
    policy = IntegrationPolicy(semialg=StrategyBudget(max_pieces=1))
    with pytest.raises(StrategyBudgetExceeded, match="piece count"):
        integrate(1, region=formula, variables=(x,), policy=policy)


def test_hypersurface_chart_boundary():
    x, y = sp.symbols("x y", real=True)
    formula = x**2 + y**2 <= 4
    permissive = IntegrationPolicy(hypersurface=StrategyBudget(max_algebraic_branches=4))
    restrictive = IntegrationPolicy(hypersurface=StrategyBudget(max_algebraic_branches=3))
    from multiple_integrate.policy import _CURRENT_POLICY

    token = _CURRENT_POLICY.set(permissive)
    try:
        atlas = build_hypersurface_atlas(x**2 + y**2 - 1, formula, (x, y))
        assert len(atlas.charts) == 4
    finally:
        _CURRENT_POLICY.reset(token)
    token = _CURRENT_POLICY.set(restrictive)
    try:
        with pytest.raises(StrategyBudgetExceeded, match="chart count 4 exceeds 3"):
            build_hypersurface_atlas(x**2 + y**2 - 1, formula, (x, y))
    finally:
        _CURRENT_POLICY.reset(token)


def test_asymptotic_parameter_boundary():
    epsilon = sp.Symbol("epsilon", positive=True)
    a, b = sp.symbols("a b")
    from multiple_integrate.policy import _CURRENT_POLICY

    policy = IntegrationPolicy(asymptotic=StrategyBudget(max_variables=2))
    token = _CURRENT_POLICY.set(policy)
    try:
        value, _ = finite_part_from_cutoff(1 / epsilon + a + b, epsilon)
        assert value == a + b
    finally:
        _CURRENT_POLICY.reset(token)
    policy = IntegrationPolicy(asymptotic=StrategyBudget(max_variables=1))
    token = _CURRENT_POLICY.set(policy)
    try:
        with pytest.raises(StrategyBudgetExceeded, match="parameter count 2 exceeds 1"):
            finite_part_from_cutoff(1 / epsilon + a + b, epsilon)
    finally:
        _CURRENT_POLICY.reset(token)


def test_asymptotic_degree_boundary():
    epsilon = sp.Symbol("epsilon", positive=True)
    from multiple_integrate.policy import _CURRENT_POLICY

    policy = IntegrationPolicy(asymptotic=StrategyBudget(max_polynomial_degree=5))
    token = _CURRENT_POLICY.set(policy)
    try:
        value, _ = finite_part_from_cutoff(epsilon**4 + 1 / epsilon, epsilon)
        assert value == 0
    finally:
        _CURRENT_POLICY.reset(token)
    policy = IntegrationPolicy(asymptotic=StrategyBudget(max_polynomial_degree=4))
    token = _CURRENT_POLICY.set(policy)
    try:
        with pytest.raises(StrategyBudgetExceeded, match="cutoff degree 5 exceeds 4"):
            finite_part_from_cutoff(epsilon**4 + 1 / epsilon, epsilon)
    finally:
        _CURRENT_POLICY.reset(token)


def test_expensive_exact_bypasses_guard():
    x, y = sp.symbols("x y", real=True)
    box = BoxRegion(((x, -1, 1), (y, -1, 1)))
    policy = IntegrationPolicy(
        hypersurface=StrategyBudget(max_variables=0, max_polynomial_degree=0),
        allow_expensive_exact=True,
    )
    value = integrate(
        1 / x,
        region=box,
        principal_value=HypersurfacePrincipalValue(x),
        policy=policy,
    )
    assert value == 0
