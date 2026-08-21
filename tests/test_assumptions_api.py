import pytest
import sympy as sp

import multiple_integrate as package
from multiple_integrate import clear_cache, multiple_integrate


def test_decomposition_is_not_public_api():
    assert "Decomposition" not in package.__all__
    assert not hasattr(package, "Decomposition")


def test_relational_positive_assumption_controls_improper_exponential():
    x, a = sp.symbols("x a", real=True)
    assert multiple_integrate(sp.exp(-a * x), (x, 0, sp.oo), assumptions={a > 0}) == 1 / a


def test_single_relational_assumption_without_container_is_accepted():
    x, a = sp.symbols("x a", real=True)
    assert multiple_integrate(sp.exp(-a * x), (x, 0, sp.oo), assumptions=a > 0) == 1 / a


def test_q_predicate_assumption_is_accepted():
    x, a = sp.symbols("x a", real=True)
    assert (
        multiple_integrate(sp.exp(-a * x), (x, 0, sp.oo), assumptions={sp.Q.positive(a)}) == 1 / a
    )


def test_multiple_assumptions_are_accepted():
    x, a, b = sp.symbols("x a b", real=True)
    result = multiple_integrate(
        b * sp.exp(-a * x),
        (x, 0, sp.oo),
        assumptions={a > 0, b != 0},
    )
    assert sp.simplify(result - b / a) == 0


def test_contradictory_relational_assumptions_are_rejected():
    x, a = sp.symbols("x a", real=True)
    with pytest.raises(ValueError, match="inconsistent assumptions"):
        multiple_integrate(sp.exp(-a * x), (x, 0, sp.oo), assumptions={a > 0, a < 0})


def test_contradictory_q_predicates_are_rejected():
    x, a = sp.symbols("x a", real=True)
    with pytest.raises(ValueError, match="inconsistent assumptions"):
        multiple_integrate(
            sp.exp(-a * x),
            (x, 0, sp.oo),
            assumptions={sp.Q.positive(a), sp.Q.negative(a)},
        )


def test_invalid_non_boolean_assumption_is_rejected():
    x, a = sp.symbols("x a", real=True)
    with pytest.raises(TypeError, match="relational conditions or Q predicates"):
        multiple_integrate(x, (x, 0, 1), assumptions={a})


def test_assumptions_are_part_of_cache_identity():
    x, a = sp.symbols("x a", real=True)
    clear_cache()
    positive = multiple_integrate(sp.exp(-a * x), (x, 0, sp.oo), assumptions={a > 0})
    negative = multiple_integrate(sp.exp(-a * x), (x, 0, sp.oo), assumptions={a < 0})
    assert positive == 1 / a
    assert negative != positive


def test_assumptions_do_not_leak_between_calls():
    x, a = sp.symbols("x a", real=True)
    clear_cache()
    assert multiple_integrate(sp.exp(-a * x), (x, 0, sp.oo), assumptions={a > 0}) == 1 / a
    without = multiple_integrate(sp.exp(-a * x), (x, 0, sp.oo))
    assert without != 1 / a


def test_assumptions_feed_specialized_simplex_dirichlet_strategy():
    x, y, a = sp.symbols("x y a", real=True)
    result = multiple_integrate(
        x ** (a - 1),
        (y, 0, 1 - x),
        (x, 0, 1),
        assumptions={a > 0},
    )
    assert sp.simplify(result - 1 / (a * (a + 1))) == 0
