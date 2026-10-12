"""Bounded generated polynomial oracles; compact exact arithmetic only."""

import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from multiple_integrate import integrate


@settings(max_examples=30, deadline=None, derandomize=True)
@given(p=st.integers(0, 3), q=st.integers(0, 3), a=st.integers(1, 4), b=st.integers(1, 4))
def test_generated_box_scaling_and_fubini(p, q, a, b):
    x, y = sp.symbols("x y", real=True)
    oracle = sp.Rational(a ** (p + 1) * b ** (q + 1), (p + 1) * (q + 1))
    assert integrate(x**p * y**q, (x, 0, a), (y, 0, b)) == oracle
    assert integrate(x**p * y**q, (y, 0, b), (x, 0, a)) == oracle
