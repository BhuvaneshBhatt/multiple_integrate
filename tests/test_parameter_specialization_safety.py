"""Undefined parameter fibers are distinct from divergent ordinary integrals."""

import pytest
import sympy as sp

from multiple_integrate import IntegrationPolicy, IntegrationStatus, ParameterCell, integrate_cells

x, a = sp.symbols("x a", real=True)
policy = IntegrationPolicy(max_singular_queries=0)


@pytest.mark.parametrize("density", [1 / a, sp.log(a), sp.gamma(a), sp.sin(a) / a])
def test_undefined_density_specialization_is_not_divergence(density):
    r = integrate_cells(
        density,
        (x, 0, 1),
        parameters=(a,),
        cells=[ParameterCell(sp.Eq(a, 0))],
        assumptions=sp.Eq(a, 0),
        policy=policy,
    )
    assert r.cases[0].status is IntegrationStatus.UNKNOWN
    assert not r.complete and r.ordinary_convergent is None and r.absolute_convergent is None
    assert r.cases[0].value.has(a)
    assert any(
        "no divergence theorem" in text
        for child in r.diagnostics["cell_results"]
        for text in child.limitations
    )


def test_an_explicit_removable_extension_remains_finite():
    density = sp.Piecewise((sp.sin(a) / a, sp.Ne(a, 0)), (1, True))
    r = integrate_cells(
        density,
        (x, 0, 1),
        parameters=(a,),
        cells=[ParameterCell(sp.Eq(a, 0))],
        assumptions=sp.Eq(a, 0),
        policy=policy,
    )
    assert r.complete and r.value.subs(a, 0) == 1
    assert r.ordinary_convergent is True and r.absolute_convergent is True


def test_coordinate_singularity_retains_ordinary_semantics():
    density = sp.Piecewise((1 / a, a > 0), (1 / x, True))
    r = integrate_cells(
        density,
        (x, 0, 1),
        parameters=(a,),
        cells=[ParameterCell(sp.Eq(a, 0))],
        assumptions=sp.Eq(a, 0),
        policy=policy,
    )
    assert r.cases[0].status is IntegrationStatus.DIVERGENT


def test_parameter_pole_does_not_erase_valid_sibling_cells():
    r = integrate_cells(
        1 / a,
        (x, 0, 1),
        parameters=(a,),
        cells=[ParameterCell(a < 0), ParameterCell(sp.Eq(a, 0)), ParameterCell(a > 0)],
        policy=policy,
    )
    assert r.coverage[0].complete and r.complete is False
    assert r.value.subs(a, -2) == -sp.Rational(1, 2) and r.value.subs(a, 2) == sp.Rational(1, 2)
    assert r.absolute_convergent is None
