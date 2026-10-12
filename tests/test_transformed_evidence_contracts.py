"""Value identities must preserve scoped existence and coverage evidence."""

from dataclasses import replace

import pytest
import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    IntegrationOutcome,
    ParameterCell,
    integrate_cells,
    integrate_result,
    regularized_result,
    verify_regularization_result,
)

x, a = sp.symbols("x a", real=True)


@pytest.mark.parametrize("center,scale", [(-2, 1), (0, 3), (4, sp.Rational(1, 2))])
def test_translation_scaling_preserves_absolute_and_coverage(center, scale):
    base = integrate_result(sp.exp(-(x**4)), (x, 0, sp.oo))
    transformed = integrate_result(
        sp.exp(-(((x - center) / scale) ** 4)) / scale, (x, center, sp.oo)
    )
    assert sp.simplify(transformed.value - base.value) == 0
    for result in (base, transformed):
        assert result.outcome is IntegrationOutcome.FINITE
        assert result.ordinary_convergent is True and result.absolute_convergent is True
        assert result.complete and any(c.complete for c in result.coverage)


@pytest.mark.parametrize("split", [1, 2, sp.Rational(1, 2)])
def test_improper_decomposition_preserves_value_and_absolute_proof(split):
    base = integrate_result(sp.exp(-(x**4)), (x, 0, sp.oo))
    parts = [
        integrate_result(sp.exp(-(x**4)), (x, lo, hi)) for lo, hi in [(0, split), (split, sp.oo)]
    ]
    assert sp.simplify(sum(r.value for r in parts) - base.value) == 0
    assert all(r.absolute_convergent is True and r.complete for r in parts)


@pytest.mark.parametrize("fiber", [-1, 0, 1, 2])
def test_parameter_specialization_keeps_case_outcome(fiber):
    result = integrate_cells(
        sp.exp(-a * x),
        (x, 0, sp.oo),
        parameters=(a,),
        cells=(ParameterCell(a < 0), ParameterCell(sp.Eq(a, 0), {a: 0}), ParameterCell(a > 0)),
    )
    active = [c for c in result.cases if c.condition.subs(a, fiber) is sp.true]
    assert len(active) == 1
    direct = integrate_result(sp.exp(-fiber * x), (x, 0, sp.oo))
    assert active[0].outcome is direct.outcome
    if fiber > 0:
        assert result.value.subs(a, fiber) == sp.Rational(1, fiber)
    else:
        assert active[0].outcome is IntegrationOutcome.DIVERGENT


@pytest.mark.parametrize("missing", [0, 1])
def test_missing_parameter_cells_prevent_global_evidence(missing):
    cells = (ParameterCell(a < 0), ParameterCell(sp.Eq(a, 0), {a: 0}), ParameterCell(a > 0))
    result = integrate_cells(
        x**2,
        (x, 0, 1),
        parameters=(a,),
        cells=tuple(c for i, c in enumerate(cells) if i != missing),
    )
    assert not result.coverage[0].complete and not result.complete
    assert result.absolute_convergent is None


@pytest.mark.parametrize("corruption", ["expansion", "divergent_terms", "cutoff"])
def test_correct_value_does_not_rescue_corrupted_regularization_proof(corruption):
    result = regularized_result(x**-2, (x, -1, 1), regularization=HadamardFinitePart())
    p = result.provenance
    changed = {
        "expansion": {"asymptotic_expansion": p.asymptotic_expansion + 1},
        "divergent_terms": {"divergent_terms": ()},
        "cutoff": {"cutoff_integral": p.cutoff_integral + 1},
    }[corruption]
    corrupted = replace(result, provenance=replace(p, **changed))
    assert corrupted.value == result.value
    assert not verify_regularization_result(corrupted).valid
