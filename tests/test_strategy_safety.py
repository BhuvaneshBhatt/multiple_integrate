"""Prerequisite and reconstruction invariants for exact strategy dispatch."""

import re
from pathlib import Path

import pytest
import semialg
import sympy as sp

from multiple_integrate import IntegrationStatus, integrate_result
from multiple_integrate.adapters import semialg as adapter
from multiple_integrate.recognition import _simplify_preserving_sums

x, a = sp.symbols("x a", real=True)


@pytest.mark.parametrize("function", [sp.sin, sp.cos])
@pytest.mark.parametrize("frequency", [-2, 2])
def test_periodic_absolute_tail_has_positive_mean(function, frequency):
    result = integrate_result(sp.Abs(function(frequency * x + 1)) / x, (x, 1, sp.oo))
    assert result.status is IntegrationStatus.DIVERGENT
    assert result.ordinary_convergent is False and result.complete


@pytest.mark.parametrize("frequency", [-2, 2])
def test_dirichlet_tail_records_conditional_convergence(frequency):
    result = integrate_result(sp.sin(frequency * x) / x, (x, 0, sp.oo))
    assert result.value == sp.sign(frequency) * sp.pi / 2
    assert result.ordinary_convergent is True and result.absolute_convergent is False


def test_zero_frequency_cannot_inherit_nonzero_tail_divergence():
    assert integrate_result(sp.Abs(sp.sin(0 * x)) / x, (x, 1, sp.oo)).value == 0
    uncertain = integrate_result(sp.Abs(sp.sin(a * x)) / x, (x, 1, sp.oo))
    assert uncertain.status is not IntegrationStatus.DIVERGENT


def test_intrinsic_coefficient_domain_failure_becomes_unknown():
    result = integrate_result(1, region=sp.Eq(x, a), variables=(x,), measure_dimension="intrinsic")
    assert result.status is IntegrationStatus.UNKNOWN
    assert result.complete is False


def test_intrinsic_adapter_declines_failed_polynomial_coercion(monkeypatch):
    def broken(*args, **kwargs):
        raise sp.CoercionFailed("symbolic coefficient outside rational domain")

    monkeypatch.setattr(adapter, "_intrinsic_parametric_cover", broken)
    assert adapter.reduce_intrinsic_domain(1, semialg.Point((0,)), variables=(x,)) is None


def test_certified_summands_survive_outer_simplification_and_substitution():
    k = sp.Dummy("k", integer=True, nonnegative=True)
    series = sp.Sum(
        (-a) ** k
        / sp.factorial(k)
        * sp.hyper((1, k + sp.Rational(3, 2)), (k + sp.Rational(5, 2),), -sp.Rational(1, 8)),
        (k, 0, sp.oo),
    )
    value = _simplify_preserving_sums((a + 1) * series / (a + 1))
    assert value == series and value.has(sp.hyper) and not value.has(sp.lerchphi)
    assert value.subs(a, 0).doit() == series.subs(a, 0).doit()


def test_readme_examples_and_known_issue_links_are_coherent():
    root = Path(__file__).resolve().parents[1]
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", (root / "README.md").read_text(), re.S):
        exec(compile(block, "README.md", "exec"), namespace)
    assert "limitations.md" in (root / "README.md").read_text()
    assert "limitations" in (root / "docs/index.rst").read_text()
    assert "Process isolation from stdin" in (root / "docs/limitations.md").read_text()


@pytest.mark.parametrize("coefficient", [2, -3, sp.I])
def test_dirichlet_shifted_tail_retains_amplitude(coefficient):
    result = integrate_result(coefficient * sp.sin(2 * x + 1) / x, (x, 1, sp.oo))
    expected = coefficient * (sp.cos(1) * (sp.pi / 2 - sp.Si(2)) - sp.sin(1) * sp.Ci(2))
    assert sp.simplify(result.value - expected) == 0
    assert result.ordinary_convergent is True and result.absolute_convergent is False


def test_local_absolute_cutoff_shortcut_matches_independent_removed_corner():
    import mpmath as mp

    from multiple_integrate import GeometricPrincipalValue

    y = sp.Symbol("y", real=True)
    formula = sp.And(x >= -2, x <= 2, y >= -2, y <= 2, x + y <= 3)
    result = integrate_result(
        x / (x * x + y * y),
        region=formula,
        variables=(x, y),
        principal_value=GeometricPrincipalValue((0, 0), exclusion="ellipsoid", axes=(2, 1)),
    )
    # The symmetric rectangle contributes zero; the removed corner is nonsingular.
    with mp.workdps(35):
        oracle = -mp.quad(lambda u: mp.quad(lambda v: u / (u * u + v * v), [3 - u, 2]), [1, 2])
        assert abs(mp.mpf(str(sp.N(result.value, 30))) - oracle) < mp.mpf("1e-25")
    assert any(c.scope == "local_cutoff_shape_independence" for c in result.coverage)


def test_local_absolute_cutoff_shortcut_requires_strict_integrability():
    from multiple_integrate.quadratic_radial import radial_region_data
    from multiple_integrate.region_types import DiskRegion

    y = sp.Symbol("y", real=True)
    disk = DiskRegion(((x, -1, 1), (y, -sp.sqrt(1 - x * x), sp.sqrt(1 - x * x))), radius=1)
    assert radial_region_data(x / (x * x + y * y), disk) is not None
    assert radial_region_data(x * x / (x * x + y * y) ** 2, disk) is None


@pytest.mark.parametrize("function", [sp.sin, sp.cos])
@pytest.mark.parametrize("limits", [(0, 1), (-1, 1)])
def test_reciprocal_phase_absolute_evidence_requires_nonzero_amplitude(function, limits):
    density = a * function(1 / x) / x
    uncertain = integrate_result(density, (x, *limits))
    nonzero = integrate_result(density, (x, *limits), assumptions=sp.Ne(a, 0))
    zero = integrate_result(density, (x, *limits), assumptions=sp.Eq(a, 0))
    assert uncertain.ordinary_convergent is True
    assert uncertain.absolute_convergent is None
    assert nonzero.ordinary_convergent is True
    assert nonzero.absolute_convergent is False
    assert zero.ordinary_convergent is True
    assert zero.absolute_convergent is True
    assert sp.simplify(uncertain.value - nonzero.value) == 0
    assert uncertain.value.subs(a, 0) == 0


def test_reciprocal_phase_away_from_singularity_allows_zero_amplitudes():
    result = integrate_result(a * sp.sin(1 / x) / x, (x, 1, 2))
    assert result.absolute_convergent is True
