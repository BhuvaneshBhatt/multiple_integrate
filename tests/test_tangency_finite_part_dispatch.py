"""Regulator-preserving tangent/contact routes must win before generic charts.

These unit tests use raw SymPy Boolean domains, so backend domain lowering
is replaced by its exact identity operation. The production route's
integration, asymptotic extraction and chart metadata remain unmodified.
"""

import pytest
import sympy as sp

from multiple_integrate import HadamardFinitePart, regularized_result, verify_regularization_result
from multiple_integrate.regularization import finite_part_from_cutoff
from multiple_integrate.singular_geometry import product_level_cutoff

x, y = sp.symbols("x y", real=True)


@pytest.fixture
def isolated_formula_dispatch(monkeypatch):
    import multiple_integrate.adapters.semialg as adapter
    import multiple_integrate.contact_charts as contact
    import multiple_integrate.convergence_evidence as convergence
    import multiple_integrate.hypersurface_principal_value as hypersurface
    import multiple_integrate.singular_geometry as singular
    import multiple_integrate.singular_maps as maps
    import multiple_integrate.tangent_cutoff as tangent

    def identity_domain(domain, variables):
        return sp.sympify(domain), tuple(variables)

    for module in (adapter, contact, hypersurface, tangent):
        monkeypatch.setattr(module, "semialgebraic_domain", identity_domain)

    def unexpected(*args, **kwargs):
        pytest.fail("generic affine/graph recognizer preempted an exact contact chart")

    monkeypatch.setattr(singular, "_affine_product_cutoff", unexpected)
    monkeypatch.setattr(maps, "graph_product_cutoff", unexpected)
    # Convergence-evidence discovery is not under test; avoid querying the
    # external semialg backend after a finite-part evaluation succeeded.
    monkeypatch.setattr(convergence, "region_absolute_rule", lambda *a, **k: False)


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("variable_order", [(x, y), (y, x)])
def test_canonical_tangent_precedes_generic_and_preserves_cutoff(
    isolated_formula_dispatch, order, variable_order
):
    g = y * (y - x**order)
    region = sp.And(x > 0, x < 1, y > 0, y < x**order)
    result = regularized_result(
        x ** (order - 1) / g,
        region=region,
        variables=variable_order,
        regularization=HadamardFinitePart(hypersurface=g),
    )
    assert sp.simplify(result.value - sp.pi**2 / (6 * order)) == 0
    assert not result.value.has(sp.Integral, sp.nan)
    assert result.provenance.details["realization"] == "tangent_graph_cutoff"
    assert result.provenance.details["coverage_complete"] is True
    assert verify_regularization_result(result).valid


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("scale", [1, 2, -3])
def test_tangent_preserves_original_regulator_scale(isolated_formula_dispatch, order, scale):
    g = y * (y - x**order)
    region = sp.And(x > 0, x < 1, y > 0, y < x**order)
    result = regularized_result(
        x ** (3 * order - 1) / g,
        region=region,
        variables=(x, y),
        regularization=HadamardFinitePart(hypersurface=scale * g),
    )
    assert sp.simplify(result.value - (1 - sp.log(abs(scale))) / order) == 0
    assert result.provenance.details["regulator_scale"] == abs(scale)
    assert verify_regularization_result(result).valid


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("variable_order", [(x, y), (y, x)])
def test_translated_polynomial_contact_precedes_generic(
    isolated_formula_dispatch, order, variable_order
):
    graph = x**3 + x
    gap = (x - 1) ** order
    g = (y - graph) * (y - graph - gap)
    region = sp.And(x > 1, x < 2, y > graph, y < graph + gap)
    result = regularized_result(
        (x - 1) ** (order - 1) / g,
        region=region,
        variables=variable_order,
        regularization=HadamardFinitePart(hypersurface=g),
    )
    assert sp.simplify(result.value - sp.pi**2 / (6 * order)) == 0
    assert result.provenance.details["realization"] == "polynomial_contact_cutoff"
    details = result.provenance.details
    source_to_target = dict(zip(details["source_variables"], details["forward_map"], strict=True))
    assert (
        sp.simplify(g.subs(source_to_target, simultaneous=True) - details["transformed_level"]) == 0
    )
    assert details["coverage_complete"] is True
    assert verify_regularization_result(result).valid


def test_unmatched_cutoff_still_uses_generic_fallback(monkeypatch):
    import multiple_integrate.adapters.semialg as adapter
    import multiple_integrate.contact_charts as contact
    import multiple_integrate.singular_geometry as singular
    import multiple_integrate.tangent_cutoff as tangent

    for module in (adapter, contact, tangent):
        monkeypatch.setattr(
            module, "semialgebraic_domain", lambda domain, variables: (domain, tuple(variables))
        )
    sentinel = (sp.Symbol("generic_result"), {"realization": "generic_affine"})
    monkeypatch.setattr(singular, "_affine_product_cutoff", lambda *a, **k: sentinel)
    region = sp.And(x > 0, x < 1, y > 0, y < 1)
    result = product_level_cutoff(
        1 / (x * y), region, x * y, sp.Symbol("e", positive=True), variables=(x, y)
    )
    assert result == sentinel


@pytest.mark.parametrize("residual", ["sympy_integral", "region_integral"])
def test_unresolved_cutoff_cannot_be_certified_as_a_finite_part(residual):
    from multiple_integrate.results import UnevaluatedRegionIntegral, UnresolvedIntegration

    epsilon = sp.Symbol("epsilon", positive=True)
    if residual == "sympy_integral":
        remainder = sp.Integral(1 / x, (x, 0, 1))
    else:
        remainder = UnevaluatedRegionIntegral(1 / (x * y), sp.And(x > 0, y > 0), (x, y))
    with pytest.raises(UnresolvedIntegration, match="shared-cutoff integral is unevaluated"):
        finite_part_from_cutoff(1 / epsilon + remainder, epsilon)
