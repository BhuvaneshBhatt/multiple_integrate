"""Independent shared-cutoff oracles on asymmetric and affine box domains."""

from dataclasses import replace

import pytest
import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    HypersurfacePrincipalValue,
    IntegrationOutcome,
    IntegrationPolicy,
    StrategyBudget,
    regularized_result,
    verify_regularization_result,
)
from multiple_integrate.region_types import BoxRegion
from multiple_integrate.singular_geometry import product_level_cutoff

x, y, a = sp.symbols("x y a", real=True)


@pytest.mark.parametrize("left,right,bottom,top", [(1, 2, 1, 3), (2, 3, 4, 5), (1, 1, 1, 1)])
def test_asymmetric_inverse_square_finite_part(left, right, bottom, top):
    r = regularized_result(
        1 / (x * y) ** 2,
        (x, -left, right),
        (y, -bottom, top),
        regularization=HadamardFinitePart(hypersurface=x * y),
    )
    expected = (sp.Rational(1, left) + sp.Rational(1, right)) * (
        sp.Rational(1, bottom) + sp.Rational(1, top)
    )
    assert r.value == expected and not r.value.atoms(sp.Float)
    assert verify_regularization_result(r).valid
    e = r.provenance.cutoff_parameter
    oracle = (
        expected + (sum(sp.log(w * h / e) - 1 for w in (left, right) for h in (bottom, top))) / e
    )
    assert sp.simplify(sp.expand_log(r.provenance.cutoff_integral - oracle, force=True)) == 0


def test_asymmetric_pv_retains_nonzero_signed_orthant_sum():
    r = regularized_result(
        1 / (x * y), (x, -1, 2), (y, -1, 3), principal_value=HypersurfacePrincipalValue(x * y)
    )
    assert r.value == sp.log(2) * sp.log(3)
    assert r.outcome is IntegrationOutcome.FINITE and r.provenance.details["orthants"] == 4


def test_boundary_planes_share_one_cutoff_instead_of_iterated_finite_parts():
    r = regularized_result(
        1 / (x * y), (x, 0, 2), (y, 0, 3), regularization=HadamardFinitePart(hypersurface=x * y)
    )
    assert sp.simplify(sp.expand(r.value - (sp.log(2) + sp.log(3)) ** 2 / 2)) == 0
    assert sp.simplify(r.value - sp.log(2) * sp.log(3)) != 0
    pv = regularized_result(
        1 / (x * y), (x, 0, 2), (y, 0, 3), principal_value=HypersurfacePrincipalValue(x * y)
    )
    assert pv.outcome is IntegrationOutcome.DIVERGENT


def test_defining_equation_scale_remains_part_of_finite_part_prescription():
    r = regularized_result(
        1 / (x * y), (x, 0, 1), (y, 0, 1), regularization=HadamardFinitePart(hypersurface=2 * x * y)
    )
    assert r.value == sp.log(2) ** 2 / 2


def test_planes_outside_box_use_inclusion_exclusion():
    r = regularized_result(
        1 / (x * y), (x, 1, 2), (y, 2, 3), principal_value=HypersurfacePrincipalValue(x * y)
    )
    assert sp.simplify(r.value - sp.log(2) * sp.log(sp.Rational(3, 2))) == 0
    assert r.provenance.backend == "dominated_convergence"
    cutoff = product_level_cutoff(
        1 / (x * y), BoxRegion(((x, 1, 2), (y, 2, 3))), x * y, sp.Symbol("eps", positive=True)
    )
    assert cutoff[1]["moment_pieces"] == 4


def test_off_center_repeated_plane():
    r = regularized_result(
        1 / (x - 1) ** 2, (x, 0, 3), regularization=HadamardFinitePart(hypersurface=(x - 1) ** 2)
    )
    assert r.value == -sp.Rational(3, 2)


def test_affine_intersection_jacobian_and_original_cutoff_domain():
    u, v = x + y, x - y
    formula = sp.And(u >= -1, u <= 2, v >= -1, v <= 3)
    r = regularized_result(
        1 / (u * v) ** 2,
        region=formula,
        variables=(x, y),
        regularization=HadamardFinitePart(hypersurface=u * v),
    )
    assert r.value == 1
    assert r.provenance.details["realization"] == "affine_product_cutoff"
    assert r.provenance.details["affine_jacobian"] == sp.Rational(1, 2)
    assert verify_regularization_result(r).valid
    bad_details = dict(r.provenance.details, affine_jacobian=sp.Integer(1))
    bad = replace(r, provenance=replace(r.provenance, details=bad_details))
    assert not verify_regularization_result(bad).valid
    assert r.provenance.cutoff_domain.has(x, y)
    assert not verify_regularization_result(replace(r, value=r.value + 1)).valid


def test_parameterized_shear_and_coordinate_permutation():
    u = x + a * y
    formula = sp.And(u >= -1, u <= 1, y >= -1, y <= 1)
    r = regularized_result(
        1 / (u * y) ** 2,
        region=formula,
        variables=(x, y),
        regularization=HadamardFinitePart(hypersurface=u * y),
    )
    assert r.value == 4 and r.provenance.details["affine_jacobian"] == 1
    second = regularized_result(
        1 / (u * y) ** 2,
        region=formula,
        variables=(y, x),
        regularization=HadamardFinitePart(hypersurface=u * y),
    )
    assert second.value == r.value


@pytest.mark.parametrize("g", [(x + y) * (2 * x + 2 * y), x * (x - 1)])
def test_dependent_or_parallel_planes_are_not_a_certified_affine_product(g):
    e = sp.Symbol("e", positive=True)
    # The first reduces to one repeated affine plane, whose box image is not rectangular.
    assert product_level_cutoff(1 / g**2, BoxRegion(((x, -1, 1), (y, -1, 1))), g, e) is None


def test_unexcluded_density_pole_declines_native_product_route():
    e = sp.Symbol("e", positive=True)
    assert (
        product_level_cutoff(1 / (x * y * (x - y)), BoxRegion(((x, -1, 2), (y, -1, 3))), x * y, e)
        is None
    )


def test_density_and_orthant_budget_is_not_bypassed():
    r = regularized_result(
        (1 + x + y) / (x * y) ** 2,
        (x, -1, 2),
        (y, -1, 2),
        regularization=HadamardFinitePart(hypersurface=x * y),
        policy=IntegrationPolicy(hypersurface=StrategyBudget(max_pieces=1)),
    )
    assert r.outcome is IntegrationOutcome.UNKNOWN and r.limitations
