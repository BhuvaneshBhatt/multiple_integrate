"""Independent geometry, convergence and cutoff references for the new families."""

from dataclasses import replace

import mpmath as mp
import pytest
import semialg
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
from multiple_integrate.assumptions import _normalize_assumptions
from multiple_integrate.cell_proofs import interval_nonnegative
from multiple_integrate.cusp_horns import try_cusp_horn
from multiple_integrate.policy import IntegrationPolicy
from multiple_integrate.quadratic_radial import _moment
from multiple_integrate.singular_maps import polygon_product_cutoff
from multiple_integrate.symbolic_geometry import _compare, integrate_symbolic_boxes

x, y, a, b = sp.symbols("x y a b", real=True)


@pytest.mark.parametrize("lo,hi,expr", [(0, 1, 1 - x * x), (0, a, b - x * x), (-a, a, b - x * x)])
def test_bernstein_order_certificate(lo, hi, expr):
    context = sp.And(a > 0, b > a * a)
    assert interval_nonnegative(expr, x, sp.sympify(lo), sp.sympify(hi), context)


def test_positive_interior_can_have_zero_endpoints():
    assert interval_nonnegative(x * (1 - x), x, sp.S.Zero, sp.S.One, strict_interior=True)
    assert not interval_nonnegative(sp.S.Zero, x, sp.S.Zero, sp.S.One, strict_interior=True)
    assert not interval_nonnegative(x - sp.Rational(1, 2), x, sp.S.Zero, sp.S.One)


@pytest.mark.parametrize(
    "left,right,condition,expected",
    [
        (a, b, a < b, -1),
        (b, a * a, b > a * a, 1),
        (sp.sqrt(b), a, sp.And(a > 0, b > a * a), 1),
        (a * a, b, sp.Eq(b, a * a), 0),
    ],
)
def test_parameter_endpoint_orders(left, right, condition, expected):
    assert _compare(left, right, _normalize_assumptions(condition)) == expected


def test_parameter_graph_strip_matches_symbolic_reference():
    formula = sp.And(x >= 0, x <= a, y >= 0, y <= b - x * x)
    r = integrate_cells(
        x + y,
        region=formula,
        variables=(x, y),
        parameters=(a, b),
        assumptions=sp.And(a > 0, b > a * a),
        cells=(ParameterCell(sp.And(a > 0, b > a * a)),),
    )
    expected = sp.integrate(x + y, (y, 0, b - x * x), (x, 0, a))
    assert sp.simplify(r.diagnostics["cell_results"][0].value - expected) == 0
    assert r.complete and r.exact
    assert r.diagnostics["cell_results"][0].absolute_convergent is True


def test_parameter_graph_boundary_coincidence_and_permutation():
    formula = sp.And(y >= 0, y <= a, x >= y * y, x <= b)
    r = integrate_symbolic_boxes(1, formula, (x, y), assumptions=sp.And(a > 0, sp.Eq(b, a * a)))
    assert sp.simplify(r.value.subs(b, a**2) - 2 * a**3 / 3) == 0
    assert r.absolute_convergent is True


def test_graph_bound_cannot_choose_a_sample_cell():
    assert (
        integrate_symbolic_boxes(
            1, sp.And(x >= 0, x <= a, y >= 0, y <= b - x * x), (x, y), assumptions=a > 0
        )
        is None
    )


def test_graph_envelope_redundancy_is_certified_globally():
    r = integrate_symbolic_boxes(
        1, sp.And(x >= 0, x <= 1, y >= 0, y <= 2 - x * x, y <= 3 - x), (x, y)
    )
    assert r.value == sp.Rational(5, 3)


@pytest.mark.parametrize(
    "power,expected",
    [
        (sp.Rational(1, 2), sp.pi),
        (sp.Rational(2, 3), sp.beta(sp.Rational(1, 2), sp.Rational(1, 3)) / sp.Rational(1, 2)),
    ],
)
def test_cusp_boundary_contact_finite(power, expected):
    r = integrate_result(
        (x**3 - y * y) ** -power, region=sp.And(x > 0, x < 1, y * y <= x**3), variables=(x, y)
    )
    assert abs(complex(r.value.evalf(30)) - complex(expected.evalf(30))) < 1e-25
    assert r.absolute_convergent is True


@pytest.mark.parametrize(
    "density", [1 / (x**3 - y * y), x * x / (x**3 - y * y), -x * x / (x**3 - y * y) ** 2]
)
def test_cusp_boundary_pole_diverges_even_if_vertex_is_integrable(density):
    r = integrate_result(density, region=sp.And(x > 0, x < 1, y * y <= x**3), variables=(x, y))
    assert r.outcome is IntegrationOutcome.DIVERGENT
    assert r.absolute_convergent is False


@pytest.mark.parametrize("order", [1, 2, 3])
def test_expanded_numerator_vanishing_removes_boundary_poles(order):
    base = x**3 - y * y
    density = sp.expand(base**order) / base ** (order + sp.Rational(1, 2))
    r = integrate_result(
        density,
        region=sp.And(x > 0, x < 1, y * y <= x**3),
        variables=(x, y),
        policy=IntegrationPolicy(allow_expensive_exact=True),
    )
    assert abs(complex(r.value.evalf(30)) - complex(sp.pi.evalf(30))) < 1e-25
    assert r.absolute_convergent is True


def test_cusp_odd_boundary_pole_never_cancels_before_absolute_proof():
    assert try_cusp_horn(y / (x**3 - y * y), sp.And(x > 0, x < 1, y * y <= x**3), (x, y)) is None


def test_cusp_generic_contact_parameter_needs_uniform_angular_proof():
    k = sp.Symbol("k", real=True)
    good = try_cusp_horn(
        (x**3 - y * y) ** -sp.Rational(1, 2),
        sp.And(x > 0, x < 1, y * y <= k * x**3),
        (x, y),
        sp.And(k > 0, k <= 1),
    )
    bad = try_cusp_horn(
        x * x / (x**3 - y * y),
        sp.And(x > 0, x < 1, y * y <= k * x**3),
        (x, y),
        sp.And(k > 0, k <= 1),
    )
    assert good is not None and bad is None


@pytest.mark.parametrize(
    "alpha,inner,outer,power",
    [
        (sp.Rational(1, 2), 0, 1, 1),
        (sp.Rational(3, 2), 0, 1, 2),
        (sp.Rational(5, 2), sp.Rational(1, 2), 1, 1),
        (sp.Rational(1, 2), 0, 2, sp.Rational(3, 2)),
    ],
)
def test_half_integer_series_matches_independent_quadrature(alpha, inner, outer, power):
    moment = _moment(
        alpha, sp.S.One, (sp.Integer(2), -sp.sympify(power)), sp.sympify(inner), sp.sympify(outer)
    )
    assert moment.has(sp.Sum)
    with mp.workdps(35):
        reference = mp.quad(
            lambda r: r ** (2 * float(alpha) - 1) * mp.exp(-r * r) / (2 + r * r) ** float(power),
            [float(inner), float(outer)],
        )
        assert abs(complex(moment.evalf(30)) - complex(reference)) < 1e-24


def test_three_dimensional_mixed_radial_family_is_native_and_finite():
    z = sp.Symbol("z", real=True)
    rho = x * x + y * y + z * z
    result = integrate_result(
        sp.exp(-rho) / (2 + rho), region=semialg.Ball((0, 0, 0), 1), variables=(x, y, z)
    )
    assert result.value.has(sp.Sum) and result.absolute_convergent is True
    with mp.workdps(35):
        expected = 4 * mp.pi * mp.quad(lambda r: r * r * mp.exp(-r * r) / (2 + r * r), [0, 1])
        assert abs(complex(result.value.evalf(30)) - complex(expected)) < 1e-24


def test_shifted_radial_tail_respects_lower_radius():
    assert _moment(sp.S.One, None, (sp.S.One, -sp.Integer(2)), sp.S.One, sp.oo) == sp.Rational(1, 4)


@pytest.mark.parametrize("p,q", [(-1, -1), (-2, -2), (0, 0), (1, -1), (-1, 2)])
def test_trapezoid_shared_cutoff_matches_numeric_integral(p, q):
    eps = sp.Symbol("eps", positive=True)
    region = sp.And(x >= 0, y >= 0, x <= sp.Rational(1, 2), x + y <= 1)
    result = polygon_product_cutoff(x**p * y**q, region, (x, y), x * y, eps)
    assert result is not None
    with mp.workdps(35):
        e = mp.mpf(".01")
        low = (1 - mp.sqrt(1 - 4 * e)) / 2
        reference = mp.quad(
            lambda u: mp.quad(lambda v: u**p * v**q, [e / u, 1 - u]), [low, mp.mpf(".5")]
        )
        assert (
            abs(complex(result[0].subs(eps, sp.Rational(1, 100)).evalf(30)) - complex(reference))
            < 1e-24
        )


def test_clipped_trapezoid_finite_part_has_independent_dilog_oracle():
    region = sp.And(x >= 0, y >= 0, x <= sp.Rational(1, 2), x + y <= 1)
    r = regularized_result(
        1 / (x * y),
        region=region,
        variables=(x, y),
        regularization=HadamardFinitePart(hypersurface=x * y),
    )
    assert sp.simplify(r.value + sp.pi**2 / 12 - sp.log(2) ** 2) == 0
    assert verify_regularization_result(r).valid
    assert not verify_regularization_result(replace(r, value=r.value + 1)).valid


def test_multiple_sloping_constraints_create_a_polygon_envelope():
    e = sp.Symbol("eps", positive=True)
    formula = sp.And(x >= 0, y >= 0, x <= 1, y <= 1, x + y <= sp.Rational(3, 2))
    result = polygon_product_cutoff(1, formula, (x, y), x * y, e)
    assert result is not None and len(result[1]["polygon_segments"]) == 2
    assert sp.limit(result[0], e, 0, dir="+") == sp.Rational(7, 8)


def test_graph_to_clipped_polygon_preserves_shared_cutoff():
    g = x * (y - x * x)
    formula = sp.And(x >= 0, x <= sp.Rational(1, 2), y - x * x >= 0, x + y - x * x <= 1)
    result = regularized_result(
        1 / g, region=formula, variables=(x, y), regularization=HadamardFinitePart(hypersurface=g)
    )
    assert sp.simplify(result.value + sp.pi**2 / 12 - sp.log(2) ** 2) == 0
    assert result.provenance.details["realization"] == "polynomial_graph_cutoff"
    assert verify_regularization_result(result).valid


def test_disconnected_graph_domain_has_a_complete_disjoint_mapped_cover():
    g = x * (y - x * x)
    formula = sp.And(
        sp.Or(sp.And(x >= -2, x <= -1), sp.And(x >= 1, x <= 2)), y - x * x >= -1, y - x * x <= 1
    )
    result = regularized_result(
        1 / g**2,
        region=formula,
        variables=(x, y),
        regularization=HadamardFinitePart(hypersurface=g),
    )
    assert result.value == -2
    assert result.provenance.details["mapped_box_pieces"] == 2
    assert verify_regularization_result(result).valid


def test_polygon_with_unproved_or_non_downward_geometry_declines():
    e = sp.Symbol("eps", positive=True)
    assert (
        polygon_product_cutoff(
            1, sp.And(x >= 0, y >= 0, x + y >= 1, x <= 2, y <= 2), (x, y), x * y, e
        )
        is None
    )


def test_geometry_extension_documentation_traceability():
    from pathlib import Path

    text = (Path(__file__).parents[1] / "docs/geometry-extensions.md").read_text()
    for reference in ("test_geometry_extensions.py", "Bernstein", "s<1", "Sum", "cutoff window"):
        assert reference in text


def test_cusp_weak_pole_threshold_does_not_erase_divergent_fiber():
    s = sp.Symbol("s", real=True)
    assert (
        try_cusp_horn(
            x**4 / (x**3 - y * y) ** s,
            sp.And(x > 0, x < 1, y * y <= x**3),
            (x, y),
            sp.And(s > 0, s <= 1),
        )
        is None
    )


def test_bernstein_weak_bound_is_not_strict_positivity():
    from multiple_integrate.cell_proofs import nonnegative

    k = sp.Symbol("k", real=True)
    assert not nonnegative(1 - k, _normalize_assumptions(sp.And(k > 0, k <= 1)), strict=True)


def test_clipped_cutoff_scale_and_signed_orthant_are_preserved():
    e = sp.Symbol("eps", positive=True)
    positive = sp.And(x >= 0, y >= 0, x <= sp.Rational(1, 2), x + y <= 1)
    negative = sp.And(x <= 0, y >= 0, -x <= sp.Rational(1, 2), -x + y <= 1)
    reference = polygon_product_cutoff(1 / (x * y), positive, (x, y), x * y, e)
    scaled = polygon_product_cutoff(1 / (3 * x * y), positive, (x, y), 3 * x * y, e)
    reflected = polygon_product_cutoff(1 / (x * y), negative, (x, y), x * y, e)
    assert reference is not None and scaled is not None and reflected is not None
    assert sp.simplify(3 * scaled[0].subs(e, 3 * e) - reference[0]) == 0
    assert sp.simplify(reflected[0] + reference[0]) == 0


def test_weak_graph_slope_cannot_divide_by_zero_on_the_cell():
    from multiple_integrate.cell_graphs import integrate_cell_graph

    assert (
        integrate_cell_graph(
            1, sp.And(x >= 0, x <= 1, y >= 0, a * y <= 1 - x * x), (x, y), assumptions=a >= 0
        )
        is None
    )


def test_explicit_predicate_cell_proofs_remain_supported():
    from multiple_integrate.cell_proofs import nonnegative

    context = _normalize_assumptions(sp.Q.positive(a))
    assert nonnegative(a, context, strict=True)
    assert nonnegative(2 * a, context, strict=True)
