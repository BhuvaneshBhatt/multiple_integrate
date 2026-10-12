"""Generated, metamorphic and independent numerical/asymptotic geometry oracles."""

import mpmath as mp
import pytest
import semialg
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from multiple_integrate import (
    HadamardFinitePart,
    IntegrationOutcome,
    IntegrationPolicy,
    integrate_result,
    regularized_result,
)
from multiple_integrate.cusp_horns import try_cusp_horn
from multiple_integrate.quadratic_radial import _moment, radial_region_data
from multiple_integrate.singular_maps import polygon_product_cutoff

x, y, a, b = sp.symbols("x y a b", real=True)


@settings(max_examples=20, deadline=None, derandomize=True)
@given(
    tx=st.integers(-2, 2),
    ty=st.integers(-2, 2),
    sx=st.integers(1, 3),
    sy=st.integers(1, 3),
    permuted=st.booleans(),
)
def test_generated_parameter_graph_transformations(tx, ty, sx, sy, permuted):
    u = (x - tx) / sx
    v = (y - ty) / sy
    formula = sp.And(u >= 0, 2 * u <= 2 * a, v >= 0, 3 * v <= 3 * (b - u * u), v <= b - u * u + 1)
    variables = (y, x) if permuted else (x, y)
    from multiple_integrate.cell_graphs import integrate_cell_graph

    child = integrate_cell_graph(
        u + 2 * v,
        formula,
        variables,
        assumptions=sp.And(a > 0, b > a * a),
    )
    assert child is not None
    U, V = sp.symbols("U V", real=True)
    reference = sx * sy * sp.integrate(U + 2 * V, (V, 0, b - U * U), (U, 0, a))
    assert sp.simplify(child.value - reference) == 0
    assert child.absolute_convergent is True and child.complete


@pytest.mark.parametrize("p,q", [(0, 0), (2, 0), (0, 2)])
@pytest.mark.parametrize("delta", [sp.Rational(-1, 12), sp.S.Zero, sp.Rational(1, 12)])
def test_cusp_vertex_threshold_neighbors(p, q, delta):
    power = (sp.Rational(5, 2) + p + sp.Rational(3 * q, 2)) / 3 + delta
    result = integrate_result(
        x**p * y**q / (x**3 - y * y) ** power,
        region=sp.And(x > 0, x < 1, y * y <= x**3 / 4),
        variables=(x, y),
    )
    assert result.outcome is (
        IntegrationOutcome.FINITE if delta < 0 else IntegrationOutcome.DIVERGENT
    )
    assert result.absolute_convergent is bool(delta < 0)


@pytest.mark.parametrize("power", [sp.Rational(11, 12), sp.S.One, sp.Rational(13, 12)])
def test_cusp_angular_threshold_neighbors(power):
    result = integrate_result(
        x * x / (x**3 - y * y) ** power,
        region=sp.And(x > 0, x < 1, y * y <= x**3),
        variables=(x, y),
    )
    assert result.outcome is (
        IntegrationOutcome.FINITE if power < 1 else IntegrationOutcome.DIVERGENT
    )


@pytest.mark.parametrize("coefficient", [1, -1])
def test_vanishing_and_sign_changing_cusp_numerators(coefficient):
    base = x**3 - y * y
    density = coefficient * sp.expand(base**2) * (1 + y) / base ** sp.Rational(5, 2)
    result = integrate_result(density, region=sp.And(x > 0, x < 1, y * y <= x**3), variables=(x, y))
    assert abs(complex(result.value.evalf(30)) - coefficient * complex(sp.pi.evalf(30))) < 1e-25
    assert result.absolute_convergent is True
    assert try_cusp_horn(y / base, sp.And(x > 0, x < 1, y * y <= x**3), (x, y)) is None


@pytest.mark.slow
@pytest.mark.parametrize(
    "dimension,weight,inner,outer",
    [(3, 0, 0, sp.Rational(1, 2)), (3, 2, sp.Rational(1, 2), sp.Rational(3, 2)), (5, 2, 0, 2)],
)
def test_weighted_shell_radial_series_at_multiple_precisions(dimension, weight, inner, outer):
    coords = sp.symbols(f"z0:{dimension}", real=True)
    rho = sum(v * v for v in coords)
    region = (
        semialg.Ball((0,) * dimension, outer)
        if inner == 0
        else semialg.SphericalShell((0,) * dimension, (inner, outer))
    )
    density = (coords[0] ** 2 if weight else 1) * sp.exp(-rho) / (2 + rho)
    result = integrate_result(
        density,
        region=region,
        variables=coords,
        policy=IntegrationPolicy(allow_expensive_exact=dimension > 4),
    )
    assert result.outcome is IntegrationOutcome.FINITE and result.absolute_convergent is True
    area = 2 * mp.pi ** (mp.mpf(dimension) / 2) / mp.gamma(mp.mpf(dimension) / 2)
    with mp.workdps(45):
        # Recompute constants inside the precision context.
        area = 2 * mp.pi ** (mp.mpf(dimension) / 2) / mp.gamma(mp.mpf(dimension) / 2)
        reference = (
            area
            / (dimension if weight else 1)
            * mp.quad(
                lambda r: r ** (dimension - 1 + weight) * mp.exp(-r * r) / (2 + r * r),
                [mp.mpf(str(sp.N(inner, 45))), mp.mpf(str(sp.N(outer, 45)))],
            )
        )
        for precision in (20, 35):
            value = mp.mpf(str(result.value.evalf(precision)))
            assert abs(value - reference) < mp.mpf(10) ** (-precision + 3)


def test_radial_quadratic_map_keeps_its_determinant_and_weight():
    z = sp.Symbol("z", real=True)
    rho = x * x / 4 + y * y / 9 + z * z / 16
    result = integrate_result(
        x * x * sp.exp(-rho) / (2 + rho),
        region=semialg.Ellipsoid((0, 0, 0), sp.diag(4, 9, 16)),
        variables=(x, y, z),
    )
    with mp.workdps(35):
        reference = (
            24 * 4 * 4 * mp.pi / 3 * mp.quad(lambda r: r**4 * mp.exp(-r * r) / (2 + r * r), [0, 1])
        )
        assert abs(mp.mpf(str(result.value.evalf(30))) - reference) < mp.mpf("1e-24")
    assert result.outcome is IntegrationOutcome.FINITE


def test_radial_endpoint_identity_and_construction_budget(monkeypatch):
    def unexpected_numeric(*args, **kwargs):
        raise AssertionError("Construction must not evaluate the series numerically")

    monkeypatch.setattr(sp.Sum, "evalf", unexpected_numeric)
    whole = _moment(sp.Rational(3, 2), sp.S.One, (sp.Integer(2), -sp.S.One), sp.S.Zero, sp.S.One)
    outer = _moment(
        sp.Rational(3, 2), sp.S.One, (sp.Integer(2), -sp.S.One), sp.Rational(1, 2), sp.S.One
    )
    inner = _moment(
        sp.Rational(3, 2), sp.S.One, (sp.Integer(2), -sp.S.One), sp.S.Zero, sp.Rational(1, 2)
    )
    # Bound construction complexity independently of precision/evaluation performance.
    assert all(v.has(sp.Sum) and sp.count_ops(v) < 200 for v in (whole, outer, inner))
    index = sp.Symbol("k", integer=True, nonnegative=True)
    functions = [v.function.subs(v.variables[0], index) for v in (whole, outer, inner)]
    assert sp.simplify(functions[0] - functions[1] - functions[2]) == 0


@pytest.mark.parametrize("dimension", [3, 5])
def test_unsupported_radial_index_and_invalid_shift_decline(dimension):
    coords = sp.symbols(f"r0:{dimension}", real=True)
    rho = sum(v * v for v in coords)
    ball = semialg.Ball((0,) * dimension, 1)
    assert (
        radial_region_data(coords[0] ** 34 * sp.exp(-rho) / (2 + rho), ball, variables=coords)
        is None
    )
    assert radial_region_data(sp.exp(-rho) / (rho - 1), ball, variables=coords) is None


@pytest.mark.parametrize("su,sv", [(-1, -1), (-1, 1), (1, -1), (1, 1)])
@pytest.mark.parametrize("p,q", [(-1, -1), (-2, 1), (0, 0)])
def test_polygon_truncations_across_orthants_and_epsilon_values(su, sv, p, q):
    u = su * x
    v = sv * y
    e = sp.Symbol("epsilon", positive=True)
    formula = sp.And(u >= 0, v >= 0, u <= 1, v <= 1, u + v <= sp.Rational(3, 2))
    result = polygon_product_cutoff(x**p * y**q, formula, (x, y), x * y, e)
    assert result is not None
    with mp.workdps(40):
        for epsilon in (sp.Rational(1, 50), sp.Rational(1, 100), sp.Rational(1, 200)):
            assert result[1]["cutoff_valid_for"].subs(e, epsilon) is sp.true
            eps = mp.mpf(str(epsilon.p)) / epsilon.q

            def integrand(t, eps=eps):
                upper = min(mp.mpf(1), mp.mpf("1.5") - t)
                lower = eps / t
                inner = (
                    mp.log(upper / lower)
                    if q == -1
                    else (upper ** (q + 1) - lower ** (q + 1)) / (q + 1)
                )
                return (su**p) * (sv**q) * t**p * inner

            reference = mp.quad(integrand, [eps, mp.mpf(".5"), mp.mpf(1)])
            value = mp.mpf(str(sp.re(result[0].subs(e, epsilon).evalf(35))))
            assert abs(value - reference) < mp.mpf("1e-29")


def test_independent_finite_part_asymptotics():
    formula = sp.And(x >= 0, y >= 0, x <= sp.Rational(1, 2), x + y <= 1)
    result = regularized_result(
        1 / (x * y),
        region=formula,
        variables=(x, y),
        regularization=HadamardFinitePart(hypersurface=x * y),
    )
    with mp.workdps(45):
        errors = []
        for eps in (mp.mpf("1e-3"), mp.mpf("1e-5"), mp.mpf("1e-7")):
            lower = 2 * eps / (1 + mp.sqrt(1 - 4 * eps))
            truncated = mp.quad(
                lambda u, eps=eps: mp.log(u * (1 - u) / eps) / u, [lower, mp.mpf(".5")]
            )
            constant = truncated - mp.log(eps) ** 2 / 2 - mp.log(2) * mp.log(eps)
            error = abs(constant - mp.mpf(str(result.value.evalf(35))))
            assert error < 2 * eps
            errors.append(error)
        assert errors[2] < errors[1] / 50 < errors[0] / 2500


@pytest.mark.parametrize("kind", ["affine", "graph"])
def test_mapped_polygon_source_geometry_and_jacobian(kind):
    u, v = (x + y, x - y) if kind == "affine" else (x, y - x * x)
    g = u * v
    formula = sp.And(u >= 0, v >= 0, u <= sp.Rational(1, 2), u + v <= 1)
    result = regularized_result(
        1 / g, region=formula, variables=(x, y), regularization=HadamardFinitePart(hypersurface=g)
    )
    expected = (-(sp.pi**2) / 12 + sp.log(2) ** 2) / (2 if kind == "affine" else 1)
    assert sp.simplify(result.value - expected) == 0


def test_unproved_polygon_window_and_budget_do_not_gain_evidence():
    e = sp.Symbol("epsilon", positive=True)
    formula = sp.And(x >= 0, y >= 0, x <= a, x + y <= 1)
    assert (
        polygon_product_cutoff(1 / (x * y), formula, (x, y), x * y, e, assumptions=a >= 0) is None
    )
    from multiple_integrate import StrategyBudget

    result = integrate_result(
        1,
        region=sp.And(x >= 0, x <= 1, y >= 0, y <= 1 - x * x),
        variables=(x, y),
        policy=IntegrationPolicy(semialg=StrategyBudget(max_polynomial_degree=0)),
    )
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert result.complete is not True and result.absolute_convergent is not True


def test_three_segment_polygon_envelope_has_independent_area():
    e = sp.Symbol("epsilon", positive=True)
    formula = sp.And(
        x >= 0,
        y >= 0,
        x <= sp.Rational(5, 4),
        y <= 1,
        y <= sp.Rational(5, 4) - x / 2,
        y <= 2 - 3 * x / 2,
    )
    result = polygon_product_cutoff(1, formula, (x, y), x * y, e)
    assert (
        result is not None
        and len({(segment[4], segment[5]) for segment in result[1]["polygon_segments"]}) == 3
    )
    assert sp.limit(result[0], e, 0, dir="+") == sp.Rational(63, 64)
