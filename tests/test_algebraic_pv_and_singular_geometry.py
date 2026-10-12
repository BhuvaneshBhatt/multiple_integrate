"""Independent PV and shared-cutoff oracles for algebraic and singular loci."""

import mpmath as mp
import pytest
import sympy as sp

from multiple_integrate import (
    HadamardFinitePart,
    HypersurfacePrincipalValue,
    IntegrationOutcome,
    integrate,
    integrate_result,
    regularized_result,
    verify_regularization_result,
)
from multiple_integrate.rational_cutoff import certified_order
from multiple_integrate.results import UnresolvedIntegration

x, y, z, a = sp.symbols("x y z a", real=True)


@pytest.mark.parametrize("sign", [-1, 1])
def test_algebraic_two_pole_pv_matches_real_primitive(sign):
    f = 1 / ((x - sign) * (x - sign * sp.sqrt(2)))
    ranges = (x, -2, 0) if sign < 0 else (x, 0, 2)
    result = integrate_result(f, ranges, principal_value=True)
    oracle = sp.log(sp.sqrt(2) - 1) / (sp.sqrt(2) - 1)
    assert result.outcome is IntegrationOutcome.FINITE
    assert abs(sp.N(result.value - oracle, 40)) < sp.Float("1e-35")


def test_algebraic_pv_truncated_numerical_oracle():
    # Numeric primitive uses real logarithms and explicit ordered exclusions.
    result = integrate_result(1 / ((x + 1) * (x + sp.sqrt(2))), (x, -2, 0), principal_value=True)
    with mp.workdps(60):
        root = mp.sqrt(2)

        def primitive(t):
            return (mp.log(abs(t + 1)) - mp.log(abs(t + root))) / (root - 1)

        eps = mp.mpf("1e-35")
        cutoff = (
            primitive(-root - eps)
            - primitive(-2)
            + primitive(-1 - eps)
            - primitive(-root + eps)
            + primitive(0)
            - primitive(-1 + eps)
        )
        assert abs(mp.mpf(str(sp.N(result.value, 55))) - cutoff) < mp.mpf("1e-23")


def test_order_is_mathematical_and_unknown_is_refused():
    assert certified_order((-1, -sp.sqrt(2), 0)) == (-sp.sqrt(2), -1, 0)
    with pytest.raises(UnresolvedIntegration):
        certified_order((a, 0))


def test_algebraic_double_pole_is_divergent_pv_but_has_finite_part():
    f = 1 / (x + sp.sqrt(2)) ** 2
    assert (
        integrate_result(f, (x, -2, 0), principal_value=True).outcome
        is IntegrationOutcome.DIVERGENT
    )
    result = regularized_result(f, (x, -2, 0), regularization=HadamardFinitePart())
    oracle = -1 / sp.sqrt(2) + 1 / (sp.sqrt(2) - 2)
    assert sp.simplify(result.value - oracle) == 0
    assert verify_regularization_result(result).valid


def test_finite_part_rejects_explicit_outside_point():
    with pytest.raises(ValueError, match="inside"):
        integrate(1 / x**2, (x, -1, 1), regularization=HadamardFinitePart(singularities=(2,)))


@pytest.mark.parametrize("dimension,expected", [(2, 4), (3, -8)])
def test_intersecting_plane_finite_parts_use_shared_product_cutoff(dimension, expected):
    variables = (x, y, z)[:dimension]
    g = sp.prod(variables)
    ranges = tuple((v, -1, 1) for v in variables)
    result = regularized_result(g**-2, *ranges, regularization=HadamardFinitePart(hypersurface=g))
    assert result.value == expected
    assert result.provenance.details["realization"] == "coordinate_product_cutoff"
    assert result.provenance.details["orthants"] == 2**dimension
    assert result.coverage[0].complete is True
    assert verify_regularization_result(result).strength == "cutoff_replay"
    assert verify_regularization_result(result).valid


def test_crossing_shared_cutoff_divergence_is_structured():
    result = regularized_result(
        1 / (x * y) ** 2, (x, -1, 1), (y, -1, 1), principal_value=HypersurfacePrincipalValue(x * y)
    )
    assert result.outcome is IntegrationOutcome.DIVERGENT
    assert result.finite is False
    with pytest.raises(ValueError, match="does not exist"):
        integrate(
            1 / (x * y) ** 2,
            (x, -1, 1),
            (y, -1, 1),
            principal_value=HypersurfacePrincipalValue(x * y),
        )


@pytest.mark.parametrize("regularization", [False, True])
def test_crossing_odd_cancellation_is_valid_on_each_truncated_domain(regularization):
    kwargs = (
        {"regularization": HadamardFinitePart(hypersurface=x * y)}
        if regularization
        else {"principal_value": HypersurfacePrincipalValue(x * y)}
    )
    result = regularized_result(1 / (x * y), (x, -1, 1), (y, -1, 1), **kwargs)
    assert result.value == 0 and result.outcome is IntegrationOutcome.FINITE


def test_shifted_scaled_product_geometry_and_polynomial_density():
    g = 2 * (x - 1) * (y + 2)
    result = regularized_result(
        1 / g**2 + 1, (x, -1, 3), (y, -5, 1), regularization=HadamardFinitePart(hypersurface=g)
    )
    assert result.value == sp.Rational(145, 6)  # 24 + (1/4)*(4/6)
    assert result.provenance.details["centers"] == (1, -2)
    assert verify_regularization_result(result).valid


def test_repeated_plane_defining_polynomial_keeps_its_cutoff_convention():
    result = regularized_result(
        x**-2, (x, -1, 1), (y, -1, 1), regularization=HadamardFinitePart(hypersurface=x**2)
    )
    eps = result.provenance.cutoff_parameter
    assert result.value == -4
    assert sp.simplify(result.provenance.cutoff_integral - (4 / sp.sqrt(eps) - 4)) == 0


def test_regular_volume_under_singular_product_cutoff():
    result = regularized_result(
        1, (x, -2, 2), (y, -3, 3), principal_value=HypersurfacePrincipalValue(x * y)
    )
    assert result.value == 24


def test_general_cusp_is_still_explicitly_unsupported():
    result = integrate_result(
        1 / (y**2 - x**3),
        (x, -1, 1),
        (y, -1, 1),
        principal_value=HypersurfacePrincipalValue(y**2 - x**3),
    )
    assert result.outcome is IntegrationOutcome.UNSUPPORTED
    assert result.limitations and result.finite is None


def test_mixed_real_and_complex_roots_have_real_smooth_remainder():
    result = integrate_result(
        1 / ((x + sp.sqrt(2)) * (x * x + 1)), (x, -2, 0), principal_value=True
    )
    oracle = (
        sp.log(2) / 6 - sp.log(2 - sp.sqrt(2)) / 3 + sp.log(5) / 6 + sp.sqrt(2) * sp.atan(2) / 3
    )
    assert sp.simplify(result.value - oracle) == 0
    assert not result.value.has(sp.I)


def test_uncertified_native_real_primitive_is_unknown(monkeypatch):
    import multiple_integrate.cauchy_principal_value as pv

    monkeypatch.setattr(pv, "real_rational_cutoff", lambda *args, **kwargs: None)
    result = integrate_result(1 / x, (x, -1, 1), principal_value=True)
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert any("primitive" in reason for reason in result.limitations)


@pytest.mark.parametrize(
    "g,density,expected",
    [
        (x * y, x**2, sp.Rational(4, 3)),
        (x**2 * y, 1, 4),
        (x * y**2, x**2 + y**2, sp.Rational(8, 3)),
    ],
)
def test_product_cutoff_recovers_regular_moments(g, density, expected):
    result = regularized_result(
        density, (x, -1, 1), (y, -1, 1), principal_value=HypersurfacePrincipalValue(g)
    )
    assert sp.simplify(result.value - expected) == 0


def test_shared_cutoff_exact_density_matches_numeric_truncation():
    result = regularized_result(
        1 / (x * y) ** 2,
        (x, -2, 2),
        (y, -3, 3),
        regularization=HadamardFinitePart(hypersurface=x * y),
    )
    e = result.provenance.cutoff_parameter
    oracle = 4 / e * (sp.log(6 / e) - 1) + sp.Rational(2, 3)
    assert sp.simplify(result.provenance.cutoff_integral - oracle) == 0
    with mp.workdps(40):
        eps = mp.mpf("0.25")
        numeric = 4 * mp.quad(lambda t: 1 / (eps * t) - 1 / (3 * t * t), [eps / 3, 2])
        value = mp.mpf(str(sp.N(result.provenance.cutoff_integral.subs(e, sp.Rational(1, 4)), 35)))
        assert abs(value - numeric) < mp.mpf("1e-30")


def test_product_density_parameter_requires_nonzero_evidence():
    result = regularized_result(
        1 / (a * x * y) ** 2,
        (x, -1, 1),
        (y, -1, 1),
        regularization=HadamardFinitePart(hypersurface=x * y),
    )
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert any("coefficients" in reason for reason in result.limitations)


def test_nonzero_parameter_assumption_enables_product_density():
    result = regularized_result(
        1 / (a * x * y) ** 2,
        (x, -1, 1),
        (y, -1, 1),
        assumptions=a > 0,
        regularization=HadamardFinitePart(hypersurface=x * y),
    )
    assert result.outcome is IntegrationOutcome.FINITE
    assert sp.simplify(result.value - 4 / a**2) == 0
