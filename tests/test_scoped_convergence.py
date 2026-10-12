"""Theorem prerequisites and counterexamples for scoped convergence evidence."""

import pytest
import sympy as sp

from multiple_integrate import IntegrationOutcome, integrate_result
from multiple_integrate.convergence_evidence import absolute_rule, compact_continuity

x, y, z, a = sp.symbols("x y z a", real=True)


@pytest.mark.parametrize(
    "density",
    [
        sp.exp(x * y),
        sp.sin(x + y),
        sp.Abs(x - y),
        sp.log(2 + x),
        sp.sqrt(1 + x * x),
        (2 + x) ** (-sp.Rational(3, 2)),
        x**2 + a * y,
    ],
)
def test_compact_continuous_families(density):
    assert compact_continuity(density, ((x, -1, 1), (y, -1, 1)))
    assert absolute_rule(density, ((x, -1, 1), (y, -1, 1)))[0] is True


@pytest.mark.parametrize(
    "density", [1 / x, sp.log(x), sp.sqrt(x), sp.tan(x), 1 / (x * x + y * y), sp.log(a + x)]
)
def test_compact_rule_refuses_broken_prerequisites(density):
    assert not compact_continuity(density, ((x, -1, 1), (y, -1, 1)))


def test_compact_proof_uses_parameter_assumptions_without_sampling():
    assert compact_continuity(sp.log(a + x), ((x, 0, 1),), assumptions=a > 0)
    assert not compact_continuity(sp.log(a + x), ((x, 0, 1),))


@pytest.mark.parametrize(
    "coordinates, exponent, expected",
    [
        ((x, y), -sp.Rational(1, 2), True),
        ((x, y), -1, False),
        ((x, y), -2, False),
        ((x, y, z), -1, True),
        ((x, y, z), -sp.Rational(3, 2), False),
    ],
)
def test_local_radial_dimension_threshold(coordinates, exponent, expected):
    f = sum(v * v for v in coordinates) ** exponent
    claim, method = absolute_rule(f, tuple((v, -1, 1) for v in coordinates))
    assert claim is expected and method == "radial_power_comparison"


@pytest.mark.parametrize(
    "exponent, expected", [(-2, True), (-1, False), (-sp.Rational(1, 2), False)]
)
def test_full_space_radial_tail(exponent, expected):
    assert (
        absolute_rule((1 + x * x + y * y) ** exponent, ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo)))[0]
        is expected
    )


def test_radial_divergence_overrides_an_unresolved_evaluation(monkeypatch):
    monkeypatch.setattr(
        "multiple_integrate.api.integration.integrate", lambda *a, **k: sp.Integral(a[0], *a[1:])
    )
    r = integrate_result(1 / (x * x + y * y), (x, -1, 1), (y, -1, 1))
    assert r.outcome is IntegrationOutcome.DIVERGENT
    assert r.ordinary_convergent is False and r.absolute_convergent is False


def test_parameter_radial_threshold_is_scoped():
    f = (x * x + y * y) ** a
    assert absolute_rule(f, ((x, -1, 1), (y, -1, 1)), assumptions=a > -1)[0] is True
    assert absolute_rule(f, ((x, -1, 1), (y, -1, 1)), assumptions=a <= -1)[0] is False
    assert absolute_rule(f, ((x, -1, 1), (y, -1, 1)))[0] is None


@pytest.mark.parametrize(
    "density, ranges, method",
    [
        (
            x ** (-sp.Rational(1, 2)) * y ** (-2),
            ((x, 0, 1), (y, 1, sp.oo)),
            "absolute_product_theorem",
        ),
        (
            sp.exp(-x * x) * (1 + y * y) * sp.exp(-y * y),
            ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo)),
            "absolute_product_theorem",
        ),
        (
            (1 + x * x + y * y) * sp.exp(-2 * x * x + 2 * x * y - 2 * y * y),
            ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo)),
            "positive_definite_gaussian_moments",
        ),
    ],
)
def test_absolute_multivariate_proofs(density, ranges, method):
    claim, actual = absolute_rule(density, ranges)
    assert claim is True
    # The stronger Gaussian rule may also certify separated Gaussian moments.
    assert actual in {method, "positive_definite_gaussian_moments"}


@pytest.mark.parametrize(
    "density, ranges",
    [
        (1 / (x * y * y), ((x, 0, 1), (y, 1, sp.oo))),
        (sp.exp(-x * x + y * y), ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))),
        (sp.exp(-((x - y) ** 2)), ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))),
        (sp.exp(-x * x - y * y), ((x, 0, y), (y, -sp.oo, sp.oo))),
        ((x * x - y * y) / (x * x + y * y) ** 2, ((x, -1, 1), (y, -1, 1))),
    ],
)
def test_multivariate_rules_do_not_claim_fubini_without_proof(density, ranges):
    assert absolute_rule(density, ranges)[0] is None


def test_coupled_gaussian_coordinate_permutation_and_symbol_renaming():
    f = (1 + x * y) * sp.exp(-2 * x * x + 2 * x * y - 2 * y * y)
    ranges = ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
    assert absolute_rule(f, ranges)[0] is True
    assert absolute_rule(f, tuple(reversed(ranges)))[0] is True
    assert absolute_rule(f.xreplace({x: z}), ((z, -sp.oo, sp.oo), ranges[1]))[0] is True


def test_coordinate_symbol_assumptions_cannot_remove_endpoint_singularity():
    t = sp.Symbol("t", positive=True)
    assert not compact_continuity(t**-2, ((t, 0, 1),))
    assert absolute_rule(t**-2, ((t, 0, 1),))[0] is None
    result = integrate_result(t**-2, (t, 0, 1))
    assert result.outcome is IntegrationOutcome.DIVERGENT
    assert result.absolute_convergent is not True


def test_coordinate_symbol_assumptions_cannot_certify_a_log_branch():
    t = sp.Symbol("t", positive=True)
    assert not compact_continuity(sp.log(t), ((t, -1, 1),))
