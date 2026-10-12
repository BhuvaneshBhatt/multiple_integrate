"""Real ellipsoid reduction must not implicitly move complex contours."""

import pytest
import sympy as sp

from multiple_integrate.quadratic_gaussian import _integrate_quadratic


@pytest.mark.parametrize("shift", [sp.I, 2 * sp.I, sp.Symbol("b")])
def test_layercake_rejects_unproved_real_translation(shift):
    assert (
        _integrate_quadratic(
            lambda t: 1 / (1 + t), sp.eye(1), sp.Matrix([2 * shift]), shift**2, 1, {}
        )
        is None
    )


@pytest.mark.parametrize("constant", [sp.I, sp.oo, sp.Symbol("c")])
def test_layercake_requires_real_finite_offset(constant):
    assert (
        _integrate_quadratic(lambda t: sp.exp(-t), sp.eye(1), sp.zeros(1, 1), constant, 1, {})
        is None
    )


def test_positive_eigenvalues_do_not_certify_complex_ellipsoid():
    matrix = sp.Matrix([[1 + sp.I, 1], [1, 1 - sp.I]])
    assert set(matrix.eigenvals()) == {1}
    assert _integrate_quadratic(lambda t: sp.exp(-t), matrix, sp.zeros(2, 1), 0, 2, {}) is None


def test_real_translation_retains_correct_rational_integral():
    assert (
        _integrate_quadratic(lambda t: 1 / (1 + t), sp.eye(1), sp.Matrix([4]), sp.Integer(4), 1, {})
        == sp.pi
    )


def test_complex_translation_counterexample_has_zero_integral():
    # Both poles are below the real axis; closing above gives zero. This
    # independent antiderivative check exposes the erroneous real-shift value pi.
    x = sp.Symbol("x", real=True)
    primitive = sp.log(x + sp.I) / (2 * sp.I) - sp.log(x + 3 * sp.I) / (2 * sp.I)
    assert sp.simplify(sp.diff(primitive, x) - 1 / (1 + (x + 2 * sp.I) ** 2)) == 0
    assert sp.limit(primitive, x, sp.oo) - sp.limit(primitive, x, -sp.oo) == 0


def test_complex_pole_locus_uses_real_and_imaginary_equations():
    from multiple_integrate.singularities import stratify_singular_locus

    x, y = sp.symbols("x y", real=True)
    denominator = 1 + (x + 2 * sp.I) ** 2 + y**2
    details = stratify_singular_locus(1 / denominator**2, (x, y), return_details=True)
    assert len(details.strata) == 1
    stratum = details.strata[0]
    assert stratum.dimension == 0 and stratum.codimension == 2
    assert stratum.regular is True and details.geometry_certified is True
    assert not stratum.condition.has(sp.I)
    assert sp.solve([sp.re(denominator), sp.im(denominator)], (x, y)) == [
        (0, -sp.sqrt(3)),
        (0, sp.sqrt(3)),
    ]


def test_composition_dispatch_declines_complex_level_set_before_real_routes(monkeypatch):
    import multiple_integrate.core as core
    from multiple_integrate.recognition import _IntegrandDecomposition

    def forbidden(*args, **kwargs):
        pytest.fail("complex level set reached a real pushforward strategy")

    monkeypatch.setattr(core, "_try_fullspace_quadratic", forbidden)
    monkeypatch.setattr(core, "_try_additive_separable", forbidden)
    x, y = sp.symbols("x y", real=True)
    inner = 1 + (x + 2 * sp.I) ** 2 + y**2
    decomposition = _IntegrandDecomposition(lambda t: t**-2, inner, True)
    assert (
        core._run_decomp_strategies(
            decomposition, [x, y], [(x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo)], {}
        )
        is None
    )
