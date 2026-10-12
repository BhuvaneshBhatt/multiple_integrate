"""Independent oracles and broken prerequisites for the new proved families."""

from dataclasses import replace

import mpmath as mp
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
from multiple_integrate.anisotropic import anisotropic_absolute, try_anisotropic_moments
from multiple_integrate.cusp_horns import try_cusp_horn
from multiple_integrate.parameter_cells import _equality_substitutions
from multiple_integrate.policy import IntegrationPolicy
from multiple_integrate.quadratic_radial import weighted_quadratic_data
from multiple_integrate.singular_maps import triangle_moment

x, y, a, b = sp.symbols("x y a b", real=True)
FULL = ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))


@pytest.mark.parametrize(
    "condition,parameters,expected",
    [
        (sp.Eq(a, 0), (a,), {a: 0}),
        (sp.And(sp.Eq(a + b, 3), sp.Eq(a - b, 1)), (a, b), {a: 2, b: 1}),
        (sp.Eq(a, b + 1), (a, b), {a: b + 1}),
        (sp.Eq(a * a, 1), (a,), {}),
        (sp.Or(sp.Eq(a, 0), sp.Eq(a, 1)), (a,), {}),
    ],
)
def test_exact_linear_cell_specialization(condition, parameters, expected):
    assert _equality_substitutions(condition, parameters, IntegrationPolicy()) == expected


def test_exceptional_cell_needs_no_manual_substitution():
    result = integrate_cells(
        1,
        region=sp.And(sp.Eq(x, 0), a >= 0),
        variables=(x,),
        measure_dimension="intrinsic",
        parameters=(a,),
        cells=(ParameterCell(a > 0), ParameterCell(sp.Eq(a, 0)), ParameterCell(a < 0)),
    )
    assert result.complete
    assert result.value.subs(a, 0) == 1
    assert result.value.subs(a, -1) == 0


@pytest.mark.parametrize("power,degree", [(1, 0), (2, 0), (3, 2), (4, 4)])
def test_mixed_gaussian_quadratic_matches_independent_radial_quadrature(power, degree):
    result = integrate_result(
        x**degree * sp.exp(-x * x - y * y) / (2 + x * x + y * y) ** power, *FULL
    )
    with mp.workdps(40):
        angular = mp.quad(lambda t: mp.cos(t) ** degree, [0, mp.pi, 2 * mp.pi])
        oracle = angular * mp.quad(
            lambda r: r ** (degree + 1) * mp.exp(-r * r) / (2 + r * r) ** power, [0, 1, mp.inf]
        )
        assert abs(complex(result.value.evalf(35)) - complex(oracle)) < 1e-30
    assert result.absolute_convergent is True


@pytest.mark.parametrize(
    "density",
    [
        sp.exp(x * x + y * y) / (1 + x * x + y * y),
        sp.exp(-x * x - y * y) / (a + x * x + y * y),
        sp.exp(-((x - y) ** 2)) / (1 + x * x + y * y),
    ],
)
def test_mixed_profile_requires_positive_shift_and_full_rank(density):
    assert weighted_quadratic_data(density, (x, y)) is None


@pytest.mark.parametrize("powers", [(4, 2), (4, 4), (6, 2), (2, 2)])
def test_anisotropic_moments_have_dirichlet_oracle(powers):
    m, n = powers
    density = 1 / (1 + x**m + y**n) ** 2
    expected = (
        4
        * sp.gamma(sp.Rational(1, m))
        * sp.gamma(sp.Rational(1, n))
        * sp.gamma(2 - sp.Rational(1, m) - sp.Rational(1, n))
        / (m * n)
    )
    result = try_anisotropic_moments(density, FULL)
    assert sp.simplify(result - expected) == 0


@pytest.mark.parametrize(
    "power,expected", [(sp.Rational(1, 2), True), (sp.Rational(3, 4), False), (1, False)]
)
def test_anisotropic_local_threshold(power, expected):
    assert anisotropic_absolute((x**4 + y * y) ** (-power), ((x, -1, 1), (y, -1, 1))) is expected


def test_anisotropic_odd_cancellation_requires_absolute_integrability():
    assert try_anisotropic_moments(x / (1 + x**4 + y * y) ** 3, FULL) == 0
    assert try_anisotropic_moments(x / (1 + x**4 + y * y) ** sp.Rational(1, 4), FULL) is None


@pytest.mark.parametrize(
    "density",
    [1 / (1 - x**4 + y * y) ** 2, 1 / (1 + x**3 + y * y) ** 2, 1 / (1 + x**4 + x * y + y * y) ** 2],
)
def test_anisotropic_broken_geometry_declines(density):
    assert try_anisotropic_moments(density, FULL) is None


@pytest.mark.parametrize(
    "p,q,expected", [(-2, -2, 0), (-1, -1, -(sp.pi**2) / 6), (0, 0, sp.Rational(1, 2))]
)
def test_triangle_cutoff_constant_term(p, q, expected):
    e = sp.Symbol("e", positive=True)
    expr = triangle_moment(p, q, e)
    expansion = sp.expand(sp.series(expr, e, 0, 1).removeO())
    constant = sum(t for t in sp.Add.make_args(expansion) if not t.has(e))
    assert sp.simplify(constant - expected) == 0


def test_clipped_affine_diamond_finite_part():
    g = (x + y) * (x - y)
    result = regularized_result(
        1 / g**2, (x, -1, 1), (y, -1, 1), regularization=HadamardFinitePart(hypersurface=g)
    )
    assert sp.simplify(result.value + 2 * sp.log(2)) == 0
    assert verify_regularization_result(result).valid


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_polynomial_graph_shared_cutoff(degree):
    g = x * (y - x**degree)
    region = sp.And(x >= -1, x <= 1, y - x**degree >= -1, y - x**degree <= 1)
    result = regularized_result(
        1 / g**2, region=region, variables=(x, y), regularization=HadamardFinitePart(hypersurface=g)
    )
    assert result.value == 4
    assert verify_regularization_result(result).valid
    details = dict(result.provenance.details, graph_jacobian=sp.Integer(2))
    assert not verify_regularization_result(
        replace(result, provenance=replace(result.provenance, details=details))
    ).valid


def test_malformed_regulator_is_invalid_without_crashing():
    result = regularized_result(1 / x**2, (x, -1, 1), regularization=HadamardFinitePart())
    broken = replace(result, provenance=replace(result.provenance, regulator=None))
    assert not verify_regularization_result(broken).valid


@pytest.mark.parametrize("power", [sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(3, 4)])
def test_cusp_blowup_matches_independent_one_dimensional_oracle(power):
    formula = sp.And(x > 0, x < 1, y * y <= x**3 / 4)
    result = integrate_result((x**3 - y * y) ** (-power), region=formula, variables=(x, y))
    with mp.workdps(40):
        s = mp.mpf(str(power.evalf(45)))
        oracle = mp.quad(lambda t: (1 - t * t) ** (-s), [-mp.mpf(".5"), 0, mp.mpf(".5")]) / (
            mp.mpf("2.5") - 3 * s
        )
        assert abs(complex(result.value.evalf(35)) - complex(oracle)) < 1e-30
    assert result.absolute_convergent is True


@pytest.mark.parametrize("power", [sp.Rational(5, 6), 1, 2])
def test_cusp_critical_and_supercritical_thresholds_diverge(power):
    result = integrate_result(
        (x**3 - y * y) ** (-power), region=sp.And(x > 0, x < 1, y * y <= x**3 / 4), variables=(x, y)
    )
    assert result.value == sp.oo
    assert result.outcome is IntegrationOutcome.DIVERGENT
    assert result.absolute_convergent is False


@pytest.mark.parametrize("k", [0, 2, -1])
def test_cusp_requires_a_real_supported_horn(k):
    assert (
        try_cusp_horn(1 / sp.sqrt(x**3 - y * y), sp.And(x > 0, x < 1, y * y <= k * x**3), (x, y))
        is None
    )


def test_parameter_branch_proofs_retain_their_conditions():
    result = integrate_result((b + x * x + y * y) ** -2, *FULL, assumptions=b >= 0)
    claims = [c for c in result.convergence_evidence if c.kind == "absolute"]
    assert any(c.established is True and c.condition == (b > 0) for c in claims)
    assert any(c.established is False and c.condition == sp.Eq(b, 0) for c in claims)
    assert result.absolute_convergent is None


def test_intrinsic_curve_has_chart_level_convergence():
    result = integrate_result(
        1,
        region=sp.And(sp.Eq(y, x), x >= 0, x <= 1),
        variables=(x, y),
        measure_dimension="intrinsic",
    )
    assert result.value == sp.sqrt(2)
    assert result.absolute_convergent is True
    assert any(
        c.method == "semialg_intrinsic_reduction" and c.scope == "absolute_convergence"
        for c in result.coverage
    )


def test_query_completion_does_not_claim_geometry_certification(monkeypatch):
    from multiple_integrate.singularities import stratify_singular_locus

    monkeypatch.setattr(
        "multiple_integrate.singularities.semialg_region_dimension", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "multiple_integrate.singularities.semialg_is_satisfiable", lambda *a, **k: None
    )
    result = stratify_singular_locus(1 / x, (x,), return_details=True)
    assert result.complete and not result.geometry_certified


def test_mixed_finite_ball_has_independent_radial_oracle():
    import semialg

    result = integrate_result(
        x * x * sp.exp(-x * x - y * y) / (2 + x * x + y * y) ** 2,
        region=semialg.Ball((0, 0), 1),
        variables=(x, y),
    )
    with mp.workdps(40):
        expected = mp.pi * mp.quad(lambda r: r**3 * mp.exp(-r * r) / (2 + r * r) ** 2, [0, 1])
        assert abs(complex(result.value.evalf(35)) - complex(expected)) < 1e-30
    assert result.absolute_convergent is True


def test_graph_cutoff_is_invariant_under_coordinate_permutation():
    g = y * (x - y * y)
    formula = sp.And(y >= -1, y <= 1, x - y * y >= -1, x - y * y <= 1)
    result = regularized_result(
        1 / g**2,
        region=formula,
        variables=(x, y),
        regularization=HadamardFinitePart(hypersurface=g),
    )
    assert result.value == 4 and verify_regularization_result(result).valid


def test_intrinsic_point_has_finite_counting_measure_evidence():
    import semialg

    result = integrate_result(
        3, region=semialg.Point((0, 0)), variables=(x, y), measure_dimension="intrinsic"
    )
    assert result.value == 3 and result.absolute_convergent is True


def test_new_family_documentation_has_executable_traceability():
    from pathlib import Path

    root = Path(__file__).parents[1]
    page = (root / "docs/proved-families.md").read_text()
    assert "tests/test_gap_improvements.py" in page
    for phrase in (
        "Scoped and intrinsic",
        "Exact parameter",
        "Mixed Gaussian",
        "Anisotropic",
        "Shared cutoffs",
        "Cusp blowup",
        "Verification limits",
    ):
        assert phrase in page
    assert "proved-families" in (root / "docs/index.rst").read_text()


@pytest.mark.parametrize("bad", [0, -1, sp.oo, float("nan"), True, "1"])
def test_backend_deadline_requires_a_positive_finite_duration(bad):
    with pytest.raises(ValueError):
        IntegrationPolicy(backend_timeout_seconds=bad)


def test_isolated_backend_timeout_reaps_worker():
    import multiprocessing

    from multiple_integrate.isolated_queries import isolated_query

    before = {p.pid for p in multiprocessing.active_children()}
    assert isolated_query("sympy", "integrate", (sp.exp(x), (x, 0, 1)), {}, 0.001) is None
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_isolated_backend_success_and_errors():
    from multiple_integrate.isolated_queries import isolated_query

    assert isolated_query("sympy", "integrate", (x, (x, 0, 1)), {}, 10) == sp.Rational(1, 2)
    with pytest.raises(AttributeError):
        isolated_query("sympy", "not_an_api", (), {}, 10)


def test_symbolic_fallback_timeout_remains_unevaluated():
    from multiple_integrate.assumptions import _integrate
    from multiple_integrate.policy import _CURRENT_POLICY

    token = _CURRENT_POLICY.set(IntegrationPolicy(backend_timeout_seconds=0.001))
    try:
        assert _integrate(sp.exp(x), (x, 0, 1)) == sp.Integral(sp.exp(x), (x, 0, 1))
    finally:
        _CURRENT_POLICY.reset(token)


@pytest.mark.parametrize(
    "density,ranges",
    [(sp.exp(-x * x), ((x, -1, 1),)), (sp.exp(-x * x - y * y), ((x, 0, 1), (y, 0, 1)))],
)
def test_independent_finite_box_verification(density, ranges):
    from multiple_integrate import verify_finite_result_numerically

    result = integrate_result(density, *ranges)
    check = verify_finite_result_numerically(result, density, *ranges, digits=25)
    assert check.consistent is True and check.certified is False
    wrong = verify_finite_result_numerically(
        replace(result, value=result.value + 1), density, *ranges, digits=25
    )
    assert wrong.consistent is False


def test_numeric_verification_specializes_parameters_and_declines_poles():
    from multiple_integrate import verify_finite_result_numerically

    result = integrate_result(sp.exp(-a * x * x), (x, 0, 1), assumptions=a > 0)
    assert (
        verify_finite_result_numerically(
            result, sp.exp(-a * x * x), (x, 0, 1), substitutions={a: 1}, digits=25
        ).consistent
        is True
    )
    assert (
        verify_finite_result_numerically(result, 1 / x, (x, 0, 1), substitutions={a: 1}).consistent
        is None
    )


def test_affine_geometry_inside_a_parameter_cell():
    from multiple_integrate.symbolic_geometry import integrate_symbolic_boxes

    u = x + a * y
    formula = sp.And(u >= 0, u <= 1, y >= 0, y <= 2)
    result = integrate_symbolic_boxes(u, formula, (x, y), assumptions=a > 0)
    assert result.value == 1 and result.absolute_convergent is True
    assert result.method == "affine_parameter_cell"


def test_semialg_deadline_is_an_unknown_query():
    from multiple_integrate.adapters.semialg import is_satisfiable
    from multiple_integrate.policy import _CURRENT_POLICY

    token = _CURRENT_POLICY.set(IntegrationPolicy(backend_timeout_seconds=0.001))
    try:
        assert is_satisfiable(x > 0, (x,)) is None
    finally:
        _CURRENT_POLICY.reset(token)


def test_policy_scope_covers_orchestration_and_is_restored():
    from multiple_integrate.policy import _CURRENT_POLICY, _with_policy

    policy = IntegrationPolicy(backend_timeout_seconds=0.001)
    before = _CURRENT_POLICY.get()

    @_with_policy
    def query(*, policy=None):
        return _CURRENT_POLICY.get()

    assert query(policy=policy) is policy
    assert _CURRENT_POLICY.get() is before


def test_plain_evaluation_cache_retains_branch_proofs():
    from multiple_integrate import clear_cache, integrate

    clear_cache()
    density = (b + x * x + y * y) ** -2
    integrate(density, *FULL, assumptions=b >= 0)
    first = integrate_result(density, *FULL, assumptions=b >= 0)
    second = integrate_result(density, *FULL, assumptions=b >= 0)
    assert first.convergence_evidence == second.convergence_evidence
    assert any(
        c.kind == "absolute" and c.established is True and c.condition == (b > 0)
        for c in second.convergence_evidence
    )
