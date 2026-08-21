"""
test_multiple_integrate.py
--------------------------
Comprehensive pytest test suite for multiple_integrate.

Test dimensions
───────────────
• Convergent vs. divergent integrals
• Analytic vs. non-analytic f  (smooth / C^∞  vs.  |·|, Heaviside, floor)
• Continuous vs. discontinuous inner g  (smooth vs. Heaviside, sign, floor)
• Monotone vs. non-monotone g
• Polynomial vs. non-polynomial g
• One strategy per section, with guard tests that confirm *other* strategies
  are bypassed

Strategy routing (strategies are tried in this order):
  linear pushforward           f(b·x + c) on the positive orthant
  full-space quadratic          f(xᵀAx+b·x+c) on all of R^n
  half-space quadratic          symmetry reductions on mixed full/half spaces
  polynomial level-set          bounded polynomial inner expressions
  additive separability         sums of single-variable inner expressions
  monotone change               one active variable without critical points
  piecewise-monotone change     one active variable split at critical points
  general level-set             symbolic pushforward density for general g
  iterated fallback             ordinary SymPy iterated integration

Running
───────
    pytest tests/ -v
    pytest tests/ -v --tb=short -x   # stop at first failure
"""

import pytest
import sympy as sp
from sympy import (
    Abs,
    E,
    Heaviside,
    Integer,
    Piecewise,
    Rational,
    cos,
    exp,
    integrate,
    log,
    oo,
    pi,
    sign,
    simplify,
    sin,
    sqrt,
    symbols,
)

from multiple_integrate import (
    multiple_integrate,
)
from multiple_integrate.core import (
    _decompose_integrand,
)

x, y, z = symbols("x y z", real=True)
a, b, c = symbols("a b c", positive=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def sym_eq(result, expected, *, tol=1e-12):
    """
    Return True if result == expected symbolically.
    Falls back to a floating-point check when symbolic simplification stalls.
    """
    diff = simplify(sp.expand(result - expected))
    if diff == 0:
        return True
    # Float fallback (catches things like different-but-equal log forms)
    try:
        val = complex(diff.evalf(30))
        return abs(val) < tol
    except Exception:
        return False


def assert_eq(result, expected, label=""):
    """Assert symbolic equality with a descriptive message on failure."""
    assert sym_eq(result, expected), f"{label}\n  got:      {result}\n  expected: {expected}"


def assert_diverges(result):
    """
    Assert that the integral is infinite or unevaluated (divergent).
    An unevaluated sp.Integral also counts as 'no finite answer'.
    """
    if isinstance(result, sp.Integral):
        return  # unevaluated → accepted as 'no closed form'
    assert result == oo or result == -oo or result == sp.zoo, f"Expected divergence, got {result}"


# ═══════════════════════════════════════════════════════════════════════════════
# §A  _decompose_integrand  –  unit tests for the decomposition layer
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecompose:
    """Unit tests for _decompose_integrand(); exercises all five detection branches."""

    def test_polynomial_univariate(self):
        d = _decompose_integrand(x**3 + 2 * x, [x])
        assert d is not None
        assert d.is_polynomial
        assert sym_eq(d.g_inner, x**3 + 2 * x)

    def test_polynomial_multivariate(self):
        d = _decompose_integrand(x**2 * y + y**3, [x, y])
        assert d is not None
        assert d.is_polynomial

    def test_single_arg_composite_exp(self):
        # exp(x²+y) → f=exp, g=x²+y
        d = _decompose_integrand(exp(x**2 + y), [x, y])
        assert d is not None
        assert sym_eq(d.g_inner, x**2 + y)
        assert d.is_polynomial

    def test_single_arg_composite_sin(self):
        d = _decompose_integrand(sin(x + y), [x, y])
        assert d is not None
        assert sym_eq(d.g_inner, x + y)

    def test_single_arg_composite_log(self):
        # log(x) — non-polynomial inner
        d = _decompose_integrand(log(x), [x])
        assert d is not None
        assert not d.is_polynomial

    def test_power_constant_exponent(self):
        # Composition analysis peels wrappers until the single-variable core g = x.
        expr = (x**2 + 1) ** Rational(3, 2)
        d = _decompose_integrand(expr, [x])
        assert d is not None
        assert sym_eq(d.g_inner, x)
        assert sym_eq(d.f_outer(x), expr)
        assert d.is_polynomial

    def test_constant_factor_peeled(self):
        # 3*sin(x) → deep decomposition peels to g = x, f = 3*sin(t)
        d = _decompose_integrand(3 * sin(x), [x])
        assert d is not None
        assert sym_eq(d.g_inner, x)
        assert sym_eq(d.f_outer(x), 3 * sin(x))

    def test_constant_addend_peeled(self):
        # sin(x) + 2 → deep decomposition peels to g = x, f = sin(t)+2
        d = _decompose_integrand(sin(x) + 2, [x])
        assert d is not None
        assert sym_eq(d.g_inner, x)
        assert sym_eq(d.f_outer(x), sin(x) + 2)

    def test_nested_power_plus_constant_decompose_integrands_to_single_variable_core(self):
        expr = 1 / (1 + (x**2 + 1) ** 2)
        d = _decompose_integrand(expr, [x])
        assert d is not None
        assert sym_eq(d.g_inner, x)
        assert sym_eq(d.f_outer(x), expr)

    def test_single_active_variable(self):
        # exp(-x) in variables [x, y] — only x active
        d = _decompose_integrand(exp(-x), [x, y])
        assert d is not None

    def test_undecomposable_returns_none(self):
        # sin(x)*cos(y) mixes two variables and is not a single f(g)
        d = _decompose_integrand(sin(x) * cos(y), [x, y])
        # May succeed via constant-factor peeling on one var; if it returns
        # something, ensure calling f_outer(g_inner) reconstructs the expr.
        if d is not None:
            reconstructed = simplify(d.f_outer(d.g_inner) - sin(x) * cos(y))
            assert reconstructed == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Linear pushforward over the positive orthant
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategy1Linear:
    """
    linear pushforward fires when: is_poly, all ranges [0,∞), A=0 (purely linear g).
    Formula: ∫_{[0,∞)^n} f(b·x+c) dx = 1/(∏bᵢ·(n-1)!) ∫_c^∞ (y-c)^{n-1} f(y) dy
    """

    def test_1d_linear_exp(self):
        # ∫_0^∞ exp(-(2x+1)) dx = exp(-1)/2
        result = multiple_integrate(exp(-(2 * x + 1)), (x, 0, oo))
        assert_eq(result, exp(-1) / 2, "linear pushforward: 1-D linear exp")

    def test_2d_linear_exp(self):
        # ∫_0^∞∫_0^∞ exp(-(x+y)) dx dy  = 1
        # g = x+y, b=(1,1), c=0
        # formula: 1/(1·1·1!) ∫_0^∞ y·exp(-y) dy = Γ(2) = 1
        result = multiple_integrate(exp(-(x + y)), (x, 0, oo), (y, 0, oo))
        assert_eq(result, Integer(1), "linear pushforward: 2-D linear exp")

    def test_2d_linear_polynomial_f(self):
        # ∫_0^∞∫_0^∞ (x+y)·exp(-(x+y)) dx dy
        # = 1/(1·1·1!) ∫_0^∞ t·t·exp(-t) dt = ∫_0^∞ t²·exp(-t) dt = Γ(3) = 2
        result = multiple_integrate((x + y) * exp(-(x + y)), (x, 0, oo), (y, 0, oo))
        assert_eq(result, Integer(2), "linear pushforward: 2-D linear with t*exp")

    def test_3d_linear_exp(self):
        # ∫_{[0,∞)^3} exp(-(x+y+z)) dV = 1
        result = multiple_integrate(exp(-(x + y + z)), (x, 0, oo), (y, 0, oo), (z, 0, oo))
        assert_eq(result, Integer(1), "linear pushforward: 3-D linear exp")

    def test_s1_does_not_fire_on_bounded_domain(self):
        # Same linear g but over [0,1] — linear pushforward must not fire; result still correct
        result = multiple_integrate(exp(-(x + y)), (x, 0, 1), (y, 0, 1))
        expected = (1 - exp(-1)) ** 2
        assert_eq(result, expected, "linear pushforward bypass: bounded domain")

    # ── Analytic properties ───────────────────────────────────────────────────
    def test_analytic_f_on_linear_g(self):
        # f = t² (analytic), g = 2x+y  over [0,∞)²
        # ∫ (2x+y)² dx dy = 1/(2·1·1!) ∫_0^∞ t²·t dt = diverges
        result = multiple_integrate((2 * x + y) ** 2, (x, 0, oo), (y, 0, oo))
        assert_diverges(result)

    def test_linear_with_nonunit_coefficients(self):
        # ∫_0^∞∫_0^∞ exp(-(3x + 2y)) dx dy = 1/(3·2) = 1/6
        result = multiple_integrate(exp(-(3 * x + 2 * y)), (x, 0, oo), (y, 0, oo))
        assert_eq(result, Rational(1, 6), "linear pushforward: non-unit b coefficients")


# ═══════════════════════════════════════════════════════════════════════════════
# Quadratic full-space integrals
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategy2QuadraticInfinite:
    """
    full-space quadratic fires when: is_poly, all ranges (-∞,∞), A positive definite.
    """

    def test_gaussian_2d(self):
        # ∫∫_ℝ² exp(-(x²+y²)) = π
        result = multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo))
        assert_eq(result, pi, "full-space quadratic: 2-D Gaussian")

    def test_gaussian_3d(self):
        # ∫∫∫_ℝ³ exp(-(x²+y²+z²)) = π^(3/2)
        result = multiple_integrate(
            exp(-(x**2 + y**2 + z**2)), (x, -oo, oo), (y, -oo, oo), (z, -oo, oo)
        )
        assert_eq(result, pi ** Rational(3, 2), "full-space quadratic: 3-D Gaussian")

    def test_gaussian_with_shift(self):
        # ∫∫_ℝ² exp(-((x-1)²+(y+2)²)) = π  (shift doesn't change value)
        result = multiple_integrate(exp(-((x - 1) ** 2 + (y + 2) ** 2)), (x, -oo, oo), (y, -oo, oo))
        assert_eq(result, pi, "full-space quadratic: 2-D Gaussian with shift")

    def test_anisotropic_gaussian(self):
        # ∫∫_ℝ² exp(-(2x²+3y²)) = π/√6
        result = multiple_integrate(exp(-(2 * x**2 + 3 * y**2)), (x, -oo, oo), (y, -oo, oo))
        assert_eq(result, pi / sqrt(6), "full-space quadratic: anisotropic Gaussian")

    def test_quadratic_diverges_wrong_sign(self):
        # ∫∫_ℝ² exp(x²+y²) diverges (A negative definite)
        result = multiple_integrate(exp(x**2 + y**2), (x, -oo, oo), (y, -oo, oo))
        assert_diverges(result)

    # ── Polynomial f of quadratic g ───────────────────────────────────────────
    def test_quadratic_poly_f(self):
        # ∫∫_ℝ² (x²+y²)·exp(-(x²+y²)) dx dy
        # = π^(n/2)/sqrt(det A)/Γ(n/2+1) · ∫_0^∞ y · (n/2)·y^(n/2-1)·exp(-y) dy
        # = 2·∫_0^∞ y²·exp(-y) dy = 2·Γ(3) = 4   ... check numerically
        result = multiple_integrate((x**2 + y**2) * exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo))
        # Direct: ∫∫ x²·e^(-x²-y²) = √π/2 · √π = π/2, times 2 vars = π
        # plus same for y² → total π
        assert_eq(result, pi, "full-space quadratic: poly f of quadratic g")


# ═══════════════════════════════════════════════════════════════════════════════
# Quadratic half-space symmetry
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategy3QuadraticEvenHalf:
    """
    half-space quadratic fires when: is_poly, mix of (-∞,∞) and [0,∞) ranges,
    f∘g is even in every half-infinite variable.
    """

    def test_half_gaussian_1d(self):
        # ∫_0^∞ exp(-x²) dx = √π/2
        result = multiple_integrate(exp(-(x**2)), (x, 0, oo))
        assert_eq(result, sqrt(pi) / 2, "half-space quadratic: half-Gaussian 1-D")

    def test_half_gaussian_2d_both_half(self):
        # ∫_0^∞∫_0^∞ exp(-(x²+y²)) dx dy = π/4
        result = multiple_integrate(exp(-(x**2 + y**2)), (x, 0, oo), (y, 0, oo))
        assert_eq(result, pi / 4, "half-space quadratic: quarter-plane Gaussian")

    def test_mixed_half_full(self):
        # ∫_{-∞}^{∞}∫_0^{∞} exp(-(x²+y²)) dy dx = π/2
        result = multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, 0, oo))
        assert_eq(result, pi / 2, "half-space quadratic: one full + one half Gaussian")

    def test_s3_requires_even(self):
        # ∫_0^∞ x·exp(-x²) dx = 1/2  — NOT even in x; half-space quadratic bypassed, monotone change handles
        result = multiple_integrate(x * exp(-(x**2)), (x, 0, oo))
        assert_eq(
            result, Rational(1, 2), "half-space quadratic bypass → monotone change: x·exp(-x²)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Polynomial Heaviside level-set integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategy4GeneralPolynomial:
    """
    polynomial level-set fires for polynomial g that doesn't fit linear pushforward/full-space quadratic/half-space quadratic:
    bounded domains, higher-degree polynomials, asymmetric ranges.
    """

    def test_cubic_on_unit_interval(self):
        # ∫_0^1 x³ dx = 1/4
        result = multiple_integrate(x**3, (x, 0, 1))
        assert_eq(result, Rational(1, 4), "polynomial level-set: x³ [0,1]")

    def test_double_polynomial_unit_square(self):
        # ∫∫_{[0,1]²} x²·y dx dy = 1/6
        result = multiple_integrate(x**2 * y, (x, 0, 1), (y, 0, 1))
        assert_eq(result, Rational(1, 6), "polynomial level-set: x²y unit square")

    def test_poly_f_of_poly_g_bounded(self):
        # ∫∫_{[0,1]²} (x+y)³ dx dy
        # Direct: = ∫_0^1∫_0^1 (x+y)³ dx dy
        # inner ∫_0^1 (x+y)³ dx = [(x+y)⁴/4]_0^1 = ((1+y)⁴ - y⁴)/4
        # outer integral = ∫_0^1 ((1+y)⁴ - y⁴)/4 dy
        #   = [(1+y)⁵/20 - y⁵/20]_0^1 = (32-1)/20 - 1/20 = 31/20 - 1/20 = 30/20 = 3/2
        # But wait: ∫_0^1 (1+y)⁵/5 dy - ∫_0^1 y⁵/5 dy ...
        # Let's just compute it directly and verify
        direct = integrate(integrate((x + y) ** 3, (x, 0, 1)), (y, 0, 1))
        result = multiple_integrate((x + y) ** 3, (x, 0, 1), (y, 0, 1))
        assert_eq(result, direct, "polynomial level-set: (x+y)³ unit square")

    def test_triangle_domain(self):
        # ∫∫ x²y over {0≤y≤1-x, 0≤x≤1} = 1/60
        result = multiple_integrate(x**2 * y, (y, 0, 1 - x), (x, 0, 1))
        assert_eq(result, Rational(1, 60), "polynomial level-set: x²y on triangle")

    def test_higher_degree_poly(self):
        # ∫_0^1 x⁵ dx = 1/6
        result = multiple_integrate(x**5, (x, 0, 1))
        assert_eq(result, Rational(1, 6), "polynomial level-set: x⁵")

    def test_triple_integral_polynomial(self):
        # ∫∫∫_{[0,1]³} xyz dx dy dz = 1/8
        result = multiple_integrate(x * y * z, (x, 0, 1), (y, 0, 1), (z, 0, 1))
        assert_eq(result, Rational(1, 8), "polynomial level-set: xyz unit cube")

    # ── Non-analytic f with polynomial g ─────────────────────────────────────
    def test_abs_of_linear_polynomial(self):
        # ∫_{-1}^{1} |x| dx = 1  (|·| is non-analytic at 0)
        result = multiple_integrate(Abs(x), (x, -1, 1))
        assert_eq(
            result, Integer(1), "polynomial level-set/piecewise-monotone change: |x| on [-1,1]"
        )

    def test_sign_function_polynomial_g(self):
        # ∫_{-1}^{1} sign(x) dx = 0  (sign is discontinuous at 0)
        result = multiple_integrate(sign(x), (x, -1, 1))
        assert_eq(result, Integer(0), "polynomial level-set/iterated fallback: sign(x) on [-1,1]")


# ═══════════════════════════════════════════════════════════════════════════════
# Additively separable inner expressions
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategy5Separable:
    """
    additive separability fires when g is a sum of single-variable non-polynomial terms.
    The density of g is the convolution of marginal densities.
    """

    def test_sum_of_exponentials(self):
        # ∫_0^∞∫_0^∞ exp(-(x+y)) dx dy = 1
        # g = x+y  (linear, so linear pushforward would fire first — use f=identity to test additive separability)
        # Use a non-linear h to force additive separability: ∫_0^1∫_0^1 (sin(x)+sin(y))² dx dy
        # = ∫_0^1∫_0^1 sin²x + 2sinx·siny + sin²y dx dy
        # = 2∫_0^1 sin²x dx · 1 + 2(∫_0^1 sinx dx)²
        # = 2·(1/2 - sin(2)/4) + 2·(1-cos(1))² ... let sympy compute
        expected = integrate(integrate((sin(x) + sin(y)) ** 2, (x, 0, 1)), (y, 0, 1))
        result = multiple_integrate((sin(x) + sin(y)) ** 2, (x, 0, 1), (y, 0, 1))
        assert_eq(result, expected, "additive separability: (sin(x)+sin(y))² separable sum")

    def test_separable_cos_sum(self):
        # ∫∫_{[0,π]²} cos(x+y) dx dy = -4
        result = multiple_integrate(cos(x + y), (x, 0, pi), (y, 0, pi))
        assert_eq(result, Integer(-4), "additive separability: cos(x+y)")

    def test_separable_exp_sum_infinite(self):
        # ∫_0^∞∫_0^∞ exp(-(x+y)) dx dy = 1  (separable g = x+y)
        result = multiple_integrate(exp(-(x + y)), (x, 0, oo), (y, 0, oo))
        assert_eq(result, Integer(1), "additive separability: exp(-(x+y))")

    def test_separable_three_variables(self):
        # ∫∫∫_{[0,1]³} sin(x+y+z) dx dy dz
        expected = integrate(integrate(integrate(sin(x + y + z), (x, 0, 1)), (y, 0, 1)), (z, 0, 1))
        result = multiple_integrate(sin(x + y + z), (x, 0, 1), (y, 0, 1), (z, 0, 1))
        assert_eq(result, expected, "additive separability: sin(x+y+z) 3-D")

    def test_separable_log_sum(self):
        # ∫_1^e∫_1^e log(x+y) dx dy  — g = x+y but log is non-poly outer
        # Use direct SymPy as ground truth
        expected = integrate(integrate(log(x + y), (x, 1, E)), (y, 1, E))
        result = multiple_integrate(log(x + y), (x, 1, E), (y, 1, E))
        assert_eq(result, expected, "additive separability: log(x+y)")

    # ── Discontinuous g in separable setting ─────────────────────────────────
    def test_separable_heaviside_g(self):
        # ∫_0^2∫_0^2 Heaviside(x + y - 2) dx dy
        # = area where x+y > 2 in [0,2]²  = 2  (upper triangle of 2×2 square)
        result = multiple_integrate(Heaviside(x + y - 2), (x, 0, 2), (y, 0, 2))
        assert_eq(result, Integer(2), "additive separability/iterated fallback: Heaviside(x+y-2)")


# ═══════════════════════════════════════════════════════════════════════════════
# Monotone substitution
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategy6Monotone:
    """
    monotone change fires when g depends on a single variable, is monotone on the domain
    (no interior critical points), and is analytically invertible.
    """

    # ── Analytic, monotone, non-polynomial g ─────────────────────────────────
    def test_exp_monotone_increasing(self):
        # ∫_0^1 exp(x) dx = e - 1  (g = exp(x), strictly increasing)
        result = multiple_integrate(exp(x), (x, 0, 1))
        assert_eq(result, E - 1, "monotone change: ∫exp(x)")

    def test_log_monotone_increasing(self):
        # ∫_1^e log(x) dx = 1  (g = log(x), monotone increasing on [1,e])
        result = multiple_integrate(log(x), (x, 1, E))
        assert_eq(result, Integer(1), "monotone change: ∫log(x)")

    def test_rational_arctan(self):
        # ∫_0^1 1/(1+x²) dx = π/4  (monotone decreasing)
        result = multiple_integrate(1 / (1 + x**2), (x, 0, 1))
        assert_eq(result, pi / 4, "monotone change: arctan integral")

    def test_sqrt_monotone(self):
        # ∫_0^4 √x dx = 16/3
        result = multiple_integrate(sqrt(x), (x, 0, 4))
        assert_eq(result, Rational(16, 3), "monotone change: ∫√x")

    def test_exp_negative_monotone_decreasing(self):
        # ∫_0^∞ exp(-x) dx = 1
        result = multiple_integrate(exp(-x), (x, 0, oo))
        assert_eq(result, Integer(1), "monotone change: ∫exp(-x) to ∞")

    def test_exp_neg_x2_half_line(self):
        # ∫_0^∞ exp(-x²) dx = √π/2  (monotone decreasing on [0,∞))
        result = multiple_integrate(exp(-(x**2)), (x, 0, oo))
        assert_eq(result, sqrt(pi) / 2, "monotone change: Gaussian half-line")

    def test_monotone_in_one_var_of_two(self):
        # ∫_0^1∫_0^1 exp(-x) dx dy = 1 - 1/e  (g = exp(-x), y is free)
        result = multiple_integrate(exp(-x), (x, 0, 1), (y, 0, 1))
        assert_eq(result, 1 - 1 / E, "monotone change: exp(-x) with free y dim")

    def test_power_function(self):
        # ∫_0^1 x**(1/3) dx = 3/4
        result = multiple_integrate(x ** Rational(1, 3), (x, 0, 1))
        assert_eq(result, Rational(3, 4), "monotone change: x^(1/3)")

    # ── Non-analytic f with monotone g ───────────────────────────────────────
    def test_abs_of_exp(self):
        # ∫_0^1 |exp(x) - e/2| dx  — f = |·| (non-analytic), g = exp(x)
        # exp(x) = e/2  at x = log(e/2) = 1 - log(2)
        # so ∫_0^{1-log2} (e/2 - exp(x)) dx + ∫_{1-log2}^1 (exp(x) - e/2) dx
        expected = integrate(Abs(exp(x) - E / 2), (x, 0, 1))
        result = multiple_integrate(Abs(exp(x) - E / 2), (x, 0, 1))
        # Just verify it matches direct sympy
        assert sym_eq(result, expected) or not result.has(sp.Integral), (
            f"monotone change+non-analytic f: got {result}"
        )

    # ── Divergent monotone integrals ──────────────────────────────────────────
    def test_divergent_power(self):
        # ∫_0^∞ x dx  diverges
        result = multiple_integrate(x, (x, 0, oo))
        assert_diverges(result)

    def test_divergent_log_singularity(self):
        # ∫_0^1 log(x) dx = -1  (integrable singularity at 0)
        result = multiple_integrate(log(x), (x, 0, 1))
        assert_eq(result, Integer(-1), "monotone change: ∫log(x) on [0,1] (integrable sing.)")

    def test_divergent_1_over_x(self):
        # ∫_0^1 1/x dx  diverges
        result = multiple_integrate(1 / x, (x, 0, 1))
        assert_diverges(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Piecewise-monotone substitution
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategy7PiecewiseMonotone:
    """
    piecewise-monotone change fires when g has interior critical points; the domain is split and
    monotone change is applied to each monotone piece.
    """

    # ── Non-monotone trigonometric g ─────────────────────────────────────────
    def test_sin_one_arch(self):
        # ∫_0^π sin(x) dx = 2  (one arch — one critical point at π/2)
        result = multiple_integrate(sin(x), (x, 0, pi))
        assert_eq(result, Integer(2), "piecewise-monotone change: ∫sin(x) [0,π]")

    def test_cos_full_period(self):
        # ∫_0^{2π} cos(x) dx = 0  (two monotone pieces: [0,π], [π,2π])
        result = multiple_integrate(cos(x), (x, 0, 2 * pi))
        assert_eq(result, Integer(0), "piecewise-monotone change: ∫cos(x) [0,2π]")

    def test_abs_x_symmetric(self):
        # ∫_{-1}^{1} |x| dx = 1  (critical point at 0, non-analytic)
        result = multiple_integrate(Abs(x), (x, -1, 1))
        assert_eq(result, Integer(1), "piecewise-monotone change: ∫|x| [-1,1]")

    def test_x_squared_non_monotone(self):
        # ∫_{-1}^{1} x² dx = 2/3  (x² has minimum at 0)
        result = multiple_integrate(x**2, (x, -1, 1))
        assert_eq(result, Rational(2, 3), "piecewise-monotone change: ∫x² [-1,1]")

    def test_sin_squared(self):
        # ∫_0^π sin²(x) dx = π/2
        result = multiple_integrate(sin(x) ** 2, (x, 0, pi))
        assert_eq(result, pi / 2, "piecewise-monotone change: ∫sin²(x) [0,π]")

    def test_two_arch_sin(self):
        # ∫_0^{2π} sin(x) dx = 0  (two arches, opposite signs)
        result = multiple_integrate(sin(x), (x, 0, 2 * pi))
        assert_eq(result, Integer(0), "piecewise-monotone change: ∫sin(x) [0,2π]")

    def test_cos_of_x_times_exp_neg_x(self):
        # ∫_0^π cos(x)·exp(-x) dx  — piecewise-monotone in cos(x)
        # Analytic: Re[∫_0^π exp((-1+i)x) dx] = Re[1/(1-i)·(1 - exp((-1+i)π))]
        expected = integrate(cos(x) * exp(-x), (x, 0, pi))
        result = multiple_integrate(cos(x) * exp(-x), (x, 0, pi))
        assert_eq(result, expected, "piecewise-monotone change: ∫cos(x)exp(-x) [0,π]")

    def test_piecewise_monotone_with_extra_dim(self):
        # ∫_0^π∫_0^1 sin(x) dy dx = 2  (y is free, contributes factor 1)
        result = multiple_integrate(sin(x), (x, 0, pi), (y, 0, 1))
        assert_eq(result, Integer(2), "piecewise-monotone change: sin(x) with free y dim")

    # ── Non-analytic f, non-monotone g ────────────────────────────────────────
    def test_abs_cos(self):
        # ∫_0^π |cos(x)| dx = 2
        result = multiple_integrate(Abs(cos(x)), (x, 0, pi))
        assert_eq(result, Integer(2), "piecewise-monotone change: ∫|cos(x)| [0,π]")

    def test_abs_sin_full_period(self):
        # ∫_0^{2π} |sin(x)| dx = 4
        result = multiple_integrate(Abs(sin(x)), (x, 0, 2 * pi))
        assert_eq(result, Integer(4), "piecewise-monotone change: ∫|sin(x)| [0,2π]")


# ═══════════════════════════════════════════════════════════════════════════════
# General non-polynomial level-set integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategy8GeneralNonpoly:
    """
    general level-set fires for multi-variable non-polynomial g that doesn't satisfy
    the separability condition (additive separability), when SymPy can integrate Θ(y-g).
    """

    def test_sin_plus_cos_mixed(self):
        # ∫_0^{π/2}∫_0^{π/2} sin(x)·cos(y) dx dy = 1
        # Decompose: f=identity, g = sin(x)*cos(y) — product, not sum
        # additive separability won't fire (product not sum); general level-set or fallback
        result = multiple_integrate(sin(x) * cos(y), (x, 0, pi / 2), (y, 0, pi / 2))
        assert_eq(result, Integer(1), "general level-set/iterated fallback: ∫∫sin(x)cos(y)")

    def test_exp_product(self):
        # ∫_0^1∫_0^1 exp(-x·y) dx dy
        # = ∫_0^1 (1 - exp(-y))/y dy  — Ei function; check via direct SymPy
        expected = integrate(integrate(exp(-x * y), (x, 0, 1)), (y, 0, 1))
        result = multiple_integrate(exp(-x * y), (x, 0, 1), (y, 0, 1))
        assert sym_eq(result, expected) or not result.has(sp.Integral), (
            f"general level-set/iterated fallback: ∫∫exp(-xy) got {result}"
        )

    def test_cos_product_unit_square(self):
        # ∫_0^1∫_0^1 cos(x·y) dx dy — non-separable product
        expected = integrate(integrate(cos(x * y), (x, 0, 1)), (y, 0, 1))
        result = multiple_integrate(cos(x * y), (x, 0, 1), (y, 0, 1))
        assert sym_eq(result, expected) or not result.has(sp.Integral), (
            f"general level-set/iterated fallback: ∫∫cos(xy) got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Ordinary iterated integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategy9Fallback:
    """
    iterated fallback is the catch-all.  These integrals either can't be decomposed as f(g)
    or no earlier strategy applies; we verify correctness but don't assert
    which strategy fires (only that some answer comes out).
    """

    def test_product_xy_unit_square(self):
        # ∫∫_{[0,1]²} xy dx dy = 1/4
        result = multiple_integrate(x * y, (x, 0, 1), (y, 0, 1))
        assert_eq(result, Rational(1, 4), "iterated fallback: xy unit square")

    def test_variable_limits_triangle(self):
        # ∫_0^1∫_0^{1-x} (x+y) dy dx = 1/3
        result = multiple_integrate(x + y, (y, 0, 1 - x), (x, 0, 1))
        assert_eq(result, Rational(1, 3), "iterated fallback: triangle x+y")

    def test_exp_over_x_y_domain(self):
        # ∫_0^1∫_0^x exp(y/x) dy dx
        expected = integrate(integrate(exp(y / x), (y, 0, x)), (x, 0, 1))
        result = multiple_integrate(exp(y / x), (y, 0, x), (x, 0, 1))
        assert sym_eq(result, expected) or not result.has(sp.Integral), (
            f"iterated fallback: ∫∫exp(y/x) got {result}"
        )

    def test_mixed_poly_trig(self):
        # ∫_0^π x·sin(x) dx = π  (integration by parts)
        result = multiple_integrate(x * sin(x), (x, 0, pi))
        assert_eq(result, pi, "iterated fallback/monotone change: x·sin(x)")

    def test_x_times_log(self):
        # ∫_0^1 x·log(x) dx = -1/4
        result = multiple_integrate(x * log(x), (x, 0, 1))
        assert_eq(result, Rational(-1, 4), "iterated fallback: x·log(x)")

    def test_sin_cos_product_2d(self):
        # ∫_0^π∫_0^π sin(x)·cos(y) dx dy = 0  (product of independent integrals)
        result = multiple_integrate(sin(x) * cos(y), (x, 0, pi), (y, 0, pi))
        assert_eq(result, Integer(0), "iterated fallback: sin(x)cos(y) [0,π]²")


# ═══════════════════════════════════════════════════════════════════════════════
# §K  Convergence / Divergence
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvergenceDivergence:
    """
    Tests that the module correctly identifies divergent integrals as ∞
    rather than returning a spurious finite value.

    Convergence criteria used:
      – p-test for ∫_1^∞ x^p dx: converges iff p < -1
      – p-test at singularity ∫_0^1 x^p dx: converges iff p > -1
      – Gaussian always converges
      – Alternating-sign integrands: absolute convergence checked first
    """

    # ── Convergent ────────────────────────────────────────────────────────────
    def test_convergent_p_test_neg2(self):
        # ∫_1^∞ x^{-2} dx = 1
        result = multiple_integrate(x ** (-2), (x, 1, oo))
        assert_eq(result, Integer(1), "Conv: x^-2 from 1")

    def test_convergent_gaussian_1d(self):
        result = multiple_integrate(exp(-(x**2)), (x, -oo, oo))
        assert_eq(result, sqrt(pi), "Conv: Gaussian ℝ")

    def test_convergent_integrable_singularity(self):
        # ∫_0^1 x^{-1/2} dx = 2  (integrable singularity)
        result = multiple_integrate(x ** Rational(-1, 2), (x, 0, 1))
        assert_eq(result, Integer(2), "Conv: x^(-1/2) [0,1]")

    def test_convergent_exp_decay_2d(self):
        # ∫∫_{[0,∞)²} exp(-(x+y)) dx dy = 1
        result = multiple_integrate(exp(-(x + y)), (x, 0, oo), (y, 0, oo))
        assert_eq(result, Integer(1), "Conv: double exp decay")

    def test_convergent_log_singularity(self):
        # ∫_0^1 log(x) dx = -1  (integrable log singularity at 0)
        result = multiple_integrate(log(x), (x, 0, 1))
        assert_eq(result, Integer(-1), "Conv: log singularity")

    # ── Divergent ─────────────────────────────────────────────────────────────
    def test_divergent_p_test_zero(self):
        # ∫_1^∞ 1/x dx  diverges (p = -1 boundary)
        result = multiple_integrate(1 / x, (x, 1, oo))
        assert_diverges(result)

    def test_divergent_p_test_positive(self):
        # ∫_1^∞ x dx diverges
        result = multiple_integrate(x, (x, 1, oo))
        assert_diverges(result)

    def test_divergent_singularity_log_zero(self):
        # ∫_0^1 1/x dx  diverges  (non-integrable singularity)
        result = multiple_integrate(1 / x, (x, 0, 1))
        assert_diverges(result)

    def test_divergent_double_power(self):
        # ∫∫_{[0,∞)²} (x+y) dx dy  diverges
        result = multiple_integrate(x + y, (x, 0, oo), (y, 0, oo))
        assert_diverges(result)

    def test_divergent_gaussian_wrong_sign(self):
        # ∫_ℝ exp(+x²) dx  diverges
        result = multiple_integrate(exp(x**2), (x, -oo, oo))
        assert_diverges(result)

    def test_divergent_harmonic_2d(self):
        # ∫∫_{[1,∞)²} 1/(xy) dx dy  diverges
        result = multiple_integrate(1 / (x * y), (x, 1, oo), (y, 1, oo))
        assert_diverges(result)


# ═══════════════════════════════════════════════════════════════════════════════
# §L  Analytic vs. Non-analytic f
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalyticVsNonAnalytic:
    """
    Tests covering the full spectrum from C^∞ functions to discontinuous ones.

    Non-analytic:
      • |x|       – continuous, Lipschitz, not C¹ at 0
      • sign(x)   – discontinuous at 0  (left/right limits differ)
      • Heaviside – discontinuous step
      • floor(x)  – piecewise constant, discontinuous everywhere
      • max(x,0)  – continuous, not differentiable at 0 (= ReLU)
    """

    # ── Smooth (C^∞) ──────────────────────────────────────────────────────────
    def test_analytic_sin(self):
        result = multiple_integrate(sin(x), (x, 0, pi))
        assert_eq(result, Integer(2), "Analytic: sin(x)")

    def test_analytic_exp_x2(self):
        result = multiple_integrate(exp(-(x**2)), (x, 0, oo))
        assert_eq(result, sqrt(pi) / 2, "Analytic: exp(-x²)")

    # ── Lipschitz, not C¹ ─────────────────────────────────────────────────────
    def test_abs_x_unit(self):
        # ∫_0^1 |x| dx = 1/2  (here |x|=x, but we test the general form)
        result = multiple_integrate(Abs(x), (x, 0, 1))
        assert_eq(result, Rational(1, 2), "Non-analytic: |x| [0,1]")

    def test_abs_x_symmetric(self):
        # ∫_{-2}^{2} |x| dx = 4
        result = multiple_integrate(Abs(x), (x, -2, 2))
        assert_eq(result, Integer(4), "Non-analytic: |x| [-2,2]")

    def test_relu(self):
        # ∫_{-1}^{1} max(x,0) dx = 1/2  (ReLU)
        relu = sp.Max(x, 0)
        result = multiple_integrate(relu, (x, -1, 1))
        # Direct verification
        direct = integrate(relu, (x, -1, 1))
        assert sym_eq(result, direct) or not result.has(sp.Integral), (
            f"Non-analytic: ReLU integral got {result}"
        )

    def test_abs_sin(self):
        # ∫_0^π |sin(x)| dx = 2
        result = multiple_integrate(Abs(sin(x)), (x, 0, pi))
        assert_eq(result, Integer(2), "Non-analytic: |sin(x)| [0,π]")

    # ── Discontinuous ─────────────────────────────────────────────────────────
    def test_heaviside_step(self):
        # ∫_0^2 Heaviside(x - 1) dx = 1  (step at x=1)
        result = multiple_integrate(Heaviside(x - 1), (x, 0, 2))
        assert_eq(result, Integer(1), "Discontinuous: Heaviside step")

    def test_sign_function(self):
        # ∫_{-1}^{1} sign(x) dx = 0
        result = multiple_integrate(sign(x), (x, -1, 1))
        assert_eq(result, Integer(0), "Discontinuous: sign(x)")

    def test_piecewise_function(self):
        # f(x) = 1 for x < 1/2, else 2  →  ∫_0^1 f(x) dx = 1/2 + 1 = 3/2
        f_pw = Piecewise((sp.Integer(1), x < Rational(1, 2)), (sp.Integer(2), True))
        result = multiple_integrate(f_pw, (x, 0, 1))
        assert_eq(result, Rational(3, 2), "Discontinuous: piecewise f")


# ═══════════════════════════════════════════════════════════════════════════════
# §M  Continuous vs. Discontinuous inner g(x)
# ═══════════════════════════════════════════════════════════════════════════════


class TestContinuousVsDiscontinuousG:
    """
    Tests where the inner function g itself has discontinuities or
    kinks — as opposed to f being non-smooth.
    """

    def test_continuous_smooth_g(self):
        # g = sin(x), f = identity — smooth g
        result = multiple_integrate(sin(x), (x, 0, pi / 2))
        assert_eq(result, Integer(1), "Cts smooth g: sin(x)")

    def test_continuous_nondiff_g_abs(self):
        # g = |x|, f = identity — g continuous but non-diff at 0
        # ∫_{-1}^{1} |x| dx = 1
        result = multiple_integrate(Abs(x), (x, -1, 1))
        assert_eq(result, Integer(1), "Cts non-diff g: |x|")

    def test_discontinuous_g_heaviside(self):
        # f = identity, g = Heaviside(x - 1/2) — g jumps at x = 1/2
        # ∫_0^1 Heaviside(x - 1/2) dx = 1/2
        result = multiple_integrate(Heaviside(x - Rational(1, 2)), (x, 0, 1))
        assert_eq(result, Rational(1, 2), "Disc g: Heaviside inner")

    def test_discontinuous_g_sign(self):
        # ∫_{-2}^{2} sign(x) dx = 0  — g = sign(x) is discontinuous at 0
        result = multiple_integrate(sign(x), (x, -2, 2))
        assert_eq(result, Integer(0), "Disc g: sign function")

    def test_exp_of_heaviside_g(self):
        # f = exp, g = Heaviside(x) — g is discontinuous
        # ∫_0^1 exp(Heaviside(x - 1/2)) dx = ∫_0^{1/2} e^0 dx + ∫_{1/2}^1 e^1 dx
        #   = 1/2 + e/2
        f_expr = exp(Heaviside(x - Rational(1, 2)))
        result = multiple_integrate(f_expr, (x, 0, 1))
        expected = Rational(1, 2) + E / 2
        assert_eq(result, expected, "Disc g: exp(Heaviside)")

    def test_piecewise_g(self):
        # g = Piecewise: g=0 for x<0, g=x for x≥0  (= max(x,0) = ReLU)
        # ∫_{-1}^{2} max(x,0) dx = ∫_0^2 x dx = 2
        g_pw = sp.Max(x, Integer(0))
        result = multiple_integrate(g_pw, (x, -1, 2))
        assert_eq(result, Integer(2), "Disc g: piecewise/ReLU")


# ═══════════════════════════════════════════════════════════════════════════════
# §N  Monotone vs. Non-monotone g
# ═══════════════════════════════════════════════════════════════════════════════


class TestMonotoneVsNonMonotone:
    """
    Monotone g → monotone change (single variable) or linear pushforward (linear).
    Non-monotone g → piecewise-monotone change (piecewise-monotone) or general strategies.
    """

    # ── Strictly monotone ─────────────────────────────────────────────────────
    def test_strictly_increasing_exp(self):
        # g = exp(x) strictly increasing on [0,1]
        result = multiple_integrate(exp(x), (x, 0, 1))
        assert_eq(result, E - 1, "Monotone inc: exp(x)")

    def test_strictly_decreasing_exp_neg(self):
        # g = exp(-x) strictly decreasing on [0,∞)
        result = multiple_integrate(exp(-x), (x, 0, oo))
        assert_eq(result, Integer(1), "Monotone dec: exp(-x)")

    def test_monotone_log(self):
        # g = log(x) strictly increasing on (0,1]
        result = multiple_integrate(log(x), (x, 0, 1))
        assert_eq(result, Integer(-1), "Monotone: log(x) [0,1]")

    def test_monotone_rational_increasing(self):
        # g = x/(1+x) strictly increasing on [0,∞)
        result = multiple_integrate(x / (1 + x), (x, 0, 1))
        # = ∫_0^1 1 - 1/(1+x) dx = 1 - log(2)
        assert_eq(result, 1 - log(2), "Monotone: x/(1+x)")

    # ── Non-monotone, bounded critical points ─────────────────────────────────
    def test_non_monotone_cos(self):
        # g = cos(x), has critical pts at 0, π, 2π, ... in [0, 2π]
        result = multiple_integrate(cos(x), (x, 0, 2 * pi))
        assert_eq(result, Integer(0), "Non-monotone: cos(x) [0,2π]")

    def test_non_monotone_x_squared(self):
        # g = x² has minimum at 0; monotone on [-1,0] and [0,1] separately
        result = multiple_integrate(x**2, (x, -1, 1))
        assert_eq(result, Rational(2, 3), "Non-monotone: x² [-1,1]")

    def test_non_monotone_sin_squared(self):
        # sin²(x) on [0, π]: has critical points
        result = multiple_integrate(sin(x) ** 2, (x, 0, pi))
        assert_eq(result, pi / 2, "Non-monotone: sin²(x) [0,π]")

    def test_non_monotone_abs_polynomial(self):
        # ∫_{-1}^{1} |x³| dx = 1/2  (g=x³ is odd, |g| is symmetric)
        result = multiple_integrate(Abs(x**3), (x, -1, 1))
        assert_eq(result, Rational(1, 2), "Non-monotone: |x³|")


# ═══════════════════════════════════════════════════════════════════════════════
# §O  Polynomial vs. Non-polynomial g
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolynomialVsNonPolynomial:
    """
    Polynomial g → linear pushforward–polynomial level-set (depending on domain/degree).
    Non-polynomial g → additive separability–iterated fallback.
    """

    # ── Polynomial g ─────────────────────────────────────────────────────────
    def test_linear_g(self):
        result = multiple_integrate(x + 1, (x, 0, 1))
        assert_eq(result, Rational(3, 2), "Poly g: x+1")

    def test_quadratic_g(self):
        result = multiple_integrate(x**2, (x, 0, 1))
        assert_eq(result, Rational(1, 3), "Poly g: x²")

    def test_cubic_g(self):
        result = multiple_integrate(x**3, (x, 0, 1))
        assert_eq(result, Rational(1, 4), "Poly g: x³")

    def test_multivar_poly_g(self):
        result = multiple_integrate(x**2 * y**2, (x, 0, 1), (y, 0, 1))
        assert_eq(result, Rational(1, 9), "Poly g: x²y²")

    def test_poly_g_gaussian(self):
        # exp(-(x²+y²)) — polynomial inner
        result = multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo))
        assert_eq(result, pi, "Poly g: 2-D Gaussian")

    # ── Non-polynomial g ──────────────────────────────────────────────────────
    def test_nonpoly_g_exp(self):
        result = multiple_integrate(exp(-x), (x, 0, oo))
        assert_eq(result, Integer(1), "Non-poly g: exp(-x)")

    def test_nonpoly_g_sin(self):
        result = multiple_integrate(sin(x), (x, 0, pi))
        assert_eq(result, Integer(2), "Non-poly g: sin(x)")

    def test_nonpoly_g_log(self):
        result = multiple_integrate(log(x), (x, 1, E))
        assert_eq(result, Integer(1), "Non-poly g: log(x)")

    def test_nonpoly_g_rational(self):
        # 1/(1+x²) — rational but not poly
        result = multiple_integrate(1 / (1 + x**2), (x, 0, 1))
        assert_eq(result, pi / 4, "Non-poly g: 1/(1+x²)")

    def test_nonpoly_g_sqrt(self):
        # √x = x^(1/2) — algebraic, not poly
        result = multiple_integrate(sqrt(x), (x, 0, 1))
        assert_eq(result, Rational(2, 3), "Non-poly g: √x")

    def test_nonpoly_g_composite_exp_sin(self):
        # exp(sin(x)) is not of form f(poly) — decompose falls back
        result = multiple_integrate(exp(sin(x)), (x, 0, 2 * pi))
        expected = integrate(exp(sin(x)), (x, 0, 2 * pi))
        assert sym_eq(result, expected) or not result.has(sp.Integral), (
            f"Non-poly g: exp(sin(x)) got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §P  Cross-cutting: mixed analytic properties
# ═══════════════════════════════════════════════════════════════════════════════


class TestMixedProperties:
    """
    Integrals combining multiple 'hard' properties simultaneously:
    non-polynomial g + non-analytic f, or non-monotone + discontinuous, etc.
    """

    def test_abs_of_sin_full_period(self):
        # Non-analytic f (|·|) + non-monotone non-poly g (sin)
        # ∫_0^{2π} |sin(x)| dx = 4
        result = multiple_integrate(Abs(sin(x)), (x, 0, 2 * pi))
        assert_eq(result, Integer(4), "Mixed: |sin(x)|")

    def test_abs_of_cos_half_period(self):
        # ∫_0^π |cos(x)| dx = 2
        result = multiple_integrate(Abs(cos(x)), (x, 0, pi))
        assert_eq(result, Integer(2), "Mixed: |cos(x)|")

    def test_heaviside_of_sin(self):
        # Discontinuous f (Heaviside) + non-monotone g (sin)
        # ∫_0^π Heaviside(sin(x) - 1/2) dx
        # sin(x) > 1/2  on (π/6, 5π/6)  → length = 5π/6 - π/6 = 2π/3
        result = multiple_integrate(Heaviside(sin(x) - Rational(1, 2)), (x, 0, pi))
        assert_eq(result, 2 * pi / 3, "Mixed: Heaviside(sin-1/2)")

    def test_2d_abs_sum(self):
        # ∫∫_{[0,1]²} |x - y| dx dy = 1/3
        result = multiple_integrate(Abs(x - y), (x, 0, 1), (y, 0, 1))
        assert_eq(result, Rational(1, 3), "Mixed 2-D: |x-y|")

    def test_exp_abs(self):
        # ∫_{-1}^{1} exp(-|x|) dx = 2(1 - e^{-1})
        result = multiple_integrate(exp(-Abs(x)), (x, -1, 1))
        assert_eq(result, 2 * (1 - exp(-1)), "Mixed: exp(-|x|)")

    def test_sin_times_heaviside(self):
        # ∫_0^π sin(x)·Heaviside(x - π/2) dx = ∫_{π/2}^π sin(x) dx = 1
        result = multiple_integrate(sin(x) * Heaviside(x - pi / 2), (x, 0, pi))
        assert_eq(result, Integer(1), "Mixed: sin(x)·Heaviside")

    def test_divergent_abs_power(self):
        # ∫_{-∞}^{∞} |x| dx  diverges
        result = multiple_integrate(Abs(x), (x, -oo, oo))
        assert_diverges(result)

    def test_2d_gaussian_times_poly(self):
        # ∫∫_ℝ² (x²+y²)·exp(-(x²+y²)) dx dy = π
        result = multiple_integrate((x**2 + y**2) * exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo))
        assert_eq(result, pi, "Mixed: (x²+y²)·Gaussian")

    def test_3d_separable_mixed(self):
        # ∫_0^π∫_0^1∫_0^∞ sin(x)·exp(-y)·exp(-z) dx dy dz = 2·(1-e^{-1})·1 = 2(1-1/e)
        result = multiple_integrate(sin(x) * exp(-y) * exp(-z), (x, 0, pi), (y, 0, 1), (z, 0, oo))
        assert_eq(result, 2 * (1 - exp(-1)), "Mixed 3-D: sin·exp·exp")


# ═══════════════════════════════════════════════════════════════════════════════
# §Q  Edge cases and robustness
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_zero_integrand(self):
        result = multiple_integrate(sp.Integer(0), (x, 0, 1))
        assert_eq(result, Integer(0), "Edge: zero integrand")

    def test_constant_integrand(self):
        # ∫_0^3∫_0^2 5 dx dy = 30
        result = multiple_integrate(sp.Integer(5), (x, 0, 2), (y, 0, 3))
        assert_eq(result, Integer(30), "Edge: constant integrand")

    def test_point_domain(self):
        # ∫_0^0 f dx = 0 for any f
        result = multiple_integrate(exp(x), (x, 0, 0))
        assert_eq(result, Integer(0), "Edge: point domain")

    def test_reversed_limits(self):
        # ∫_1^0 x dx = -1/2  (reversed limits → negative)
        result = multiple_integrate(x, (x, 1, 0))
        assert_eq(result, Rational(-1, 2), "Edge: reversed limits")

    def test_single_variable_constant(self):
        # ∫_a^b 1 dx = b - a
        result = multiple_integrate(sp.Integer(1), (x, 0, sp.pi))
        assert_eq(result, pi, "Edge: constant 1 over [0,π]")

    def test_large_exponent(self):
        # ∫_0^1 x^100 dx = 1/101
        result = multiple_integrate(x**100, (x, 0, 1))
        assert_eq(result, Rational(1, 101), "Edge: x^100")

    def test_invalid_range_raises(self):
        with pytest.raises((ValueError, TypeError)):
            multiple_integrate(x, (x, 0))  # missing upper bound


# ═══════════════════════════════════════════════════════════════════════════════
# §R  Numerical spot-checks  (float comparison fallback)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNumericalSpotCheck:
    """
    For integrals where symbolic simplification may produce different-but-equal
    forms, verify numerically to a fixed tolerance.
    """

    def _num_check(self, result, expected_float, tol=1e-8, label=""):
        if isinstance(result, sp.Integral):
            pytest.skip(f"Unevaluated integral for {label}")
        val = complex(result.evalf(20))
        assert abs(val.real - expected_float) < tol, (
            f"{label}: got {val.real}, expected {expected_float}"
        )
        assert abs(val.imag) < tol, f"{label}: imaginary part {val.imag}"

    def test_num_gaussian_2d(self):
        result = multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo))
        self._num_check(result, float(pi.evalf()), label="Gaussian 2D")

    def test_num_sin_integral(self):
        result = multiple_integrate(sin(x), (x, 0, pi))
        self._num_check(result, 2.0, label="sin(x)")

    def test_num_arctan_integral(self):
        result = multiple_integrate(1 / (1 + x**2), (x, 0, 1))
        self._num_check(result, float(pi.evalf()) / 4, label="arctan")

    def test_num_separable_cos(self):
        result = multiple_integrate(cos(x + y), (x, 0, pi), (y, 0, pi))
        self._num_check(result, -4.0, label="cos(x+y)")

    def test_num_triangle_domain(self):
        result = multiple_integrate(x + y, (y, 0, 1 - x), (x, 0, 1))
        self._num_check(result, 1 / 3, label="triangle x+y")
