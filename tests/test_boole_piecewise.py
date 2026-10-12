import sympy as sp

from multiple_integrate import Boole, integrate


class TestBooleSupport:
    def test_boole_interval_restriction(self):
        x = sp.symbols("x", real=True)
        result = integrate(x * Boole(x >= 0), (x, -1, 1))
        assert sp.simplify(result - sp.Rational(1, 2)) == 0

    def test_boole_graph_region_from_box(self):
        x, y = sp.symbols("x y", real=True)
        result = integrate((x + y) * Boole(y <= 1 - x), (x, 0, 1), (y, 0, 1))
        assert sp.simplify(result - sp.Rational(1, 3)) == 0

    def test_boole_disk_restriction_from_square(self):
        x, y = sp.symbols("x y", real=True)
        result = integrate(Boole(x**2 + y**2 <= 1), (x, -1, 1), (y, -1, 1))
        assert sp.simplify(result - sp.pi) == 0


class TestPiecewiseSupport:
    def test_piecewise_univariate(self):
        x = sp.symbols("x", real=True)
        expr = sp.Piecewise((x, x < 1), (x**2, True))
        result = integrate(expr, (x, 0, 2))
        assert sp.simplify(result - sp.Rational(17, 6)) == 0

    def test_piecewise_indicator_on_box(self):
        x, y = sp.symbols("x y", real=True)
        expr = sp.Piecewise((1, y <= 1 - x), (0, True))
        result = integrate(expr, (x, 0, 1), (y, 0, 1))
        assert sp.simplify(result - sp.Rational(1, 2)) == 0

    def test_piecewise_graph_branches(self):
        x, y = sp.symbols("x y", real=True)
        expr = sp.Piecewise((y, y <= 1 - x), (x, True))
        result = integrate(expr, (x, 0, 1), (y, 0, 1))
        assert sp.simplify(result - sp.Rational(1, 2)) == 0

    def test_piecewise_constant_outside_supported_restriction_falls_back(self):
        x, y = sp.symbols("x y", real=True)
        expr = sp.Piecewise((1, x * y < 1), (2, True))
        result = integrate(expr, (x, 0, 1), (y, 0, 1))
        # On the unit square x*y < 1 except at the point (1,1), which has measure zero.
        assert sp.simplify(result - 1) == 0


def test_nonexhaustive_piecewise_preserves_undefined_domain():
    from multiple_integrate import IntegrationOutcome, integrate_result

    x = sp.symbols("x", real=True)
    density = sp.Piecewise((x, x < 0))
    result = integrate_result(density, (x, -1, 1))
    assert result.outcome is IntegrationOutcome.UNKNOWN
    assert result.value.has(sp.Integral)
    assert result.absolute_convergent is not True


def test_nonexhaustive_piecewise_can_cover_the_requested_interval():
    x = sp.symbols("x", real=True)
    assert integrate(sp.Piecewise((x, x < 0)), (x, -2, -1)) == -sp.Rational(3, 2)


def test_piecewise_overlapping_conditions_keep_first_match():
    x = sp.symbols("x", real=True)
    density = sp.Piecewise((1, x < 1), (2, x < 2), (3, True))
    assert integrate(density, (x, 0, 3)) == 6


def test_nested_indicators_and_piecewise_share_exact_semantics():
    x, y = sp.symbols("x y", real=True)
    density = Boole(x >= 0) * Boole(y <= 1 - x) * sp.Piecewise((2, y < x), (1, True))
    assert integrate(density, (y, 0, 1), (x, 0, 1)) == sp.Rational(3, 4)
