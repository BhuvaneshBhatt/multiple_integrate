"""Run one checked representative for every family in the method reference guide.

    PYTHONPATH=src python examples/method_catalogue.py

The names describe mathematical families; automatic dispatch can select a faster
applicable strategy for the same integral. Proof-only and verification methods use
their specialist APIs explicitly. Each example checks a value or theorem claim.
"""

import mpmath as mp
import semialg
import sympy as sp

from multiple_integrate import (
    Boole,
    HadamardFinitePart,
    integrate_cells,
    integrate_result,
    regularized_result,
    verify_finite_result_numerically,
)
from multiple_integrate.cell_proofs import interval_nonnegative
from multiple_integrate.coarea import CoareaChart, integrate_coarea
from multiple_integrate.convergence_evidence import compact_continuity
from multiple_integrate.series_evaluation import evaluate_entire_box

x, y, z, a, t, u = sp.symbols("x y z a t u", real=True)
unit = ((x, 0, 1), (y, 0, 1))
plane = ((x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
triangle = sp.And(x >= 0, y >= 0, x + y <= 1)


def check(f, *ranges, expected, **options):
    result = integrate_result(f, *ranges, **options)
    assert sp.simplify(sp.expand_func(result.value - expected)) == 0, (result.value, expected)
    return result


def iterated():
    return check(x * x * y, (x, 0, 1), (y, 0, 2), expected=sp.Rational(2, 3))


def box_moments():
    return check((x + y) ** 2, *unit, expected=sp.Rational(7, 6))


def exp_trig():
    exp_result = check(x * sp.exp(2 * x), (x, 0, 1), expected=(sp.exp(2) + 1) / 4)
    check(x * sp.cos(x), (x, 0, sp.pi), expected=-2)
    return exp_result


def gaussian_recurrence():
    return check(x * x * sp.exp(-x * x), (x, -sp.oo, sp.oo), expected=sp.sqrt(sp.pi) / 2)


def quadratic_gaussian():
    return check(sp.exp(-x * x - x * y - y * y), *plane, expected=2 * sp.pi / sp.sqrt(3))


def beta_mellin():
    return check(1 / (sp.sqrt(x) * (1 + x)), (x, 0, sp.oo), expected=sp.pi)


def laplace():
    return check(sp.sqrt(x) * sp.exp(-x), (x, 0, sp.oo), expected=sp.sqrt(sp.pi) / 2)


def rational():
    return check(1 / (1 + x * x), (x, 0, 1), expected=sp.pi / 4)


def oscillatory():
    result = check(sp.sin(1 / x) / x, (x, 0, 1), expected=sp.pi / 2 - sp.Si(1))
    assert result.ordinary_convergent is True and result.absolute_convergent is False
    return result


def simplex():
    return check(
        x * y,
        region=semialg.Simplex(((0, 0), (1, 0), (0, 1))),
        variables=(x, y),
        expected=sp.Rational(1, 24),
    )


def spherical_moments():
    return check(
        x * x,
        region=semialg.Sphere((0, 0, 0), 1),
        variables=(x, y, z),
        measure_dimension="intrinsic",
        expected=4 * sp.pi / 3,
    )


def mixed_radial():
    q = x * x + y * y + z * z
    result = integrate_result(
        sp.exp(-q) / (2 + q), region=semialg.Ball((0, 0, 0), 1), variables=(x, y, z)
    )
    assert result.absolute_convergent is True
    with mp.workdps(35):
        oracle = 4 * mp.pi * mp.quad(lambda r: r * r * mp.exp(-r * r) / (2 + r * r), [0, 1])
        assert abs(mp.mpf(str(result.value.evalf(30))) - oracle) < mp.mpf("1e-25")
    return result


def even_power():
    return check(
        sp.exp(-(x**4) - y * y), *plane, expected=sp.gamma(sp.Rational(1, 4)) * sp.sqrt(sp.pi) / 2
    )


def layer_cake():
    # Sublevel area in the positive quadrant is pi*sqrt(t)/4.
    value = sp.integrate(sp.exp(-t) * sp.pi * sp.sqrt(t) / 4, (t, 0, sp.oo))
    return check(sp.exp(-((x * x + y * y) ** 2)), (x, 0, sp.oo), (y, 0, sp.oo), expected=value)


def pushforward():
    return check(
        sp.exp(-((x + y) ** 3)),
        (x, 0, sp.oo),
        (y, 0, sp.oo),
        expected=sp.gamma(sp.Rational(2, 3)) / 3,
    )


def intrinsic():
    return check(
        1, region=semialg.Point((2, 3)), variables=(x, y), measure_dimension="intrinsic", expected=1
    )


def graph():
    return check(
        1,
        region=sp.And(x >= 0, x <= 1, y >= 0, y <= 2 - x * x),
        variables=(x, y),
        expected=sp.Rational(5, 3),
    )


def quadratic_clip():
    return check(
        1, region=sp.And(x * x + y * y <= 1, x >= 0, y >= 0), variables=(x, y), expected=sp.pi / 4
    )


def bernstein():
    assert interval_nonnegative(x * x - x + 1, x, 0, 1, strict_interior=True)
    return True


def parameter_cells():
    result = integrate_cells(
        1, region=x * x <= a, variables=(x,), bounds={x: (-1, 1)}, parameters=(a,)
    )
    assert result.complete and result.value.subs(a, sp.Rational(1, 4)) == 1
    return result


def compact_proof():
    assert compact_continuity(sp.exp(x * y), unit)
    return True


def cusp():
    return check(
        (x**3 - y * y) ** -sp.Rational(1, 2),
        region=sp.And(x > 0, x < 1, y * y <= x**3),
        variables=(x, y),
        expected=sp.pi,
    )


def shear():
    result = check(
        sp.exp(-(x**4) - (y - x * x) ** 2),
        *plane,
        expected=sp.gamma(sp.Rational(1, 4)) * sp.sqrt(sp.pi) / 2,
    )
    assert result.method == "nonseparable_improper" and result.absolute_convergent is True
    return result


def entire_series():
    result = evaluate_entire_box(sp.exp(x * y), *unit, tolerance=sp.Rational(1, 10**12))
    assert result.converged and result.certified and result.error_bound <= sp.Rational(1, 10**12)
    return result


def reflection():
    return check(sp.sin(x + y), (x, -1, 1), (y, -1, 1), expected=0)


def principal_value():
    return check(1 / x, (x, -1, 1), principal_value=True, expected=0)


def finite_part():
    result = regularized_result(x**-2, (x, -1, 1), regularization=HadamardFinitePart())
    assert result.value == -2
    return result


def shared_cutoff():
    polygon = sp.And(x >= 0, x <= sp.Rational(1, 2), y >= 0, x + y <= 1)
    result = regularized_result(
        1 / (x * y),
        region=polygon,
        variables=(x, y),
        regularization=HadamardFinitePart(hypersurface=x * y),
    )
    assert sp.simplify(result.value + sp.pi**2 / 12 - sp.log(2) ** 2) == 0
    return result


def symbolic_fallback():
    return check(sp.log(2 + x), (x, 0, 1), expected=3 * sp.log(3) - 2 * sp.log(2) - 1)


def indicators():
    indicator = check(Boole(x + y <= 1), *unit, expected=sp.Rational(1, 2))
    check(sp.Piecewise((y, y <= 1 - x), (x, True)), *unit, expected=sp.Rational(1, 2))
    return indicator


def exponential_boundary():
    return check(sp.exp(x + y), region=triangle, variables=(x, y), expected=1)


def polynomial_boundary():
    region = sp.And(x >= 0, y >= 0, z >= 0, x + y + z <= 1)
    return check(x * y * z, region=region, variables=(x, y, z), expected=sp.Rational(1, 720))


def numerical_check():
    result = integrate_result(x * x, (x, 0, 1))
    verification = verify_finite_result_numerically(result, x * x, (x, 0, 1), digits=25)
    assert verification.consistent is True and verification.certified is False
    return verification


def coarea():
    affine = integrate_coarea(sp.exp(-x - y), level_map=x + y, variables=(x, y), ranges=unit)
    assert sp.simplify(affine.value - (1 - sp.exp(-1)) ** 2) == 0
    radial = integrate_coarea(
        sp.exp(-x * x - y * y), level_map=x * x + y * y, variables=(x, y), level_bounds=(0, sp.oo)
    )
    assert radial.value == sp.pi
    chart = CoareaChart((t, u), (t + u * u, u), (x - y * y, y), ((u, 0, 1), (t, 0, 1)))
    region = sp.And(y >= 0, y <= 1, x - y * y >= 0, x - y * y <= 1)
    supplied = integrate_coarea(
        x, level_map=x - y * y, variables=(x, y), region=region, charts=(chart,)
    )
    assert supplied.value == sp.Rational(5, 6)
    from semialg.chart_certificates import polar_chart

    q = x * x + y * y
    angular = integrate_coarea(
        x * x * sp.exp(-q), level_map=q, variables=(x, y), level_bounds=(0, sp.oo)
    )
    assert angular.value == sp.pi / 2 and angular.absolute_convergent
    polar = integrate_coarea(
        x * x,
        level_map=q,
        variables=(x, y),
        region=q <= 1,
        charts=(polar_chart((x, y), (t, u), (0, 1), squared=True),),
    )
    assert sp.simplify(polar.value - sp.pi / 4) == 0
    dispatched = integrate_result(sp.exp(-((x + y) ** 3)), *unit)
    assert dispatched.method == "coarea_affine" and dispatched.absolute_convergent
    from multiple_integrate.coarea import integrate_coarea_cells
    from multiple_integrate.parameter_cells import ParameterCell

    vector = integrate_coarea(
        x + y + z,
        level_map=(2 * x, -3 * y),
        variables=(x, y, z),
        ranges=((z, 0, 1), (y, 0, 1), (x, 0, 1)),
    )
    assert vector.value == sp.Rational(3, 2) and vector.absolute_convergent
    fourier = integrate_coarea(
        sp.cos(sp.atan2(y, x)) ** 2, level_map=q, variables=(x, y), level_bounds=(0, 1)
    )
    assert fourier.value == sp.pi / 2 and fourier.absolute_convergent
    reciprocal = integrate_coarea(
        1 / (3 + x / sp.sqrt(q)), level_map=q, variables=(x, y), level_bounds=(0, 1)
    )
    assert sp.simplify(reciprocal.value - sp.pi / sp.sqrt(8)) == 0
    rate = sp.Symbol("coarea_rate", real=True)
    cells = (ParameterCell(rate < 0), ParameterCell(sp.Eq(rate, 0)), ParameterCell(rate > 0))
    parameterized = integrate_coarea_cells(
        x + y, cells=cells, parameters=(rate,), level_map=rate * x, variables=(x, y), ranges=unit
    )
    assert parameterized.complete and all(c.value == 1 for c in parameterized.cases)
    return (
        affine,
        radial,
        supplied,
        angular,
        polar,
        dispatched,
        vector,
        fourier,
        reciprocal,
        parameterized,
    )


METHOD_EXAMPLES = {
    "Iterated integration, product separation and inactive coordinates": iterated,
    "Finite polynomial box moments": box_moments,
    "Polynomial–exponential and polynomial–trigonometric recurrences": exp_trig,
    "Polynomial Gaussian recurrences and shifted/truncated moments": gaussian_recurrence,
    "Full-space quadratic Gaussian and halfspace reductions": quadratic_gaussian,
    "Beta, Mellin and rational power families": beta_mellin,
    "Laplace, stretched-exponential, endpoint and Laurent moments": laplace,
    "Pole-free linear/quadratic and selected quartic rational formulas": rational,
    "Reciprocal-phase and periodic trigonometric tails": oscillatory,
    "Simplex and affine-simplex moments": simplex,
    "Balls, shells, ellipsoids and spherical polynomial moments": spherical_moments,
    "Mixed radial profiles and half-integer moments": mixed_radial,
    "Weighted even-power and superellipse moments": even_power,
    "Layer-cake and level-set integration": layer_cake,
    "Linear pushforwards, monotone and piecewise changes": pushforward,
    "Intrinsic point, curve and surface measure": intrinsic,
    "Graph strips, affine cells and quadratic parameter geometry": graph,
    "Clipped polygons, parallel quadratic clips and angular sectors": quadratic_clip,
    "Bernstein interval positivity certificates": bernstein,
    "Automatic transition cells and caller-supplied cells": parameter_cells,
    "Compact continuity, products, sums and comparison proofs": compact_proof,
    "Endpoint vanishing, cusp horns and logarithmic tangent orders": cusp,
    "Nonseparable full-space shear profiles": shear,
    "Uniform entire-box series and bounded Taylor evaluation": entire_series,
    "Reflection cancellation": reflection,
    "Cauchy, geometric and hypersurface principal values": principal_value,
    "Hadamard finite parts, rational finite parts at infinity": finite_part,
    "Shared product cutoffs, affine/graph intersections and tangent regulators": shared_cutoff,
    "Fallback symbolic integration": symbolic_fallback,
    "Boolean indicators and certified region restrictions": indicators,
    "Affine exponential boundary flux": exponential_boundary,
    "Polynomial boundary flux on polygons and 3D polytopes": polynomial_boundary,
    "Numerical consistency checks": numerical_check,
    "Verified affine, radial and certified-chart coarea reductions": coarea,
}


if __name__ == "__main__":
    for method, example in METHOD_EXAMPLES.items():
        example()
        print(method, ": passed", flush=True)
