"""Convergence consensus, bounded entire moments and parameter geometry."""

import sympy as sp

from multiple_integrate import ParameterCell, integrate_cells, integrate_result
from multiple_integrate.series_evaluation import evaluate_entire_box


def examples():
    x, y, a, b = sp.symbols("x y a b", real=True)
    point = integrate_cells(
        sp.Piecewise((1 / a, a > 0), (0, True)),
        region=sp.Eq(x, a),
        variables=(x,),
        parameters=(a,),
        cells=[ParameterCell(a >= 0)],
        assumptions=a >= 0,
        measure_dimension="intrinsic",
    )
    assert point.ordinary_convergent is True and point.absolute_convergent is True
    series = evaluate_entire_box(
        sp.exp(x * y), (x, 0, 1), (y, 0, 1), tolerance=sp.Rational(1, 10**20)
    )
    assert series.converged and series.error_bound <= sp.Rational(1, 10**20)
    quartic = integrate_result(1 / (1 + (x * x + 2 * x + b) ** 2), (x, -sp.oo, sp.oo))
    assert quartic.method == "quartic_rational_moments"
    intervals = integrate_cells(1, region=sp.And(x >= a, x <= b), variables=(x,), parameters=(a, b))
    assert intervals.complete and intervals.value.subs({a: 1, b: 2}) == 1
    shear = integrate_cells(
        1,
        region=sp.And(x >= 0, x <= 1, y - a * x >= 0, y - a * x <= b),
        variables=(x, y),
        parameters=(a, b),
    )
    assert shear.complete and shear.value.subs({a: 2, b: 3}) == 3
    return point, series, quartic, intervals, shear


if __name__ == "__main__":
    for result in examples():
        print(result)
