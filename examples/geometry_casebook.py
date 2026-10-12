"""Executable public-API examples shared by the README, guides, and demo."""

from dataclasses import replace

import mpmath as mp
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


def summary(result):
    """A small presentation table; inspect the evidence objects for full details."""
    return {
        "value": str(result.value),
        "outcome": result.outcome.value,
        "complete": result.complete,
        "absolute_convergent": result.absolute_convergent,
        "coverage": [(p.scope, p.complete, p.method) for p in result.coverage],
    }


def parameter_geometry():
    # example-start: parameter
    x, y, a, b = sp.symbols("x y a b", real=True)
    result = integrate_cells(
        1,
        region=sp.And(x >= 0, x <= a, y >= 0, y <= b - x**2),
        variables=(x, y),
        parameters=(a, b),
        assumptions=sp.And(a > 0, b >= a**2),
        cells=(ParameterCell(b > a**2), ParameterCell(sp.Eq(b, a**2), {b: a**2})),
    )
    assert sp.simplify(result.cases[0].value - (a * b - a**3 / 3)) == 0
    assert sp.simplify(result.cases[1].value - 2 * a**3 / 3) == 0
    # example-end: parameter
    assert result.complete
    return result


def cusp_contact():
    # example-start: cusp
    x, y = sp.symbols("x y", real=True)
    base = x**3 - y**2
    horn = sp.And(x > 0, x < 1, y**2 <= x**3)
    result = integrate_result(base ** -sp.Rational(1, 2), region=horn, variables=(x, y))
    divergent = integrate_result(x**2 / base, region=horn, variables=(x, y))
    assert sp.simplify(sp.expand_func(result.value) - sp.pi) == 0
    assert result.absolute_convergent is True
    assert divergent.outcome is IntegrationOutcome.DIVERGENT
    # example-end: cusp
    cancelled = integrate_result(
        sp.expand(base**2) / base ** sp.Rational(5, 2), region=horn, variables=(x, y)
    )
    assert sp.simplify(sp.expand_func(cancelled.value) - sp.pi) == 0
    return result


def radial_series():
    # example-start: radial
    x, y, z = sp.symbols("x y z", real=True)
    rho = x**2 + y**2 + z**2
    result = integrate_result(
        sp.exp(-rho) / (2 + rho), region=semialg.Ball((0, 0, 0), 1), variables=(x, y, z)
    )
    assert result.value.has(sp.Sum) and result.absolute_convergent is True
    with mp.workdps(35):
        reference = 4 * mp.pi * mp.quad(lambda r: r**2 * mp.exp(-r * r) / (2 + r * r), [0, 1])
        assert abs(mp.mpf(str(result.value.evalf(30))) - reference) < mp.mpf("1e-25")
    # example-end: radial
    return result


def clipped_cutoff():
    # example-start: cutoff
    x, y = sp.symbols("x y", real=True)
    polygon = sp.And(x >= 0, y >= 0, x <= sp.Rational(1, 2), x + y <= 1)
    result = regularized_result(
        1 / (x * y),
        region=polygon,
        variables=(x, y),
        regularization=HadamardFinitePart(hypersurface=x * y),
    )
    assert sp.simplify(result.value + sp.pi**2 / 12 - sp.log(2) ** 2) == 0
    assert verify_regularization_result(result).valid
    assert not verify_regularization_result(replace(result, value=result.value + 1)).valid
    # example-end: cutoff
    g = x * (y - x**2)
    graph = sp.And(x >= 0, x <= sp.Rational(1, 2), y - x**2 >= 0, x + y - x**2 <= 1)
    transformed = regularized_result(
        1 / g, region=graph, variables=(x, y), regularization=HadamardFinitePart(hypersurface=g)
    )
    assert sp.simplify(transformed.value - result.value) == 0
    assert verify_regularization_result(transformed).valid
    return result


def evidence_example():
    # example-start: evidence
    result = cusp_contact()
    print(summary(result))
    for proof in result.convergence_evidence:
        print(proof.kind, proof.established, proof.condition, proof.method)
    # example-end: evidence


EXAMPLES = {
    "parameter": parameter_geometry,
    "cusp": cusp_contact,
    "radial": radial_series,
    "cutoff": clipped_cutoff,
}


def run():
    for name, example in EXAMPLES.items():
        result = example()
        if name == "cutoff":
            print(
                name,
                result.value,
                "prescription replay:",
                verify_regularization_result(result).valid,
            )
        else:
            print(name, summary(result))


if __name__ == "__main__":
    run()
