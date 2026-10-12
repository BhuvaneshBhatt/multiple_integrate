"""Installed-wheel contract spanning MultipleIntegrate and its sibling packages."""

from __future__ import annotations

import sympy as sp


def main(*, with_asymptotic: bool = False) -> None:
    import funcprops
    import semialg

    import multiple_integrate as mi

    if with_asymptotic:
        import asymptotic

        assert callable(asymptotic.multiseries)
    assert callable(
        getattr(funcprops, "singularities", None)
        or getattr(funcprops, "function_singularities", None)
    )
    assert callable(semialg.reduce_region_integral)

    x, y = sp.symbols("x y", real=True)
    from multiple_integrate.adapters import funcprops as properties

    assert properties.function_symmetry(x**2, x) == "even"
    assert properties.function_singularities(1 / x, x) is not None
    assert mi.integrate(x**2 * y, (x, 0, 1), (y, 0, 2)) == sp.Rational(2, 3)

    q = x**4 + (y - x * x) ** 2
    improper = mi.integrate_result(sp.exp(-q), (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
    assert sp.simplify(improper.value - sp.sqrt(sp.pi) * sp.gamma(sp.Rational(1, 4)) / 2) == 0
    assert improper.complete and improper.absolute_convergent

    checked = mi.integrate_result(x**2, (x, 0, 1))
    assert (
        mi.verify_finite_result_numerically(checked, x**2, (x, 0, 1), digits=25).consistent is True
    )
    assert mi.NumericalVerification(None).certified is False

    box = semialg.Box(((-1, 1), (-1, 1)))
    assert mi.integrate(x + y, region=box, variables=(x, y)) == 0

    regularized = mi.regularized_result(
        x**-2,
        (x, -1, 1),
        regularization=mi.HadamardFinitePart(),
    )
    assert regularized.value == -2
    if with_asymptotic:
        assert regularized.provenance.details["asymptotic_backend"] == "asymptotic.multiseries"
    assert mi.verify_regularization_result(regularized).valid

    domain = sp.And(x >= 0, x <= 1, y >= 0, y <= 1)
    assert mi.integrate(1, region=domain, variables=(x, y)) == 1

    conditional = mi.integrate_result(sp.exp(-sp.Symbol("a", real=True) * x), (x, 0, sp.oo))
    assert conditional.status is mi.IntegrationStatus.CONDITIONAL


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--with-asymptotic", action="store_true")
    main(with_asymptotic=parser.parse_args().with_asymptotic)
