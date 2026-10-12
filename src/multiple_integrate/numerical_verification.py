"""Independent numerical consistency checks for finite continuous box integrals."""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
import sympy as sp

from .convergence_evidence import compact_continuity
from .results import IntegrationResult, IntegrationStatus


@dataclass(frozen=True)
class NumericalVerification:
    """A quadrature consistency check, explicitly not a rigorous certificate."""

    consistent: bool | None
    numerical_value: sp.Expr | None = None
    absolute_error: sp.Expr | None = None
    refinement_error: sp.Expr | None = None
    digits: int = 40
    strength: str = "numerical_consistency"
    certified: bool = False
    limitations: tuple[str, ...] = ("quadrature estimates have no rigorous error bound",)


def verify_finite_result_numerically(result, f, *ranges, digits=40, substitutions=None):
    """Check an ordinary result on a proved continuous finite 1-D or 2-D box.

    Explicit substitutions resolve parameters. Unsupported prerequisites return
    an inconclusive check; PV/finite-part values require their own prescriptions.
    """
    if not isinstance(result, IntegrationResult):
        raise TypeError("result must be an IntegrationResult")
    if type(digits) is not int or not 20 <= digits <= 200:
        raise ValueError("digits must be an integer between 20 and 200")

    def unsupported(reason):
        return NumericalVerification(None, digits=digits, strength="none", limitations=(reason,))

    if result.status not in (IntegrationStatus.EXACT, IntegrationStatus.CONDITIONAL):
        return unsupported(
            "only ordinary exact or explicitly specialized conditional results are supported"
        )
    mapping = (
        {}
        if substitutions is None
        else {sp.sympify(k): sp.sympify(v) for k, v in substitutions.items()}
    )
    if not 1 <= len(ranges) <= 2 or any(len(r) != 3 for r in ranges):
        return unsupported("one or two independent finite ranges are required")
    coords = tuple(r[0] for r in ranges)
    if len(set(coords)) != len(coords) or any(
        not isinstance(v, sp.Symbol) or v.is_real is not True for v in coords
    ):
        return unsupported("distinct real coordinate symbols are required")
    if set(coords) & mapping.keys():
        raise ValueError("substitutions must not replace integration coordinates")
    limits = tuple(
        (
            v,
            sp.sympify(lo).subs(mapping, simultaneous=True),
            sp.sympify(hi).subs(mapping, simultaneous=True),
        )
        for v, lo, hi in ranges
    )
    expr = sp.sympify(f).subs(mapping, simultaneous=True)
    value = result.value.subs(mapping, simultaneous=True)
    if value.free_symbols or expr.free_symbols - set(coords) or value.is_finite is not True:
        return unsupported("all parameters and a finite claimed value must be resolved")
    if any(
        lo.free_symbols or hi.free_symbols or (hi - lo).is_positive is not True
        for _, lo, hi in limits
    ):
        return unsupported("finite ordered independent endpoints are required")
    if not compact_continuity(expr, limits):
        return unsupported("continuity on the closed box was not proved")

    def quadrature(precision):
        with mp.workdps(precision):
            endpoints = tuple(
                (mp.mpf(str(lo.evalf(precision))), mp.mpf(str(hi.evalf(precision))))
                for _, lo, hi in limits
            )

            def integrate_axis(index, arguments):
                if index == len(coords):
                    return function(*arguments)
                lo, hi = endpoints[index]
                return mp.quad(
                    lambda t: integrate_axis(index + 1, (*arguments, t)),
                    [lo, hi],
                    method="gauss-legendre",
                )

            return integrate_axis(0, ())

    try:
        function = sp.lambdify(coords, expr, modules="mpmath")
        coarse = quadrature(digits + 10)
        fine = quadrature(digits + 20)
        with mp.workdps(digits + 20):
            claimed = complex_parts = sp.N(value, digits + 20)
            claimed = mp.mpc(str(sp.re(complex_parts)), str(sp.im(complex_parts)))
            error = abs(fine - claimed)
            refinement = abs(fine - coarse)
            tolerance = mp.mpf(10) ** (-digits) * (1 + abs(fine))
            consistent = bool(error <= tolerance and refinement <= tolerance)
            numeric = sp.Float(str(mp.re(fine)), digits) + sp.I * sp.Float(str(mp.im(fine)), digits)
            return NumericalVerification(
                consistent,
                numeric,
                sp.Float(str(error), digits),
                sp.Float(str(refinement), digits),
                digits,
            )
    except (ValueError, TypeError, ArithmeticError, NotImplementedError, NameError):
        return unsupported("quadrature or numerical evaluation failed")
