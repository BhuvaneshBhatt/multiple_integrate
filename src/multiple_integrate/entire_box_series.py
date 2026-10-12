"""Exact uniformly convergent moments for nonseparable polynomial phases."""

from __future__ import annotations

from itertools import product

import sympy as sp

from .assumptions import _ask, _normalize_assumptions
from .cell_proofs import finite_external_parameters
from .convergence_evidence import _finite
from .fast_rules import _polynomial
from .planner import record_coverage
from .policy import _normalize_policy


def entire_box_series(expr, ranges, assumptions=None):
    """Integrate polynomial times exp/sin/cos of a polynomial on a finite box.

    Each monomial exponential supplies an independent nonnegative index. The
    absolute sum is uniformly bounded by exp(sum(abs(c_m*x**m))), justifying
    compact termwise moments and every Fubini order without truncation.
    """
    variables = tuple(r[0] for r in ranges)
    if len(variables) < 2 or any(sp.sympify(b).has(*variables) for r in ranges for b in r[1:]):
        return None
    context = _normalize_assumptions(assumptions)
    if not finite_external_parameters(expr, variables, context) or not all(
        _finite(sp.sympify(b), context) and _ask(sp.Q.real(b), context) is True
        for r in ranges
        for b in r[1:]
    ):
        return None
    specials = [f for f in sp.Mul.make_args(expr) if f.func in (sp.exp, sp.sin, sp.cos)]
    if len(specials) != 1:
        return None
    function = specials[0]
    amplitude = _polynomial(expr / function, variables)
    phase = _polynomial(function.args[0], variables)
    if (
        amplitude is None
        or phase is None
        or not all(_finite(c, context) for c in (*amplitude.coeffs(), *phase.coeffs()))
    ):
        return None
    # Preserve faster existing separable paths.
    if not any(sum(k != 0 for k in powers) > 1 for powers, _ in phase.terms()):
        return None
    constant = phase.coeff_monomial((0,) * len(variables))
    terms = [(powers, c) for powers, c in phase.terms() if any(powers)]
    budget = _normalize_policy(None)
    if not budget.allow_expensive_exact and (
        len(variables) > budget.semialg.max_variables
        or (function.func is not sp.exp and 2 ** len(terms) > budget.semialg.max_pieces)
    ):
        return None
    indices = tuple(sp.Dummy("phase_index", integer=True, nonnegative=True) for _ in terms)

    def exponential(multiplier, parity=None):
        summand = sp.S.Zero
        for powers, c in amplitude.terms():
            degrees = tuple(
                p + sp.Add(*(m[i] * k for (m, _), k in zip(terms, indices, strict=True)))
                for i, p in enumerate(powers)
            )
            moment = sp.prod(
                (sp.sympify(hi) ** (d + 1) - sp.sympify(lo) ** (d + 1)) / (d + 1)
                for (_, lo, hi), d in zip(ranges, degrees, strict=True)
            )
            summand += c * moment
        summand *= sp.prod(
            (multiplier * c) ** k / sp.factorial(k)
            for (_, c), k in zip(terms, indices, strict=True)
        )
        if parity is not None:
            # Separate even/odd real terms; avoids complex hypergeometric evalf.
            summand = summand.subs(
                {k: 2 * k + p for k, p in zip(indices, parity, strict=True)},
                simultaneous=True,
            ) * (-1) ** sum(indices)
            return sp.Sum(summand, *((k, 0, sp.oo) for k in indices))
        return sp.exp(multiplier * constant) * sp.Sum(summand, *((k, 0, sp.oo) for k in indices))

    if function.func is sp.exp:
        value = exponential(sp.S.One)
    else:
        trig = sp.sin if function.func is sp.sin else sp.cos
        value = sp.Add(
            *(
                trig(constant + sp.pi * sum(parity) / 2) * exponential(sp.S.One, parity)
                for parity in product((0, 1), repeat=len(indices))
            )
        )
    record_coverage(
        "integration_domain",
        True,
        "entire_box_series",
        basis="whole finite Cartesian box",
        indices=len(indices),
    )
    record_coverage(
        "absolute_convergence",
        True,
        "entire_box_series",
        basis="uniform absolute exponential-series majorant on a compact box",
    )
    return value
