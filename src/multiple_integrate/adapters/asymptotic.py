"""Lazy boundary to the optional asymptotic expansion package."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class AsymptoticCapabilities:
    """Asymptotic operations used by cutoff regularization."""

    multiseries: Callable[..., object] | None
    provider: str | None


def load_asymptotic_capabilities() -> AsymptoticCapabilities:
    """Load the public multiseries capability when the sibling is installed."""
    try:
        module = importlib.import_module("asymptotic")
    except ImportError:
        return AsymptoticCapabilities(None, None)
    operation = getattr(module, "multiseries", None)
    if callable(operation):
        return AsymptoticCapabilities(operation, "asymptotic.multiseries")
    return AsymptoticCapabilities(None, "asymptotic")


def cutoff_expansion(
    expr: sp.Expr, epsilon: sp.Symbol, *, force_sympy: bool = False
) -> tuple[sp.Expr, str]:
    """Expand an exact cutoff through *inclusive* algebraic order zero.

    The optional ``asymptotic`` sibling is preferred in order-targeted mode:
    it must establish a complete Laurent--logarithmic prefix through order 0.
    Unlike a fixed ``terms`` count, this includes all negative powers and all
    logarithmic terms at order zero, without requesting unnecessary positive
    powers of the regulator.

    If the sibling is unavailable, incompatible, incomplete, or refuses the
    expansion, fall back to SymPy on the *original* shared-cutoff expression.
    ``force_sympy`` supports deterministic certificate replay of that fallback.
    This layer does not itself certify the finite-part prescription or the
    analytic remainder: the caller is responsible for those conditions.
    """
    expression = sp.sympify(expr)
    if not force_sympy:
        capabilities = load_asymptotic_capabilities()
        if capabilities.multiseries is not None:
            try:
                result = capabilities.multiseries(
                    expression,
                    epsilon,
                    point=0,
                    order=0,
                    return_result=True,
                )
                # Term-count prefixes (including SymPy Expr) are deliberately
                # not trusted here: they provide no order-zero completeness
                # witness. An older provider lacking `order` raises TypeError.
                complete = getattr(result, "is_complete_through", None)
                truncate = getattr(result, "truncate", None)
                if callable(complete) and callable(truncate) and complete(0):
                    expansion = sp.sympify(truncate())
                    if not expansion.has(sp.Integral, sp.Limit, sp.Order, sp.nan):
                        return expansion, capabilities.provider or "asymptotic"
            except (
                TypeError,
                ValueError,
                NotImplementedError,
                TimeoutError,
                sp.PoleError,
                sp.SympifyError,
                RecursionError,
            ):
                # An inconclusive optional expansion must not suppress a
                # supported evaluation by the independent SymPy fallback.
                pass

    try:
        return sp.series(expression, epsilon, 0, 1).removeO(), "sympy.series_fallback"
    except (TypeError, ValueError, NotImplementedError, sp.PoleError):
        return sp.expand(expression), "sympy.expand_fallback"


__all__ = ["AsymptoticCapabilities", "cutoff_expansion", "load_asymptotic_capabilities"]
