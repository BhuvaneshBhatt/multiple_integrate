"""Typed lazy adapter for :mod:`funcprops` reasoning.

MultipleIntegrate owns integration strategy.  This module is the only place the
package talks to funcprops, which owns reusable function-property decisions.
"""

from __future__ import annotations

from enum import Enum

import sympy as sp

_QUERY_ERRORS = (ImportError, AttributeError, NotImplementedError, sp.PolynomialError)


def _module():
    try:
        import funcprops
    except ImportError:
        return None
    return funcprops


def funcprops_available() -> bool:
    """Return whether the funcprops backend can be imported."""
    return _module() is not None


def _primary(value):
    return getattr(value, "primary", getattr(value, "classification", value))


def _enum_value(value):
    value = _primary(value)
    return value.value if isinstance(value, Enum) else value


def _call(name: str, *args, **kwargs):
    module = _module()
    if module is None:
        return None
    query = getattr(module, name.removeprefix("function_"), None)
    if not callable(query):
        query = getattr(module, name, None)
    if not callable(query):
        return None
    try:
        return query(*args, **kwargs)
    except _QUERY_ERRORS:
        return None


def _domain_arg(domain):
    return True if domain is None else domain


def _assumption_arg(assumptions):
    if assumptions is None:
        return True
    predicate_expr = getattr(assumptions, "predicate_expr", None)
    return predicate_expr if predicate_expr is not None else assumptions


def _range_constraints(domain, var):
    if domain is None or domain is True or domain is sp.true:
        return True
    if isinstance(domain, sp.Interval):
        lower = var > domain.start if domain.left_open else var >= domain.start
        upper = var < domain.end if domain.right_open else var <= domain.end
        return sp.And(lower, upper)
    if isinstance(domain, sp.Set):
        return sp.Contains(var, domain)
    return sp.sympify(domain)


def function_symmetry(expr: object, var: sp.Symbol, *, domain=None, assumptions=None) -> str | None:
    kwargs = {"domain": _domain_arg(domain), "assumptions": _assumption_arg(assumptions)}
    result = _call("function_symmetry", sp.sympify(expr), var, **kwargs)
    value = _enum_value(result)
    return value if isinstance(value, str) else None


def function_sign(
    expr: object, var: sp.Symbol | None = None, *, domain=None, assumptions=None
) -> str | None:
    args = [sp.sympify(expr)]
    if var is not None:
        args.append(var)
    kwargs = {"domain": _domain_arg(domain), "assumptions": _assumption_arg(assumptions)}
    result = _call("function_sign", *args, **kwargs)
    value = _enum_value(result)
    return value if isinstance(value, str) else None


def function_range(expr: object, var: sp.Symbol, *, domain=None, assumptions=None):
    image_var = sp.Dummy("range_value", real=True)
    result = _call(
        "function_range",
        sp.sympify(expr),
        var,
        image_var,
        constraints=_range_constraints(domain, var),
        assumptions=_assumption_arg(assumptions),
    )
    result = _primary(result)
    if isinstance(result, sp.Set):
        return result
    if isinstance(result, sp.Basic) and result.has(image_var):
        try:
            return result.as_set()
        except (NotImplementedError, TypeError, ValueError):
            return result
    return result


def function_monotonicity(
    expr: object, var: sp.Symbol, *, domain=None, assumptions=None
) -> str | None:
    kwargs = {"domain": _domain_arg(domain), "assumptions": _assumption_arg(assumptions)}
    result = _call("function_monotonicity", sp.sympify(expr), var, **kwargs)
    value = _enum_value(result)
    return value if isinstance(value, str) else None


def function_domain(expr: object, var: sp.Symbol, *, assumptions=None):
    return _call(
        "function_domain",
        sp.sympify(expr),
        var,
        assumptions=_assumption_arg(assumptions),
    )


def function_singularities(expr: object, var: sp.Symbol, *, domain=None, assumptions=None):
    kwargs = {
        "domain": _domain_arg(domain),
        "assumptions": _assumption_arg(assumptions),
        "return_result": True,
    }
    return _call("function_singularities", sp.sympify(expr), var, **kwargs)


def function_injective(
    expr: object, var: sp.Symbol, *, domain=None, assumptions=None
) -> bool | None:
    kwargs = {"domain": _domain_arg(domain), "assumptions": _assumption_arg(assumptions)}
    result = _call("function_injective", sp.sympify(expr), var, **kwargs)
    value = _enum_value(result)
    if value in (True, "true"):
        return True
    if value in (False, "false"):
        return False
    return None


__all__ = [
    "funcprops_available",
    "function_domain",
    "function_injective",
    "function_monotonicity",
    "function_range",
    "function_sign",
    "function_singularities",
    "function_symmetry",
]
