"""Bounded exact parameter and Bernstein interval proofs without sampling fibers."""

from __future__ import annotations

import sympy as sp
from sympy.core.relational import Relational

from .assumptions import _ask, _condition_to_predicate, _normalize_assumptions
from .policy import _normalize_policy, check_semialg_budget


def _positive_atoms(context):
    for condition in context.original:
        for atom in sp.And.make_args(condition):
            if isinstance(atom, sp.AppliedPredicate):
                if atom.function in (sp.Q.positive, sp.Q.nonnegative):
                    yield atom.arguments[0], atom.function == sp.Q.positive
                elif atom.function in (sp.Q.negative, sp.Q.nonpositive):
                    yield -atom.arguments[0], atom.function == sp.Q.negative
                elif atom.function == sp.Q.zero:
                    yield atom.arguments[0], False
                    yield -atom.arguments[0], False
            elif isinstance(atom, Relational) and atom.rel_op == "==":
                yield sp.expand(atom.lhs - atom.rhs), False
                yield sp.expand(atom.rhs - atom.lhs), False
            elif isinstance(atom, Relational) and atom.rel_op in (">", ">=", "<", "<="):
                yield (
                    (
                        sp.expand(atom.lhs - atom.rhs)
                        if atom.rel_op in (">", ">=")
                        else sp.expand(atom.rhs - atom.lhs)
                    ),
                    atom.rel_op in (">", "<"),
                )


def nonnegative(expr, context, *, strict=False, _depth=0):
    """Use explicit cell inequalities and known nonnegative residuals."""
    if _depth > 8:
        return False
    expr = sp.sympify(expr)
    predicate = sp.Q.positive(expr) if strict else sp.Q.nonnegative(expr)
    # Strict cell proofs use explicit strict inequalities or intrinsic signs.
    # SymPy can incorrectly strengthen a weak bound at its equality fiber.
    intrinsic = expr.is_positive if strict else expr.is_nonnegative
    if intrinsic is True:
        return True
    roots = [r for r in expr.atoms(sp.Pow) if r.exp == sp.Rational(1, 2)]
    if len(roots) == 1:
        root = roots[0]
        coefficient = sp.expand(expr).coeff(root)
        remainder = sp.simplify(expr - coefficient * root)
        if (
            coefficient.is_number
            and _ask(sp.Q.real(root.base) & sp.Q.finite(root.base), context) is True
            and nonnegative(root.base, context, _depth=_depth + 1)
        ):
            if coefficient.is_negative is True and nonnegative(
                remainder, context, strict=True, _depth=_depth + 1
            ):
                if nonnegative(
                    sp.factor(remainder**2 - coefficient**2 * root.base),
                    context,
                    strict=strict,
                    _depth=_depth + 1,
                ):
                    return True
            elif (
                coefficient.is_positive is True
                and nonnegative(-remainder, context, strict=True, _depth=_depth + 1)
                and nonnegative(
                    sp.factor(coefficient**2 * root.base - remainder**2),
                    context,
                    strict=strict,
                    _depth=_depth + 1,
                )
            ):
                return True
    for difference, atom_strict in _positive_atoms(context):
        if difference != 0:
            ratio = sp.cancel(expr / difference)
            if ratio.is_number and ratio.is_positive is True and (not strict or atom_strict):
                return True
            if ratio.is_number and ratio.is_negative is True and (strict or atom_strict):
                return False
    numerator, denominator = sp.cancel(expr).as_numer_denom()
    if denominator != 1:
        if nonnegative(denominator, context, strict=True, _depth=_depth + 1) and nonnegative(
            numerator, context, strict=strict, _depth=_depth + 1
        ):
            return True
        if nonnegative(-denominator, context, strict=True, _depth=_depth + 1) and nonnegative(
            -numerator, context, strict=strict, _depth=_depth + 1
        ):
            return True
    if not strict and nonnegative(expr, context, strict=True, _depth=_depth + 1):
        return True
    if (
        strict
        and expr.is_Add
        and all(nonnegative(term, context, _depth=_depth + 1) for term in expr.args)
    ) and any(nonnegative(term, context, strict=True, _depth=_depth + 1) for term in expr.args):
        return True
    if strict and expr.is_Mul:
        signs = []
        for factor in expr.args:
            if nonnegative(factor, context, strict=True, _depth=_depth + 1):
                signs.append(1)
            elif -factor != expr and nonnegative(-factor, context, strict=True, _depth=_depth + 1):
                signs.append(-1)
            else:
                break
        if len(signs) == len(expr.args) and signs.count(-1) % 2 == 0:
            return True
    if strict and expr.is_Pow and expr.exp.is_real is True:
        if nonnegative(expr.base, context, strict=True, _depth=_depth + 1):
            return True
        if expr.exp.is_even is True and nonnegative(
            -expr.base, context, strict=True, _depth=_depth + 1
        ):
            return True
    for difference, atom_strict in _positive_atoms(context):
        residual = sp.expand(expr - difference)
        required = (
            sp.Q.nonnegative(residual) if not strict or atom_strict else sp.Q.positive(residual)
        )
        if (
            residual.is_positive is True
            if strict and not atom_strict
            else _ask(required, context) is True
        ):
            return True
    return not strict and _ask(predicate, context) is True


def nonzero(expr, context, *, _depth=0):
    """Exclude zero using strict signs, factor structure or explicit cell predicates."""
    expr = sp.sympify(expr)
    if _depth > 8:
        return False
    if expr.is_zero is False and expr.is_finite is True:
        return True
    if expr.func is sp.gamma:
        # Gamma has no zeros, but its poles must be excluded first.
        return finite_parameter_domain(expr, context)
    if expr.func is sp.exp:
        return finite_parameter_domain(expr.args[0], context)
    if expr.func is sp.log:
        argument = expr.args[0]
        return nonzero(argument, context, _depth=_depth + 1) and nonzero(
            argument - 1, context, _depth=_depth + 1
        )
    if expr.func is sp.cosh and _ask(sp.Q.real(expr.args[0]), context) is True:
        return finite_parameter_domain(expr.args[0], context)
    if expr.func is sp.Abs:
        return nonzero(expr.args[0], context, _depth=_depth + 1)
    atoms = tuple(atom for condition in context.original for atom in sp.And.make_args(condition))
    alternatives = [atom for atom in atoms if isinstance(atom, sp.Or)]
    if alternatives and sp.prod(len(atom.args) for atom in alternatives) <= 32:
        disjunction = alternatives[0]
        others = tuple(atom for atom in atoms if atom != disjunction)
        for branch in disjunction.args:
            try:
                branch_context = _normalize_assumptions((*others, branch))
            except ValueError:
                continue
            if not nonzero(expr, branch_context, _depth=_depth + 1):
                return False
        return True
    if nonnegative(expr, context, strict=True) or nonnegative(-expr, context, strict=True):
        return True
    for difference, strict in _positive_atoms(context):
        if strict and any(
            factor.is_Pow and factor.base in (expr, -expr) and factor.exp.is_positive is True
            for factor in sp.Mul.make_args(difference)
        ):
            # A strictly positive power cannot have a zero base. This retains
            # cells such as b**2 > 0 without inventing a sign for b itself.
            return True
    if expr.is_Mul:
        return all(nonzero(factor, context, _depth=_depth + 1) for factor in expr.args)
    if expr.is_Pow:
        return nonzero(expr.base, context, _depth=_depth + 1)
    for condition in context.original:
        for atom in sp.And.make_args(condition):
            difference = (
                atom.lhs - atom.rhs
                if isinstance(atom, sp.Unequality)
                else atom.arguments[0]
                if isinstance(atom, sp.AppliedPredicate) and atom.function == sp.Q.nonzero
                else None
            )
            if difference is not None and (
                sp.expand(expr - difference) == 0 or sp.expand(expr + difference) == 0
            ):
                return True
    return False


def finite_denominator(expr, context):
    """Check polynomial parameter denominators, leaving special-function proofs alone."""
    denominator = sp.sympify(expr).as_numer_denom()[1]
    if not denominator.free_symbols:
        return True
    base = denominator.base if denominator.is_Pow else denominator
    try:
        sp.Poly(base, *sorted(base.free_symbols, key=sp.default_sort_key))
    except (sp.PolynomialError, ValueError):
        return True
    return nonzero(base, context)


def finite_parameter_domain(expr, context):
    """Exclude exceptional parameter fibers of algebraic and recognized functions.

    This guards finiteness claims, not real branch selection. In particular a
    logarithm of a negative nonzero argument can have a finite complex value.
    Bound summation indices are not free parameters of the result.
    """
    expr = sp.sympify(expr)
    if not expr.free_symbols or expr.is_Symbol:
        return True
    if not finite_denominator(expr, context):
        return False
    parameters = expr.free_symbols
    for node in sp.preorder_traversal(expr):
        if not isinstance(node, sp.Expr) or not node.free_symbols & parameters:
            continue
        if node.is_Pow:
            base, exponent = node.args
            if (
                exponent.is_nonnegative is not True
                and not nonnegative(exponent, context)
                and not nonzero(base, context)
            ):
                return False
        elif node.func is sp.log:
            if not nonzero(node.args[0], context):
                return False
        elif node.func in (sp.tan, sp.sec, sp.cot, sp.csc, sp.tanh, sp.sech, sp.coth, sp.csch):
            argument = node.args[0]
            denominator = (
                sp.cos(argument)
                if node.func in (sp.tan, sp.sec)
                else sp.sin(argument)
                if node.func in (sp.cot, sp.csc)
                else sp.cosh(argument)
                if node.func in (sp.tanh, sp.sech)
                else sp.sinh(argument)
            )
            real_hyperbolic = _ask(sp.Q.real(argument), context) is True and (
                node.func in (sp.tanh, sp.sech)
                or (node.func in (sp.coth, sp.csch) and nonzero(argument, context))
            )
            if not real_hyperbolic and not nonzero(denominator, context):
                return False
        elif node.func in (sp.atanh, sp.acoth):
            if not all(nonzero(node.args[0] + sign, context) for sign in (-1, 1)):
                return False
        elif node.func in (sp.atan, sp.acot):
            if not all(nonzero(node.args[0] + sign * sp.I, context) for sign in (-1, 1)):
                return False
        elif node.func in (sp.Ei, sp.Ci, sp.Chi, sp.bessely, sp.besselk, sp.hankel1, sp.hankel2):
            if not nonzero(node.args[-1], context):
                return False
        elif node.func is sp.li:
            if not nonzero(node.args[0] - 1, context):
                return False
        elif node.func is sp.zeta:
            if not nonzero(node.args[0] - 1, context):
                return False
            if len(node.args) == 2:
                argument = node.args[1]
                if not (
                    nonnegative(argument, context, strict=True)
                    or argument.is_integer is False
                    or argument.is_real is False
                ):
                    return False
        elif node.func in (sp.gamma, sp.loggamma, sp.polygamma):
            argument = node.args[-1]
            if not (
                nonnegative(argument, context, strict=True)
                or nonnegative(sp.re(argument), context, strict=True)
                or argument.is_integer is False
                or _ask(sp.Q.integer(argument), context) is False
                or argument.is_real is False
            ):
                return False
    return True


def finite_external_parameters(expr, coordinates, context):
    """Keep parameter poles that evaluation can erase by symmetry or cancellation.

    Coordinate singularities are handled by integration/convergence strategies;
    only coordinate-independent subexpressions receive parameter-domain guards.
    Piecewise densities retain their branch assumptions rather than requiring
    every coefficient to be defined on unreachable parameter fibers.
    """
    expr = sp.sympify(expr)
    if not isinstance(expr, sp.Expr):
        return True
    if not expr.has(*coordinates) and not isinstance(expr, sp.Piecewise):
        return finite_parameter_domain(expr, context)
    if isinstance(expr, sp.Piecewise):
        remaining = sp.true
        for value, condition in expr.args:
            effective = sp.And(remaining, condition)
            branch = context
            if not effective.has(*coordinates):
                try:
                    branch = _normalize_assumptions(sp.And(*context.original, effective))
                except ValueError:
                    remaining = sp.And(remaining, sp.Not(condition))
                    continue
            if not finite_external_parameters(value, coordinates, branch):
                return False
            remaining = sp.And(remaining, sp.Not(condition))
        return True
    return all(finite_external_parameters(arg, coordinates, context) for arg in expr.args)


def native_predicate_true(predicate, assumptions):
    """Shared native strict-sign fast paths, retaining conjunction semantics."""
    context = _normalize_assumptions(assumptions)
    if predicate.func is sp.Not and isinstance(predicate.args[0], sp.AppliedPredicate):
        inner = predicate.args[0]
        if inner.function == sp.Q.zero:
            return nonzero(inner.arguments[0], context)
    if isinstance(predicate, sp.And):
        return all(native_predicate_true(part, context) for part in predicate.args)
    if isinstance(predicate, sp.AppliedPredicate):
        if predicate.function == sp.Q.nonzero:
            return nonzero(predicate.arguments[0], context)
        if predicate.function == sp.Q.zero and nonzero(predicate.arguments[0], context):
            return False
        if predicate.function == sp.Q.positive:
            return nonnegative(predicate.arguments[0], context, strict=True)
        if predicate.function == sp.Q.negative:
            return nonnegative(-predicate.arguments[0], context, strict=True)
    return _ask(predicate, context) is True


def proves(condition, assumptions=None):
    """Prove an implication with native algebra or an opt-in bounded backend call."""
    context = _normalize_assumptions(assumptions)
    if (not isinstance(condition, Relational) or condition.rel_op not in (">", "<")) and _ask(
        _condition_to_predicate(condition), context
    ) is True:
        return True
    if isinstance(condition, Relational) and condition.rel_op in (">", ">=", "<", "<="):
        difference = (
            sp.expand(condition.lhs - condition.rhs)
            if condition.rel_op in (">", ">=")
            else sp.expand(condition.rhs - condition.lhs)
        )
        if nonnegative(difference, context, strict=condition.rel_op in (">", "<")):
            return True
    if not context.original:
        return False
    formula = sp.And(*context.original, sp.Not(condition))
    variables = tuple(sorted(formula.free_symbols, key=sp.default_sort_key))
    if any(v.is_real is not True for v in variables):
        return False
    try:
        for atom in formula.atoms(Relational):
            sp.Poly(atom.lhs - atom.rhs, *variables)
    except (sp.PolynomialError, ValueError, TypeError):
        return False
    budget = _normalize_policy(None)
    allowed, _ = check_semialg_budget(formula, variables, budget)
    if not allowed or (len(variables) > 2 and not budget.allow_expensive_exact):
        return False
    # Spawn isolation bounds even native order proofs; no default unrestricted QE.
    from .isolated_queries import isolated_query

    result = isolated_query(
        "semialg", "is_satisfiable", (formula, variables), {}, budget.backend_timeout_seconds or 2.0
    )
    return result is False


def polynomial_sign(expr, context):
    """Certify a sign on the whole caller cell, declining unproved cases."""
    if _ask(sp.Q.zero(expr), context) is True:
        return 0
    if nonnegative(expr, context, strict=True):
        return 1
    if nonnegative(-expr, context, strict=True):
        return -1
    # Polynomial-only fallback; radicals must be reduced explicitly by the caller.
    try:
        sp.Poly(expr, *tuple(sorted(expr.free_symbols, key=sp.default_sort_key)))
    except (sp.PolynomialError, ValueError, TypeError):
        return None
    for condition, sign in ((sp.Eq(expr, 0), 0), (expr > 0, 1), (expr < 0, -1)):
        if proves(condition, context):
            return sign
    return None


def interval_nonnegative(expr, x, lo, hi, assumptions=None, *, strict_interior=False):
    """Bernstein coefficients prove positivity on a complete finite interval."""
    context = _normalize_assumptions(assumptions)
    t = sp.Dummy("bernstein_coordinate", real=True)
    try:
        poly = sp.Poly(sp.expand(expr.subs(x, lo + (hi - lo) * t)), t)
    except sp.PolynomialError:
        return False
    degree = 0 if poly.is_zero else int(poly.degree())
    budget = _normalize_policy(None)
    if not budget.allow_expensive_exact and degree > budget.semialg.max_polynomial_degree:
        return False
    coefficients = [
        sp.simplify(
            sum(poly.nth(k) * sp.binomial(j, k) / sp.binomial(degree, k) for k in range(j + 1))
        )
        for j in range(degree + 1)
    ]
    if not all(nonnegative(c, context) for c in coefficients):
        return False
    return not strict_interior or any(nonnegative(c, context, strict=True) for c in coefficients)
