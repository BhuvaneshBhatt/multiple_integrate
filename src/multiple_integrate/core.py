"""Core algorithms for exact multiple integration."""

from __future__ import annotations

from collections import OrderedDict

import sympy as sp
from sympy import (
    Heaviside,
    Piecewise,
    oo,
    sign,
)

from multiple_integrate.adapters import (
    reduce_domain,
    reduce_intrinsic_domain,
)
from multiple_integrate.analytic_strategies import (
    _const_result,
    _iterated_integrate,
    _region_shortcut,
    _try_additive_separable,
    _try_fullspace_quadratic,
    _try_gauss_linear_map,
    _try_general_layercake,
    _try_graph_reversal,
    _try_halfspace_quadratic,
    _try_linear_pushforward,
    _try_monotone_change,
    _try_piecewise_change,
    _try_poly_layercake,
    _try_product_separable,
    _try_superellipse,
)
from multiple_integrate.assumptions import (
    _CURRENT_ASSUMPTIONS,
    _ask,
    _AssumptionContext,
    _integrate,
    _normalize_assumptions,
)
from multiple_integrate.cauchy_principal_value import iterated_pv, one_dimensional_pv
from multiple_integrate.convergence import specialized_value
from multiple_integrate.geometric_principal_value import geometric_pv
from multiple_integrate.hadamard import hadamard_finite_part
from multiple_integrate.hypersurface_principal_value import hypersurface_pv
from multiple_integrate.planner import (
    convergence_trace,
    coverage_trace,
    isolate_nested_coverage,
    record_convergence,
    record_coverage,
    record_strategy,
)
from multiple_integrate.policy import (
    _CURRENT_POLICY,
    _normalize_policy,
    enforce_semialg_budget,
    enforce_semialg_piece_budget,
)
from multiple_integrate.principal_value_prescriptions import (
    CauchyPrincipalValue,
    GeometricPrincipalValue,
    HypersurfacePrincipalValue,
    IteratedPrincipalValue,
    _normalize_principal_value,
)
from multiple_integrate.recognition import (
    _cached_decomposition,
    _decompose_integrand,
    _fast_simplify,
    _IntegrandDecomposition,
    _is_constant_wrt,
    _normalize_seq,
    _range_has_singularity,
    _split_additive_terms,
    _split_superellipse,
    _validate_ranges,
    _vars_set,
)
from multiple_integrate.region_formulas import indicator_condition, restrict_region
from multiple_integrate.region_recognition import region_from_ranges
from multiple_integrate.region_types import (
    AffineSimplexRegion,
    BallRegion,
    BoxRegion,
    DiskRegion,
    EllipsoidRegion,
    Region,
    SimplexRegion,
)
from multiple_integrate.regularization import HadamardFinitePart
from multiple_integrate.results import UnresolvedIntegration
from multiple_integrate.semialg_geometry import domain_geometry
from multiple_integrate.singularities import split_integral
from multiple_integrate.transforms import _try_region_transform


def _recorded(method: str, value):
    """Record the selected planner route and return *value* unchanged."""
    record_strategy(method)
    if method in {
        "constant_volume",
        "region_shortcut",
        "region_transform",
        "gaussian_linear_map",
        "iterated_fallback",
    }:
        record_coverage(
            "integration_domain",
            True,
            method,
            basis="full-domain strategy under active assumptions",
        )
    elif method == "cache":
        record_coverage(
            "problem_identity", True, method, basis="identical integrand, region and assumptions"
        )
    return value


_STRATEGY_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    ZeroDivisionError,
    OverflowError,
    sp.PolynomialError,
)


_CACHE_MAXSIZE = 512


def clear_cache() -> None:
    """Clear MultipleIntegrate's bounded process-local result cache."""
    cache = getattr(integrate, "_cache", None)
    if cache is not None:
        cache.clear()
    _cached_decomposition.cache_clear()
    from .region_recognition import _recognize_ranges

    _recognize_ranges.cache_clear()
    from .boundary_reduction import clear_boundary_cache

    clear_boundary_cache()
    from .coarea import clear_coarea_cache

    clear_coarea_cache()


def _geometry_orient_sign(
    region: Region, ranges: list[tuple], assumptions: _AssumptionContext
) -> sp.Integer | None:
    """Return the proved orientation sign for explicit structured ranges."""
    variables = {entry[0] for entry in ranges}
    sign_value = sp.Integer(1)
    found = False
    for _var, lower, upper in ranges:
        lower_expr = sp.sympify(lower)
        upper_expr = sp.sympify(upper)
        if (lower_expr.free_symbols | upper_expr.free_symbols) & variables:
            continue
        found = True
        width = sp.simplify(upper_expr - lower_expr)
        if _ask(sp.Q.positive(width), assumptions) is True:
            continue
        if _ask(sp.Q.negative(width), assumptions) is True:
            sign_value = -sign_value
            continue
        return None
    if not found:
        return None
    # For radial quadratic regions, dependent square-root widths use their
    # principal nonnegative branch, so the independent-bound sign captures
    # the orientation.  Simplices can have additional affine orientation
    # factors; only their positive orientation is accepted here.
    if isinstance(region, (SimplexRegion, AffineSimplexRegion)) and sign_value != 1:
        return None
    return sign_value


def _apply_symmetry(
    expr: sp.Expr,
    region: Region,
    ranges: list[tuple],
) -> tuple[sp.Expr | None, list[tuple], Region, sp.Expr]:
    """Apply safe odd cancellation and even-range halving before transforms."""
    reduced_ranges = list(ranges)
    scale = sp.Integer(1)
    for index, (variable, _lower, _upper) in enumerate(reduced_ranges):
        symmetric = region.symmetric_range(variable)
        if symmetric is None:
            continue
        lower, upper = symmetric
        if _range_has_singularity(expr, variable, lower, upper):
            continue
        try:
            reflected = expr.subs(variable, -variable)
            if sp.simplify(reflected + expr) == 0:
                return sp.Integer(0), reduced_ranges, region, scale
            if sp.simplify(reflected - expr) == 0:
                reduced_ranges[index] = (variable, sp.Integer(0), upper)
                scale *= 2
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            continue
    if scale != 1:
        region = region_from_ranges(reduced_ranges, structural_order="inner-first")
    return None, reduced_ranges, region, scale


def _cache_result(cache_key, result, *, divergent=False):
    cache = integrate._cache
    cache.pop(cache_key, None)
    cache[cache_key] = (result, divergent, convergence_trace(), coverage_trace())
    while len(cache) > _CACHE_MAXSIZE:
        cache.popitem(last=False)
    return result


def _integrate_piecewise(
    expr: sp.Piecewise,
    region: Region,
    ranges: list[tuple],
    assumptions: _AssumptionContext,
    principal_value: bool,
    opts: dict,
) -> sp.Expr:
    """Integrate Piecewise branches over representable region restrictions."""
    if expr.args[-1][1] is not sp.true:
        # Piecewise is undefined outside its listed conditions. Domain restriction
        # sums must not silently replace that missing part by zero.
        try:
            direct = sp.sympify(_integrate(expr, *ranges, **opts))
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            direct = None
        if direct is None or direct.has(sp.Integral, sp.nan, sp.zoo):
            return sp.Integral(expr, *ranges)
        return _fast_simplify(direct)
    total = sp.Integer(0)
    remaining = sp.true
    for branch_expr, branch_cond in expr.args:
        effective = (
            remaining
            if branch_cond in (True, sp.true)
            else sp.simplify(sp.And(remaining, branch_cond))
        )
        if sp.sympify(branch_expr) == 0:
            if branch_cond not in (True, sp.true):
                remaining = sp.simplify(sp.And(remaining, sp.Not(branch_cond)))
            continue
        sub_region = restrict_region(region, effective)
        if sub_region is None:
            try:
                direct = _integrate(expr, *ranges, **opts)
            except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
                direct = None
            if direct is not None and not sp.sympify(direct).has(sp.Integral):
                return _fast_simplify(direct)
            return sp.Integral(expr, *ranges)
        total += integrate(
            sp.sympify(branch_expr),
            sub_region,
            assumptions=assumptions,
            principal_value=principal_value,
        )
        if branch_cond not in (True, sp.true):
            remaining = sp.simplify(sp.And(remaining, sp.Not(branch_cond)))
    return _fast_simplify(total)


def _try_indicator_product(
    expr: sp.Expr,
    region: Region,
    assumptions: _AssumptionContext,
    principal_value: bool,
) -> sp.Expr | None:
    """Convert multiplicative indicator factors into a narrow restriction."""
    if not expr.is_Mul:
        return None
    conditions = []
    remaining = []
    for factor in expr.args:
        condition = indicator_condition(factor)
        if condition is None:
            remaining.append(factor)
        else:
            conditions.append(condition)
    if not conditions:
        return None
    restricted = restrict_region(region, sp.And(*conditions))
    if restricted is None:
        return None
    new_expr = _fast_simplify(sp.Mul(*remaining) if remaining else sp.Integer(1))
    return integrate(
        new_expr,
        restricted,
        assumptions=assumptions,
        principal_value=principal_value,
    )


def _run_decomp_strategies(
    decomposition: _IntegrandDecomposition,
    variables: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """Run composition strategies from cheap/specific to general."""
    outer = decomposition.f_outer
    inner = decomposition.g_inner
    if decomposition.is_polynomial:
        # Every composition route below treats the inner polynomial as a real
        # level-set coordinate. Complex coefficients require contour geometry.
        coefficients = sp.Poly(inner, *variables).coeffs()
        if not all(_ask(sp.Q.real(c), opts.get("_mi_assumptions")) is True for c in coefficients):
            return None
        for strategy in (
            _try_linear_pushforward,
            _try_fullspace_quadratic,
            _try_halfspace_quadratic,
            _try_superellipse,
            _try_poly_layercake,
        ):
            result = strategy(outer, inner, variables, ranges, opts)
            if result is not None:
                return result
    for strategy in (
        _try_additive_separable,
        _try_monotone_change,
        _try_piecewise_change,
        _try_general_layercake,
    ):
        result = strategy(outer, inner, variables, ranges, opts)
        if result is not None:
            return result
    return None


def _normalize_evaluation_domain(ranges, assumptions):
    """Normalize ranges and assumptions once before strategy dispatch."""
    assumption_context = _normalize_assumptions(assumptions)
    normalized = _normalize_seq(ranges)
    direct_region = (
        normalized[0] if len(normalized) == 1 and isinstance(normalized[0], Region) else None
    )
    if direct_region is not None:
        if isinstance(direct_region, BoxRegion):
            from multiple_integrate.cell_proofs import nonnegative

            if not all(
                _ask(sp.Q.extended_real(sp.sympify(b)), assumption_context) is True
                for r in direct_region.ranges
                for b in r[1:]
            ):
                raise UnresolvedIntegration("geometric box endpoints must be proved extended real")
            normalized_bounds = []
            for v, lo, hi in direct_region.ranges:
                width = sp.sympify(hi) - sp.sympify(lo)
                if nonnegative(width, assumption_context):
                    normalized_bounds.append((v, lo, hi))
                elif nonnegative(-width, assumption_context):
                    normalized_bounds.append((v, hi, lo))
                else:
                    normalized_bounds.append((v, sp.Min(lo, hi), sp.Max(lo, hi)))
            direct_region = BoxRegion(tuple(normalized_bounds))
        parsed_ranges = list(direct_region.ranges)
    else:
        parsed_ranges = []
        for entry in normalized:
            if len(entry) != 3:
                raise ValueError(f"Each range must be (variable, lower, upper); got {entry}")
            parsed_ranges.append(tuple(entry))
        _validate_ranges(parsed_ranges)
    return assumption_context, parsed_ranges, direct_region


def _evaluate_prescription(
    expr: sp.Expr,
    parsed_ranges: list[tuple],
    direct_region: Region | None,
    assumptions: _AssumptionContext,
    principal_value,
    regularization,
):
    """Evaluate explicit regularization and principal-value requests."""
    if regularization is not None:
        if principal_value not in (False, None):
            raise ValueError("principal_value= and regularization= are mutually exclusive")
        if not isinstance(regularization, HadamardFinitePart):
            raise TypeError("regularization must be a HadamardFinitePart prescription")
        return "hadamard_finite_part", hadamard_finite_part(
            expr,
            *parsed_ranges,
            prescription=regularization,
            assumptions=assumptions,
        )

    pv = _normalize_principal_value(principal_value)
    if isinstance(pv, CauchyPrincipalValue):
        if direct_region is not None or len(parsed_ranges) != 1:
            raise NotImplementedError(
                "CauchyPrincipalValue requires one explicit one-dimensional range; "
                "use IteratedPrincipalValue or GeometricPrincipalValue for multiple integrals"
            )
        return "principal_value", one_dimensional_pv(
            expr, parsed_ranges[0], assumptions=assumptions
        )
    if isinstance(pv, IteratedPrincipalValue):
        return "iterated_principal_value", iterated_pv(
            expr, parsed_ranges, pv, assumptions=assumptions
        )
    return pv


def _region_principal_value(expr: sp.Expr, region: Region, pv, assumptions: _AssumptionContext):
    if isinstance(pv, GeometricPrincipalValue):
        return "geometric_principal_value", geometric_pv(expr, region, pv, assumptions=assumptions)
    if isinstance(pv, HypersurfacePrincipalValue):
        return "hypersurface_principal_value", hypersurface_pv(
            expr, region, pv, assumptions=assumptions
        )
    return None


def _integral_cache_key(
    expr: sp.Expr, region: Region, principal_value, assumptions: _AssumptionContext
):
    if not hasattr(integrate, "_cache"):
        integrate._cache = OrderedDict()
    return (
        _fast_simplify(expr),
        (type(region), region),
        bool(principal_value),
        assumptions.cache_key,
    )


def _cached_lookup(cache_key):
    if cache_key not in integrate._cache:
        return None
    value = integrate._cache.pop(cache_key)
    integrate._cache[cache_key] = value
    result, divergent, claims, covers = value
    for cover in covers:
        record_coverage(cover.scope, cover.complete, cover.method, **dict(cover.details))
    for claim in claims:
        record_convergence(claim.kind, claim.established, claim.method, claim.condition)
    if divergent:
        record_strategy(
            "divergent_interval_piece",
            reason="cached one-sided divergence evidence for the identical problem",
        )
    return result


def _pre_strategy_reduction(
    expr: sp.Expr,
    region: Region,
    parsed_ranges: list[tuple],
    assumptions: _AssumptionContext,
    principal_value,
    opts: dict,
    cache_key,
    direct_region: Region | None,
):
    """Apply cheap exact reductions before general integrand strategies."""
    variables = list(region.variables)
    variable_set = _vars_set(variables)

    if isinstance(expr, sp.Piecewise):
        return "piecewise", _integrate_piecewise(
            expr, region, parsed_ranges, assumptions, principal_value, opts
        )
    indicator_result = _try_indicator_product(expr, region, assumptions, principal_value)
    if indicator_result is not None:
        return "indicator", indicator_result

    orientation = (
        sp.Integer(1)
        if direct_region is not None
        else _geometry_orient_sign(region, parsed_ranges, assumptions)
    )
    if _is_constant_wrt(expr, variable_set):
        if orientation is not None:
            value = orientation * _const_result(
                expr, region, parsed_ranges, use_region_volume=True, assumptions=assumptions
            )
            return "constant_volume", _cache_result(cache_key, _fast_simplify(value))
        if isinstance(
            region,
            (SimplexRegion, AffineSimplexRegion, DiskRegion, BallRegion, EllipsoidRegion),
        ):
            return "constant_volume", _cache_result(cache_key, sp.Integral(expr, *parsed_ranges))
        return "constant_volume", _cache_result(
            cache_key,
            _const_result(
                expr, region, parsed_ranges, use_region_volume=False, assumptions=assumptions
            ),
        )

    if len(parsed_ranges) == 1 and not principal_value:
        from .fast_rules import fast_interval_rule

        v, lo, hi = parsed_ranges[0]
        fast = fast_interval_rule(expr, v, lo, hi, assumptions)
        if fast is not None:
            method, value = fast
            return method, _cache_result(cache_key, _fast_simplify(value))
    if orientation is not None:
        shortcut = _region_shortcut(region, expr, assumptions=assumptions)
        if shortcut is not None:
            return "region_shortcut", _cache_result(
                cache_key, _fast_simplify(orientation * shortcut)
            )

    from multiple_integrate.quadratic_radial import try_weighted_quadratic

    weighted = try_weighted_quadratic(expr, parsed_ranges, assumptions=assumptions)
    if weighted is not None:
        return "weighted_quadratic", _cache_result(cache_key, _fast_simplify(weighted))

    from multiple_integrate.anisotropic import try_anisotropic_moments

    anisotropic = try_anisotropic_moments(expr, parsed_ranges, assumptions=assumptions)
    if anisotropic is not None:
        return "anisotropic_moments", _cache_result(cache_key, anisotropic)

    from .nonseparable_improper import try_nonseparable_improper

    improper = try_nonseparable_improper(expr, parsed_ranges, assumptions)
    if improper is not None:
        return "nonseparable_improper", _cache_result(cache_key, improper)

    coarea_orientation = orientation
    if (
        coarea_orientation is None
        and parsed_ranges
        and all(lo == -sp.oo and hi == sp.oo for _, lo, hi in parsed_ranges)
    ):
        coarea_orientation = sp.S.One
    if (
        not principal_value
        and coarea_orientation is not None
        and not (set(opts) - {"_mi_assumptions"})
        and len(variables) > 1
    ):
        from .coarea import try_automatic_coarea

        coarea = try_automatic_coarea(expr, region, tuple(parsed_ranges), assumptions)
        if coarea is not None:
            method, value = coarea
            if method == "coarea_radial":
                value = coarea_orientation * value
            return method, _cache_result(cache_key, value, divergent=value in (sp.oo, -sp.oo))

    zero_result, parsed_ranges, region, scale = _apply_symmetry(expr, region, parsed_ranges)
    if zero_result is not None:
        return "symmetry", _cache_result(cache_key, zero_result)

    if orientation is not None:
        transformed = _try_region_transform(
            region, expr, assumptions=assumptions, principal_value=principal_value
        )
        if transformed is not None:
            value = scale * _fast_simplify(orientation * transformed)
            return "region_transform", _cache_result(cache_key, _fast_simplify(value))

    gaussian = _try_gauss_linear_map(expr, parsed_ranges, assumptions=assumptions)
    if gaussian is not None:
        return "gaussian_linear_map", _cache_result(
            cache_key, _fast_simplify(scale * gaussian) if scale != 1 else gaussian
        )

    reversed_result = _try_graph_reversal(expr, region, assumptions, principal_value)
    if reversed_result is not None:
        value = _fast_simplify(scale * reversed_result) if scale != 1 else reversed_result
        return "graph_reversal", _cache_result(cache_key, value)

    return expr, region, parsed_ranges, scale, variables, variable_set, orientation


def _fallback_integral(expr, parsed_ranges, opts, cache_key, scale):
    value = _iterated_integrate(expr, parsed_ranges, opts)
    if scale != 1:
        value = _fast_simplify(scale * value)
    return "iterated_fallback", _cache_result(cache_key, value)


def _dispatch_integrand_strategies(
    expr: sp.Expr,
    region: Region,
    parsed_ranges: list[tuple],
    variables: list[sp.Symbol],
    variable_set: set[sp.Symbol],
    assumptions: _AssumptionContext,
    principal_value,
    opts: dict,
    cache_key,
    scale: sp.Expr,
    orientation,
):
    """Run algebraic and decomposition strategies after cheap reductions."""

    def scaled(result):
        return _fast_simplify(scale * result) if scale != 1 else result

    terms = _split_additive_terms(expr)
    if terms is not None and all(
        sp.sympify(lower) not in (-oo, oo) and sp.sympify(upper) not in (-oo, oo)
        for _, lower, upper in parsed_ranges
    ):
        total = sp.Add(
            *(
                integrate(
                    term,
                    *parsed_ranges,
                    assumptions=assumptions,
                    principal_value=principal_value,
                )
                for term in terms
            )
        )
        return "additive_split", _cache_result(cache_key, scaled(_fast_simplify(total)))

    try:
        normalized_expr = sp.trigsimp(sp.powsimp(expr))
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        normalized_expr = expr
    if normalized_expr != expr:
        expr = normalized_expr
        if _is_constant_wrt(expr, variable_set):
            value = _const_result(
                expr,
                region,
                parsed_ranges,
                use_region_volume=orientation is not None,
                assumptions=assumptions,
            )
            if orientation is not None:
                value *= orientation
            return None, _cache_result(cache_key, scaled(value))

    step_functions = (Heaviside, Piecewise, sign)
    if isinstance(expr, step_functions) or (
        expr.is_Mul and any(isinstance(factor, step_functions) for factor in expr.args)
    ):
        return "step_function", _cache_result(
            cache_key, scaled(_iterated_integrate(expr, parsed_ranges, opts))
        )

    separable = _try_product_separable(expr, variables, parsed_ranges, opts)
    if separable is not None:
        return "product_separable", _cache_result(cache_key, scaled(separable))

    active_vars = expr.free_symbols & variable_set
    if active_vars != set(variables) and active_vars:
        active_ranges = [entry for entry in parsed_ranges if entry[0] in active_vars]
        from multiple_integrate.strategy_common import _inactive_finite_volume

        volume = _inactive_finite_volume(active_vars, variables, parsed_ranges, simplify=False)
        if volume is not None:
            inner = integrate(
                expr,
                *active_ranges,
                assumptions=assumptions,
                principal_value=principal_value,
            )
            return "inactive_dimension_factor", _cache_result(
                cache_key, scaled(_fast_simplify(volume * inner))
            )

    if len(parsed_ranges) == 1:
        var, lower, upper = parsed_ranges[0]
        if not principal_value:
            split = split_integral(
                expr, var, lower, upper, opts, assumptions=assumptions, return_evidence=True
            )
            if split is not None:
                value, divergent = split
                return "funcprops_singularity_split", _cache_result(
                    cache_key, scaled(value), divergent=divergent
                )
        return _fallback_integral(expr, parsed_ranges, opts, cache_key, scale)

    if expr.is_Add:
        constant_limits = [
            entry
            for entry in parsed_ranges
            if not sp.sympify(entry[1]).free_symbols & variable_set
            and not sp.sympify(entry[2]).free_symbols & variable_set
        ]
        if len(constant_limits) == len(parsed_ranges):
            term_results = []
            for term in expr.args:
                result = _try_product_separable(term, variables, parsed_ranges, opts)
                if result is None:
                    result = integrate(
                        term,
                        *parsed_ranges,
                        assumptions=assumptions,
                        principal_value=principal_value,
                    )
                term_results.append(result)
            total = _fast_simplify(sp.Add(*term_results))
            if not total.has(sp.Integral):
                return None, _cache_result(cache_key, scaled(total))

    # Fractional powers carry principal branches. Combining bases without
    # proven sign assumptions can change the integrand on negative coordinates.
    expr = sp.powsimp(expr)
    if expr.is_Mul:
        constants = [
            factor for factor in expr.args if not sp.sympify(factor).free_symbols & variable_set
        ]
        if constants:
            coefficient = sp.Mul(*constants)
            inner = integrate(
                expr / coefficient,
                *parsed_ranges,
                assumptions=assumptions,
                principal_value=principal_value,
            )
            return None, _cache_result(cache_key, scaled(_fast_simplify(coefficient * inner)))

    superellipse = _split_superellipse(expr, variables)
    if superellipse is not None:
        outer, inner_expr = superellipse
        result = _try_superellipse(outer, inner_expr, variables, parsed_ranges, opts)
        if result is not None:
            return "superellipse", _cache_result(cache_key, scaled(result))

    # Keep existing coordinate-sum closed forms ahead of exact series.
    decomposition = _cached_decomposition(expr, tuple(variables))
    inner = decomposition.g_inner if decomposition is not None else sp.S.Zero
    factored_inner = inner
    if decomposition is not None and decomposition.is_polynomial:
        try:
            if sp.Poly(inner, *variables).total_degree() <= 2:
                factored_inner = sp.factor(inner)
        except sp.PolynomialError:
            pass
    sum_forms = ([inner] if inner.is_Add else []) + [
        power.base for power in factored_inner.atoms(sp.Pow) if power.exp == 2 and power.base.is_Add
    ]
    if (
        decomposition is not None
        and all(sp.sympify(b) not in (-sp.oo, sp.oo) for r in parsed_ranges for b in r[1:])
        and any(
            all(len(term.free_symbols & variable_set) <= 1 for term in form.args)
            for form in sum_forms
        )
    ):
        result = _run_decomp_strategies(decomposition, variables, parsed_ranges, opts)
        if result is not None:
            return "decomposition", _cache_result(cache_key, scaled(result))
        # Preserve the existing closed-form iterated treatment of coordinate sums.
        return _fallback_integral(expr, parsed_ranges, opts, cache_key, scale)

    if not principal_value:
        from .entire_box_series import entire_box_series

        series = entire_box_series(expr, parsed_ranges, assumptions)
        if series is not None:
            return "entire_box_series", _cache_result(cache_key, scaled(series))

    if decomposition is None:
        return _fallback_integral(expr, parsed_ranges, opts, cache_key, scale)

    result = _run_decomp_strategies(decomposition, variables, parsed_ranges, opts)
    if result is not None:
        return "decomposition", _cache_result(cache_key, scaled(result))

    deeper = _decompose_integrand(expr, variables)
    if deeper is not None and (
        deeper.g_inner != decomposition.g_inner
        or deeper.f_outer(sp.Symbol("_u")) != decomposition.f_outer(sp.Symbol("_u"))
    ):
        result = _run_decomp_strategies(deeper, variables, parsed_ranges, opts)
        if result is not None:
            return "decomposition", _cache_result(cache_key, scaled(result))

    return _fallback_integral(expr, parsed_ranges, opts, cache_key, scale)


def _evaluate_integral(
    f: sp.Expr,
    *ranges,
    assumptions=None,
    principal_value=False,
    regularization=None,
) -> sp.Expr:
    """Symbolically evaluate a definite multiple integral."""
    assumption_context, parsed_ranges, direct_region = _normalize_evaluation_domain(
        ranges, assumptions
    )
    opts = {"_mi_assumptions": assumption_context}
    expr = sp.sympify(f)

    prescription = _evaluate_prescription(
        expr,
        parsed_ranges,
        direct_region,
        assumption_context,
        principal_value,
        regularization,
    )
    if isinstance(prescription, tuple):
        return _recorded(*prescription)
    pv = prescription

    region = direct_region or region_from_ranges(parsed_ranges, structural_order="inner-first")
    region_pv = _region_principal_value(expr, region, pv, assumption_context)
    if region_pv is not None:
        return _recorded(*region_pv)

    cache_key = _integral_cache_key(expr, region, principal_value, assumption_context)
    cached = _cached_lookup(cache_key)
    if cached is not None:
        return _recorded("cache", cached)

    reduced = _pre_strategy_reduction(
        expr,
        region,
        parsed_ranges,
        assumption_context,
        principal_value,
        opts,
        cache_key,
        direct_region,
    )
    if len(reduced) == 2:
        return _recorded(*reduced)

    expr, region, parsed_ranges, scale, variables, variable_set, orientation = reduced
    method, value = _dispatch_integrand_strategies(
        expr,
        region,
        parsed_ranges,
        variables,
        variable_set,
        assumption_context,
        principal_value,
        opts,
        cache_key,
        scale,
        orientation,
    )
    return value if method is None else _recorded(method, value)


def _integrate_domain(
    f: sp.Expr,
    region: object,
    *,
    variables=None,
    bounds=None,
    parameters=None,
    assumptions=None,
    principal_value=False,
) -> sp.Expr:
    from multiple_integrate.adapters.semialg import semialgebraic_domain

    formula, vars_ = semialgebraic_domain(region, variables)
    enforce_semialg_budget(formula, vars_, _CURRENT_POLICY.get())
    if bounds is None:
        from multiple_integrate.clipped_quadratic import integrate_clipped_quadratic_cell

        clipped = integrate_clipped_quadratic_cell(f, formula, vars_, assumptions=assumptions)
        if clipped is None:
            from multiple_integrate.quadratic_sectors import integrate_quadratic_sector

            clipped = integrate_quadratic_sector(f, formula, vars_, assumptions=assumptions)
        if clipped is None and principal_value in (False, None):
            from multiple_integrate.boundary_reduction import integrate_boundary

            clipped = integrate_boundary(f, formula, vars_, assumptions=assumptions)
        if clipped is not None:
            record_strategy(clipped.method)
            for claim in clipped.convergence_evidence:
                record_convergence(claim.kind, claim.established, claim.method, claim.condition)
            for claim in clipped.coverage:
                record_coverage(claim.scope, claim.complete, claim.method, **dict(claim.details))
            return clipped.value
        from multiple_integrate.cusp_horns import try_cusp_horn

        horn = try_cusp_horn(sp.sympify(f), formula, vars_, assumptions)
        if horn is not None:
            return horn
        from multiple_integrate.quadratic_radial import try_weighted_radial_region

        weighted = try_weighted_radial_region(f, region, variables=vars_, assumptions=assumptions)
        if weighted is not None:
            return weighted
    reduced = reduce_domain(f, region, variables=variables, bounds=bounds, parameters=parameters)
    if reduced is None:
        from multiple_integrate.adapters.semialg import region_dimension

        dimension = region_dimension(formula, vars_) if formula.free_symbols <= set(vars_) else None
        if dimension is not None and dimension < len(vars_):
            record_strategy("semialgebraic_domain_reduction", reason="certified ambient null set")
            record_coverage(
                "ambient_null_set",
                True,
                "semialgebraic_domain_reduction",
                dimension=dimension,
                ambient_dimension=len(vars_),
                basis="certified dimension below ambient measure",
                source_formula=formula,
                source_variables=vars_,
                source_density=sp.sympify(f),
            )
            return sp.Integer(0)
        raise UnresolvedIntegration(
            "semialg could not produce a certified cylindrical reduction for this region"
        )
    enforce_semialg_piece_budget(len(reduced.pieces), _CURRENT_POLICY.get())
    total = sp.Integer(0)
    for piece in reduced.pieces:
        total += integrate(
            piece.integrand,
            *piece.limits,
            assumptions=assumptions,
            principal_value=principal_value,
        )
    strategy = (
        "semialg_cad_reduction"
        if "cad" in reduced.method.lower()
        else "semialgebraic_domain_reduction"
    )
    from multiple_integrate.planner import record_piece_convergence

    record_piece_convergence(reduced.pieces, strategy, assumptions=assumptions)
    record_strategy(strategy, reason=reduced.method)
    record_coverage(
        "returned_domain_pieces",
        True,
        strategy,
        pieces=len(reduced.pieces),
        basis="Semialg reduction contract; all returned pieces consumed",
    )
    return _fast_simplify(total)


def _integrate_intrinsic(
    f: sp.Expr,
    region: object,
    *,
    variables=None,
    measure_dimension=None,
    assumptions=None,
    principal_value=False,
) -> sp.Expr:
    from multiple_integrate.adapters.semialg import SingularStratum, semialgebraic_domain

    formula, vars_ = semialgebraic_domain(region, variables)
    enforce_semialg_budget(formula, vars_, _CURRENT_POLICY.get())
    dimension = None if measure_dimension in ("intrinsic", "top") else measure_dimension
    reduced = reduce_intrinsic_domain(f, region, variables=variables, dimension=dimension)
    if reduced is None:
        raise UnresolvedIntegration(
            "semialg could not produce a certified intrinsic stratification for this region"
        )
    if reduced.singular_strata:
        indices = ", ".join(
            str(item.cell_index if isinstance(item, SingularStratum) else item["cell_index"])
            for item in reduced.singular_strata
        )
        reasons = tuple(
            reason
            for item in reduced.singular_strata
            for reason in (
                item.reasons if isinstance(item, SingularStratum) else item.get("reasons", ())
            )
        )
        detail = f"; {reasons[0]}" if reasons else ""
        raise NotImplementedError(
            f"intrinsic integration encountered uncertified/singular target-dimensional strata: {indices}{detail}"
        )
    enforce_semialg_piece_budget(len(reduced.pieces), _CURRENT_POLICY.get())
    total = sp.Integer(0)
    for piece in reduced.pieces:
        if piece.direct_value is not None:
            total += piece.direct_value
            continue
        if not piece.limits:
            total += piece.integrand
            continue
        total += integrate(
            piece.integrand,
            *piece.limits,
            assumptions=assumptions,
            principal_value=principal_value,
        )
    from multiple_integrate.planner import record_piece_convergence

    record_piece_convergence(reduced.pieces, "semialg_intrinsic_reduction", assumptions=assumptions)
    record_strategy(
        "semialg_intrinsic_reduction",
        reason=f"dimension={reduced.dimension}; pieces={len(reduced.pieces)}",
    )
    record_coverage(
        "intrinsic_cover",
        True,
        "semialg_intrinsic_reduction",
        pieces=len(reduced.pieces),
        dimension=reduced.dimension,
        basis="verified Semialg cover; no unresolved target-dimensional strata",
    )
    return _fast_simplify(total)


def _canonical_ambient_measure_is_zero(region: object) -> bool:
    """Return whether a canonical region has lower dimension than its ambient space."""
    dimension = getattr(region, "dimension", None)
    ambient_dimension = getattr(region, "ambient_dimension", None)
    if not callable(dimension) or not callable(ambient_dimension):
        return False
    try:
        return int(dimension()) < int(ambient_dimension())
    except (TypeError, ValueError):
        return False


def _is_range_triple(value) -> bool:
    return isinstance(value, (tuple, list, sp.Tuple)) and len(value) == 3


def _bounded_request(region, variables, bounds):
    """Intersect requested coordinate bounds before selecting native geometry."""
    from collections.abc import Mapping

    from .adapters.semialg import semialgebraic_domain
    from .region_types import BoxRegion

    coords = tuple(variables or getattr(region, "variables", ()))
    formula, coords = semialgebraic_domain(region, coords or None)
    names = {str(v): v for v in coords}
    entries = (
        ((v, *limits) for v, limits in bounds.items()) if isinstance(bounds, Mapping) else bounds
    )
    restrictions = {}
    atoms = []
    for entry in entries:
        if len(entry) != 3:
            raise ValueError("bounds entries must be variable/lower/upper triples")
        v, lo, hi = entry
        v = names.get(v, sp.sympify(v)) if isinstance(v, str) else sp.sympify(v)
        if v not in coords:
            raise ValueError("bound variable is not an integration coordinate")
        lo, hi = sp.sympify(lo), sp.sympify(hi)
        atoms.extend((sp.Ge(v, lo, evaluate=False), sp.Le(v, hi, evaluate=False)))
        if v in restrictions:
            old_lo, old_hi = restrictions[v]
            lo, hi = sp.Max(lo, old_lo), sp.Min(hi, old_hi)
        restrictions[v] = (lo, hi)
    if isinstance(region, BoxRegion):
        clipped = []
        for v, lo, hi in region.ranges:
            if v in restrictions:
                a, b = restrictions[v]
                lo, hi = sp.Max(lo, a), sp.Min(hi, b)
            clipped.append((v, lo, hi))
        if all(not sp.sympify(bound).has(*coords) for r in clipped for bound in r[1:]) and all(
            sp.simplify(hi - lo).is_positive is True for _, lo, hi in clipped
        ):
            return BoxRegion(tuple(clipped)), None
    return sp.And(formula, *atoms), coords


@isolate_nested_coverage
def integrate(
    f: sp.Expr,
    *ranges,
    assumptions=None,
    principal_value=False,
    regularization=None,
    region=None,
    variables=None,
    bounds=None,
    measure_dimension: int | str = "ambient",
    policy=None,
) -> sp.Expr:
    """Symbolically evaluate a definite multiple integral.

    Parameters
    ----------
    f : sympy.Expr
        Integrand.
    *ranges : tuple
        Inner-first ``(variable, lower, upper)`` ranges, matching
        ``sympy.integrate`` ordering, or one supported ``Region`` object.
    assumptions : sympy Boolean condition or iterable of conditions, optional
        Additional mathematical facts used by convergence checks and symbolic
        refinement. Examples are ``a > 0``, ``{a > 0, b != 0}``, and
        ``{Q.positive(a), Q.integer(n)}``.
    principal_value : bool or principal-value prescription, default=False
        ``True`` requests a one-dimensional Cauchy PV. Use ``IteratedPrincipalValue``,
        ``GeometricPrincipalValue``, or ``HypersurfacePrincipalValue`` for explicit semantics.
    regularization : HadamardFinitePart, optional
        Extract a finite part from the shared-cutoff asymptotic expansion. Mutually exclusive
        with ``principal_value``.
    region : Region, semialg.Geometry, SemialgebraicRegion, or Boolean condition, optional
        Integration domain. Canonical semialg geometry and raw formulas require
        ``variables=`` to name ambient coordinates; structured integration domains own
        their coordinate order.
    variables : sequence of Symbol, optional
        Ambient integration variables for a raw semialgebraic condition.
    bounds : mapping or sequence, optional
        Optional explicit finite bounds forwarded to semialg's structural reducer.
    measure_dimension : {"ambient", "intrinsic", "top"} or int, default="ambient"
        Measure dimension for a supported region. Intrinsic/top measure consumes
        semialg's unified native/CAD parametric-cover contract and its induced
        Hausdorff metric factors.

    Returns
    -------
    sympy.Expr
        Exact result when available, otherwise an unevaluated SymPy expression.
    """
    if region is not None and ranges:
        raise ValueError("pass either explicit ranges or region=, not both")
    if region is None and len(ranges) == 1:
        candidate = ranges[0]
        is_semialgebraic_region = hasattr(candidate, "formula") and hasattr(candidate, "variables")
        if is_semialgebraic_region or (variables is not None and not _is_range_triple(candidate)):
            region = candidate
            ranges = ()
    context = _normalize_assumptions(assumptions)
    policy_value = _normalize_policy(policy)
    if region is not None and bounds is not None:
        region, variables = _bounded_request(region, variables, bounds)
        bounds = None
    if regularization is not None:
        if measure_dimension not in (None, "ambient"):
            raise NotImplementedError("finite-part regularization supports ambient measure only")
        if principal_value not in (False, None):
            raise ValueError("principal_value= and regularization= are mutually exclusive")
        if not isinstance(regularization, HadamardFinitePart):
            raise TypeError("regularization must be a HadamardFinitePart prescription")
        policy_token = _CURRENT_POLICY.set(policy_value)
        try:
            return _recorded(
                "hadamard_finite_part",
                hadamard_finite_part(
                    f,
                    *ranges,
                    region=region,
                    variables=variables,
                    bounds=bounds,
                    prescription=regularization,
                    assumptions=context,
                ),
            )
        finally:
            _CURRENT_POLICY.reset(policy_token)
    pv = _normalize_principal_value(principal_value)
    if measure_dimension not in (None, "ambient") and isinstance(
        pv, (GeometricPrincipalValue, HypersurfacePrincipalValue)
    ):
        raise NotImplementedError(
            "geometric and hypersurface principal values support ambient measure only"
        )
    policy_token = _CURRENT_POLICY.set(policy_value)
    token = _CURRENT_ASSUMPTIONS.set(context)
    try:
        if region is not None and isinstance(pv, HypersurfacePrincipalValue):
            return _recorded(
                "hypersurface_principal_value",
                hypersurface_pv(
                    f,
                    region,
                    pv,
                    variables=variables,
                    bounds=bounds,
                    assumptions=context,
                ),
            )
        if region is not None and isinstance(pv, GeometricPrincipalValue):
            return _recorded(
                "geometric_principal_value",
                geometric_pv(
                    f,
                    region,
                    pv,
                    variables=variables,
                    bounds=bounds,
                    assumptions=context,
                ),
            )
        if region is not None and isinstance(region, Region):
            if bounds is not None:
                raise ValueError("bounds= is not used with structured integration domains")
            if measure_dimension not in (None, "ambient"):
                geometry = domain_geometry(region)
                if geometry is None:
                    raise ValueError(
                        "intrinsic measure is unavailable for this integration-specific Region"
                    )
                owned_variables = tuple(region.variables)
                if variables is not None and tuple(variables) != owned_variables:
                    raise ValueError("variables= must match the integration-domain variable order")
                return _integrate_intrinsic(
                    f,
                    geometry,
                    variables=owned_variables,
                    measure_dimension=measure_dimension,
                    assumptions=context,
                    principal_value=principal_value,
                )
            if variables is not None:
                raise ValueError("variables= is not used with structured integration domains")
            return _evaluate_integral(
                f,
                region,
                assumptions=context,
                principal_value=pv,
            )
        if region is not None:
            if measure_dimension in (None, "ambient") and _canonical_ambient_measure_is_zero(
                region
            ):
                record_strategy(
                    "ambient_null_set", reason="canonical dimension below ambient dimension"
                )
                record_coverage(
                    "ambient_null_set",
                    True,
                    "canonical_geometry",
                    basis="canonical dimension below ambient dimension",
                    source_region=region,
                    source_density=sp.sympify(f),
                )
                return sp.Integer(0)
            if measure_dimension not in (None, "ambient"):
                if bounds is not None:
                    raise ValueError(
                        "bounds= is not supported with intrinsic measure; include bounds in the region formula"
                    )
                return _integrate_intrinsic(
                    f,
                    region,
                    variables=variables,
                    measure_dimension=measure_dimension,
                    assumptions=context,
                    principal_value=principal_value,
                )
            return _integrate_domain(
                f,
                region,
                variables=variables,
                bounds=bounds,
                assumptions=context,
                principal_value=principal_value,
            )
        if (
            variables is not None
            or bounds is not None
            or measure_dimension not in (None, "ambient")
        ):
            raise ValueError(
                "variables=, bounds=, and non-ambient measure_dimension are only valid with region="
            )
        if not principal_value and len(ranges) == 1:
            theorem_value = specialized_value(sp.sympify(f), ranges, context)
            if theorem_value is not None:
                return _recorded("conditional_convergence", theorem_value)
        return _evaluate_integral(
            f,
            *ranges,
            assumptions=context,
            principal_value=principal_value,
        )
    finally:
        _CURRENT_ASSUMPTIONS.reset(token)
        _CURRENT_POLICY.reset(policy_token)
