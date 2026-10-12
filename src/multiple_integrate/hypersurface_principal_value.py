"""Principal values across regular polynomial hypersurfaces."""

from __future__ import annotations

import sympy as sp

from .adapters.semialg import implies as semialg_implies
from .adapters.semialg import reduce_domain, region_condition, semialgebraic_domain
from .hypersurface_atlas import HypersurfaceAtlas, build_hypersurface_atlas
from .planner import record_coverage, record_strategy
from .principal_value_prescriptions import HypersurfacePrincipalValue
from .radial_parametrization import radial_parametrization
from .recognition import _fast_simplify
from .region_types import BallRegion, DiskRegion, EllipsoidRegion, Region
from .regularization import RegularizationProvenance, _record_regularization
from .results import DivergentIntegration, UnresolvedIntegration
from .singular_geometry import product_level_cutoff


def _hypersurface_graph_chart(
    polynomial: sp.Expr, variables: tuple[sp.Symbol, ...]
) -> tuple[sp.Symbol, sp.Symbol, sp.Expr, sp.Expr] | None:
    """Return transverse variable, level variable, inverse chart, and Jacobian.

    A chart exists when the polynomial is affine in one integration variable
    and its derivative with respect to that variable is a nonzero constant.
    """
    g = sp.expand(polynomial)
    try:
        sp.Poly(g, *variables)
    except sp.PolynomialError as exc:
        raise ValueError("hypersurface must be polynomial in the integration variables") from exc
    for variable in reversed(variables):
        derivative = sp.simplify(sp.diff(g, variable))
        if derivative == 0 or derivative.has(*variables):
            continue
        if derivative.is_zero is not False:
            continue
        base = sp.simplify(g - derivative * variable)
        level = sp.Dummy("pv_level", real=True)
        inverse = sp.simplify((level - base) / derivative)
        jacobian = sp.simplify(1 / sp.Abs(derivative))
        return variable, level, inverse, jacobian
    return None


def _range_formula(limits: tuple[tuple[sp.Symbol, sp.Expr, sp.Expr], ...]) -> sp.Expr:
    atoms = []
    for variable, lower, upper in limits:
        if lower != -sp.oo:
            atoms.append(sp.Ge(variable, lower))
        if upper != sp.oo:
            atoms.append(sp.Le(variable, upper))
    return sp.And(*atoms) if atoms else sp.true


def _certified_relation(
    relation: sp.Expr, limits: tuple[tuple[sp.Symbol, sp.Expr, sp.Expr], ...]
) -> bool:
    simplified = sp.simplify(relation)
    if simplified is sp.true:
        return True
    outer_variables = tuple(limit[0] for limit in limits)
    if not outer_variables:
        return simplified is sp.true
    try:
        return semialg_implies(_range_formula(limits), relation, outer_variables) is True
    except (ArithmeticError, TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        return False


def _cutoff_inner_integral(
    integrand: sp.Expr,
    level: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    epsilon: sp.Symbol,
    outer_limits: tuple[tuple[sp.Symbol, sp.Expr, sp.Expr], ...],
) -> sp.Expr:
    lower_nonnegative = lower.is_nonnegative is True or _certified_relation(
        sp.Ge(lower, 0), outer_limits
    )
    upper_nonpositive = upper.is_nonpositive is True or _certified_relation(
        sp.Le(upper, 0), outer_limits
    )
    lower_nonpositive = lower.is_nonpositive is True or _certified_relation(
        sp.Le(lower, 0), outer_limits
    )
    upper_nonnegative = upper.is_nonnegative is True or _certified_relation(
        sp.Ge(upper, 0), outer_limits
    )
    if lower_nonnegative:
        start = sp.Max(lower, epsilon) if lower != 0 else epsilon
        return sp.integrate(integrand, (level, start, upper))
    if upper_nonpositive:
        stop = sp.Min(upper, -epsilon) if upper != 0 else -epsilon
        return sp.integrate(integrand, (level, lower, stop))
    if lower_nonpositive and upper_nonnegative:
        return sp.integrate(integrand, (level, lower, -epsilon)) + sp.integrate(
            integrand, (level, epsilon, upper)
        )
    raise NotImplementedError(
        "semialg reduction did not certify which side of the hypersurface a cylindrical fiber occupies"
    )


def _structured_hypersurface_radial_data(
    region: Region, polynomial: sp.Expr
) -> (
    tuple[tuple[sp.Symbol, ...], tuple[sp.Expr, ...], sp.ImmutableMatrix, sp.Expr, sp.Expr, sp.Expr]
    | None
):
    """Recognize a concentric regular quadric inside a radial structured region.

    Returns ``(variables, center, affine, outer_radius, lambda, constant)`` for
    ``g(center + affine*r*omega) = lambda*r**2 + constant``.
    """
    if isinstance(region, (DiskRegion, BallRegion)):
        variables = tuple(region.variables)
        center = tuple(region.center)
        affine = sp.ImmutableMatrix(sp.eye(len(variables)))
        outer_radius = sp.sympify(region.radius)
    elif isinstance(region, EllipsoidRegion):
        variables = tuple(region.variables)
        center = tuple(region.center)
        affine = sp.ImmutableMatrix(sp.diag(*region.axes))
        outer_radius = sp.Integer(1)
    else:
        return None
    g = sp.expand(sp.sympify(polynomial))
    if any(
        sp.simplify(sp.diff(g, variable).subs(dict(zip(variables, center, strict=True)))) != 0
        for variable in variables
    ):
        return None
    hessian = sp.ImmutableMatrix(
        [[sp.diff(g, left, right) for right in variables] for left in variables]
    )
    if any(entry.has(*variables) for entry in hessian):
        return None
    quadratic = sp.simplify(hessian / 2)
    radial_form = sp.simplify(affine.T * quadratic * affine)
    lam = sp.simplify(radial_form[0, 0])
    if lam.is_positive is not True:
        return None
    identity_form = sp.ImmutableMatrix(sp.eye(len(variables))) * lam
    if radial_form != identity_form:
        return None
    constant = sp.simplify(g.subs(dict(zip(variables, center, strict=True)), simultaneous=True))
    root_sq = sp.simplify(-constant / lam)
    if root_sq.is_positive is not True:
        return None
    if sp.simplify(outer_radius**2 - root_sq).is_positive is not True:
        return None
    return variables, center, affine, outer_radius, lam, constant


def _quadric_hypersurface_cutoff_integral(
    expr: sp.Expr,
    region: Region,
    polynomial: sp.Expr,
    epsilon: sp.Symbol,
    *,
    assumptions=None,
) -> sp.Expr | None:
    """Evaluate a closed concentric quadric cutoff through one affine radial chart."""
    data = _structured_hypersurface_radial_data(region, polynomial)
    if data is None:
        return None
    variables, center, affine, outer_radius, lam, constant = data
    chart = radial_parametrization(variables, center, affine_matrix=affine)
    transformed = sp.simplify(
        sp.sympify(expr).subs(dict(chart.mapping), simultaneous=True) * chart.jacobian_factor
    )
    from .core import integrate

    angular = integrate(transformed, *chart.angular_ranges, assumptions=assumptions)
    radial = chart.radial_variable
    root_sq = sp.simplify(-constant / lam)
    lower_radius = sp.sqrt(sp.simplify(root_sq - epsilon / lam))
    upper_radius = sp.sqrt(sp.simplify(root_sq + epsilon / lam))
    cutoff = sp.integrate(angular, (radial, 0, lower_radius)) + sp.integrate(
        angular, (radial, upper_radius, outer_radius)
    )
    return sp.simplify(cutoff)


def _piecewise_cutoff_inner_integral(
    integrand: sp.Expr,
    level: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    epsilon: sp.Symbol,
) -> sp.Expr:
    """Fallback cutoff for fibers whose side changes across outer parameters."""
    ordinary = sp.integrate(integrand, (level, lower, upper))
    crossing = sp.integrate(integrand, (level, lower, -epsilon)) + sp.integrate(
        integrand, (level, epsilon, upper)
    )
    if sp.simplify(lower) == 0:
        ordinary = sp.integrate(integrand, (level, epsilon, upper))
    elif sp.simplify(upper) == 0:
        ordinary = sp.integrate(integrand, (level, lower, -epsilon))
    return sp.Piecewise(
        (ordinary, sp.Or(sp.Ge(lower, 0), sp.Le(upper, 0))),
        (crossing, sp.And(sp.Lt(lower, 0), sp.Gt(upper, 0))),
        (ordinary, True),
    )


def _atlas_cutoff_integral(
    expr: sp.Expr,
    formula: sp.Expr,
    vars_: tuple[sp.Symbol, ...],
    atlas: HypersurfaceAtlas,
    epsilon: sp.Symbol,
    *,
    bounds=None,
    assumptions=None,
) -> tuple[sp.Expr, dict[str, object]]:
    """Evaluate one shared-cutoff integral over disjoint implicit local charts."""
    total = sp.Integer(0)
    chart_records: list[dict[str, object]] = []
    empty_charts: list[int] = []
    from .core import integrate

    for chart in atlas.charts:
        if not chart.explicit:
            if semialg_implies(formula, sp.Not(chart.patch_condition), vars_) is True:
                empty_charts.append(chart.chart_index)
                continue
            record_coverage(
                "atlas_charts",
                False,
                "hypersurface_atlas",
                failed_chart=chart.chart_index,
                reason="unrealized chart without emptiness evidence",
            )
            raise NotImplementedError("hypersurface atlas has an unrealized nonempty chart")
        others = tuple(variable for variable in vars_ if variable != chart.transverse)
        ordered = (chart.transverse, *others)
        reduced = reduce_domain(
            expr,
            sp.And(formula, chart.patch_condition),
            variables=ordered,
            bounds=bounds,
        )
        if reduced is None or not reduced.pieces:
            if semialg_implies(formula, sp.Not(chart.patch_condition), vars_) is True:
                empty_charts.append(chart.chart_index)
                continue
            record_coverage(
                "atlas_charts",
                False,
                "hypersurface_atlas",
                failed_chart=chart.chart_index,
                reason="reduction unavailable without emptiness evidence",
            )
            raise UnresolvedIntegration(
                "hypersurface atlas chart reduction failed without emptiness evidence"
            )
        used_pieces = 0
        mapping = {chart.transverse: chart.inverse}
        for piece in reduced.pieces:
            transverse_limits = tuple(
                limit for limit in piece.limits if limit[0] == chart.transverse
            )
            if len(transverse_limits) != 1:
                raise NotImplementedError(
                    "semialg atlas reduction must expose the transverse coordinate as one cylindrical limit"
                )
            _, lower, upper = transverse_limits[0]
            outer_limits = tuple(limit for limit in piece.limits if limit[0] != chart.transverse)
            lower_level = sp.simplify(
                atlas.polynomial.subs(chart.transverse, lower, simultaneous=True)
            )
            upper_level = sp.simplify(
                atlas.polynomial.subs(chart.transverse, upper, simultaneous=True)
            )
            if chart.derivative_sign < 0:
                lower_level, upper_level = upper_level, lower_level
            transformed = sp.simplify(
                piece.integrand.subs(mapping, simultaneous=True) * chart.jacobian
            )
            try:
                inner = _cutoff_inner_integral(
                    transformed,
                    chart.level,
                    lower_level,
                    upper_level,
                    epsilon,
                    outer_limits,
                )
            except NotImplementedError:
                inner = _piecewise_cutoff_inner_integral(
                    transformed,
                    chart.level,
                    lower_level,
                    upper_level,
                    epsilon,
                )
            if outer_limits:
                inner = integrate(inner, *outer_limits, assumptions=assumptions)
            total += inner
            used_pieces += 1
        if used_pieces:
            chart_records.append(
                {
                    "index": chart.chart_index,
                    "transverse": chart.transverse,
                    "derivative_sign": chart.derivative_sign,
                    "patch": chart.patch_condition,
                    "implicit_relation": chart.implicit_relation,
                    "derivative": chart.derivative,
                    "inverse": chart.inverse,
                    "jacobian": chart.jacobian,
                    "explicit": chart.explicit,
                    "piece_count": used_pieces,
                }
            )
    if not chart_records and (not atlas.charts or len(empty_charts) != len(atlas.charts)):
        raise NotImplementedError(
            "semialg could not reduce any certified implicit-hypersurface chart"
        )
    record_coverage(
        "atlas_charts",
        True,
        "hypersurface_atlas",
        requested=len(atlas.charts),
        integrated=len(chart_records),
        certified_empty=len(atlas.charts) - len(chart_records),
        basis="all requested charts integrated or independently certified empty",
    )
    return sp.simplify(total), {
        "coverage_complete": True,
        "atlas_requested_chart_count": len(atlas.charts),
        "atlas_empty_chart_count": len(empty_charts),
        "atlas_empty_charts": tuple(empty_charts),
        "atlas_chart_count": len(chart_records),
        "atlas_partition": atlas.partition_rule,
        "regularity_certified": atlas.regularity_certified,
        "charts": tuple(chart_records),
    }


def hypersurface_cutoff_integral(
    expr: sp.Expr,
    region: object,
    prescription: HypersurfacePrincipalValue,
    *,
    variables=None,
    bounds=None,
    assumptions=None,
    return_details: bool = False,
):
    """Construct the shared-epsilon cutoff integral for a polynomial hypersurface.

    A global affine graph chart is used when available.  Otherwise semialg
    certifies regularity and the hypersurface is covered by a disjoint atlas of
    gradient-dominance graph patches, all sharing the same cutoff parameter.
    """
    if isinstance(region, Region):
        formula = region_condition(region)
        vars_ = tuple(region.variables)
        if formula is None:
            raise NotImplementedError("region cannot be translated to a semialgebraic formula")
    else:
        formula, vars_ = semialgebraic_domain(region, variables)
    epsilon = sp.Dummy("regularization_epsilon", positive=True, real=True)
    product_cutoff = product_level_cutoff(
        sp.sympify(expr),
        region,
        prescription.hypersurface,
        epsilon,
        assumptions=assumptions,
        variables=vars_,
    )
    if product_cutoff is not None:
        total, details = product_cutoff
        record_coverage(
            "integration_domain",
            True,
            str(details["realization"]),
            basis=details.get(
                "coverage_basis", "all orthants under the requested shared |g| cutoff"
            ),
            orthants=details["orthants"],
        )
        cutoff_domain = sp.And(sp.sympify(formula), sp.Gt(prescription.hypersurface**2, epsilon**2))
        result = (total, epsilon, cutoff_domain, sp.sympify(formula))
        return (*result, details) if return_details else result
    chart = _hypersurface_graph_chart(prescription.hypersurface, vars_)
    details: dict[str, object]
    if chart is None:
        atlas = build_hypersurface_atlas(prescription.hypersurface, sp.sympify(formula), vars_)
        radial_cutoff = (
            _quadric_hypersurface_cutoff_integral(
                sp.sympify(expr),
                region,
                prescription.hypersurface,
                epsilon,
                assumptions=assumptions,
            )
            if isinstance(region, Region)
            else None
        )
        if radial_cutoff is not None:
            total = radial_cutoff
            record_coverage(
                "integration_domain",
                True,
                "affine_radial_quadric",
                basis="whole-domain radial realization; atlas charts not used for integration",
            )
            details = {
                "coverage_complete": True,
                "atlas_chart_count": len(atlas.charts),
                "atlas_partition": atlas.partition_rule,
                "regularity_certified": atlas.regularity_certified,
                "charts": tuple(
                    {
                        "index": item.chart_index,
                        "transverse": item.transverse,
                        "derivative_sign": item.derivative_sign,
                        "patch": item.patch_condition,
                        "implicit_relation": item.implicit_relation,
                        "derivative": item.derivative,
                        "inverse": item.inverse,
                        "jacobian": item.jacobian,
                        "explicit": item.explicit,
                    }
                    for item in atlas.charts
                ),
                "realization": "affine_radial_quadric",
            }
            backend = "semialg_certified_implicit_atlas_affine_radial"
        else:
            total, details = _atlas_cutoff_integral(
                sp.sympify(expr),
                sp.sympify(formula),
                vars_,
                atlas,
                epsilon,
                bounds=bounds,
                assumptions=assumptions,
            )
            backend = "semialg_implicit_hypersurface_atlas"
        transformed_formula = sp.sympify(formula)
    else:
        transverse, level, inverse, jacobian = chart
        others = tuple(variable for variable in vars_ if variable != transverse)
        mapping = {transverse: inverse}
        transformed_formula = sp.simplify(sp.sympify(formula).subs(mapping, simultaneous=True))
        transformed_expr = sp.simplify(sp.sympify(expr).subs(mapping, simultaneous=True) * jacobian)
        ordered_variables = (level, *others)
        reduced = reduce_domain(
            transformed_expr,
            transformed_formula,
            variables=ordered_variables,
            bounds=bounds,
        )
        if reduced is None or not reduced.pieces:
            raise NotImplementedError(
                "semialg could not cylindrically reduce the hypersurface chart"
            )

        total = sp.Integer(0)
        from .core import integrate

        for piece in reduced.pieces:
            level_limits = tuple(limit for limit in piece.limits if limit[0] == level)
            if len(level_limits) != 1:
                raise NotImplementedError(
                    "semialg hypersurface reduction must expose the level-set coordinate as one cylindrical limit"
                )
            level_limit = level_limits[0]
            outer_limits = tuple(limit for limit in piece.limits if limit[0] != level)
            inner = _cutoff_inner_integral(
                piece.integrand,
                level,
                level_limit[1],
                level_limit[2],
                epsilon,
                outer_limits,
            )
            if outer_limits:
                inner = integrate(inner, *outer_limits, assumptions=assumptions)
            total += inner
        record_coverage(
            "returned_cutoff_pieces",
            True,
            "global_hypersurface_chart",
            pieces=len(reduced.pieces),
            basis="global inverse chart; all returned cylindrical pieces consumed",
        )
        details = {
            "coverage_complete": True,
            "atlas_chart_count": 1,
            "atlas_partition": "global_graph",
            "regularity_certified": True,
            "charts": ({"transverse": transverse, "inverse": inverse, "jacobian": jacobian},),
        }
        backend = "semialg_cylindrical_reduction"
    cutoff_domain = sp.And(
        sp.sympify(formula), sp.Gt(sp.expand(prescription.hypersurface**2), epsilon**2)
    )
    details["backend"] = backend
    result = (sp.simplify(total), epsilon, cutoff_domain, transformed_formula)
    if return_details:
        return (*result, details)
    return result


def hypersurface_pv(
    expr: sp.Expr,
    region: object,
    prescription: HypersurfacePrincipalValue,
    *,
    variables=None,
    bounds=None,
    assumptions=None,
) -> sp.Expr:
    """Evaluate a polynomial-hypersurface PV using one semialgebraic level-set cutoff."""
    from .regular_density import regular_density_cutoff

    regular = regular_density_cutoff(
        expr,
        region,
        prescription.hypersurface,
        assumptions=assumptions,
        prescription="HypersurfacePrincipalValue",
    )
    if regular is not None:
        return regular
    cutoff, epsilon, cutoff_domain, _, details = hypersurface_cutoff_integral(
        expr,
        region,
        prescription,
        variables=variables,
        bounds=bounds,
        assumptions=assumptions,
        return_details=True,
    )
    value = sp.limit(cutoff, epsilon, 0, dir="+")
    if value.has(sp.nan, sp.Limit, sp.Integral, sp.AccumBounds):
        raise UnresolvedIntegration("the polynomial-hypersurface cutoff limit is unresolved")
    if value.has(sp.oo, -sp.oo, sp.zoo):
        record_strategy(
            "hypersurface_principal_value", succeeded=False, reason="shared cutoff limit diverges"
        )
        raise DivergentIntegration(
            "the polynomial-hypersurface principal value does not exist finitely", value
        )
    provenance = RegularizationProvenance(
        method="polynomial_hypersurface_pv",
        prescription="HypersurfacePrincipalValue",
        cutoff_parameter=epsilon,
        regulator=sp.Abs(prescription.hypersurface),
        singular_locus=sp.Eq(prescription.hypersurface, 0),
        cutoff_domain=cutoff_domain,
        cutoff_integral=cutoff,
        constant_term=sp.simplify(value),
        shared_cutoff=True,
        backend=str(details.get("backend")),
        details=details,
    )
    _record_regularization(provenance)
    record_strategy(
        "hypersurface_principal_value", reason="shared |g| > epsilon semialgebraic cutoff"
    )
    return _fast_simplify(value)


__all__ = ["hypersurface_cutoff_integral", "hypersurface_pv"]
