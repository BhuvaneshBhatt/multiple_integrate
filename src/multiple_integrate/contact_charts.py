"""Proved polynomial shear charts for finite-order graph contacts."""

import sympy as sp
from sympy.core.relational import Relational

from .adapters.semialg import semialgebraic_domain
from .assumptions import _ask, _normalize_assumptions
from .fast_rules import _polynomial


def polynomial_contact_cutoff(expr, region, level, epsilon, variables, assumptions=None):
    """Lower graph pairs P(x), P(x)+c*(x-a)**m by a global polynomial shear.

    No implicit-contact chart is inferred from an unproved local inverse.
    """
    from .tangent_cutoff import tangent_graph_cutoff

    variables = tuple(variables)
    if len(variables) != 2:
        return None
    context = _normalize_assumptions(assumptions)
    formula, _ = semialgebraic_domain(region, variables)

    def proved(q):
        return _ask(q, context) is True

    for x, y in (variables, tuple(reversed(variables))):
        poly = _polynomial(level, (x, y))
        if poly is None:
            continue
        try:
            constant, factors = sp.factor_list(level, x, y)
        except (sp.PolynomialError, ValueError):
            continue
        if len(factors) != 2 or any(m != 1 for _, m in factors):
            continue
        graphs = []
        for factor, _ in factors:
            p = sp.Poly(factor, y)
            slope = p.nth(1)
            if (
                p.degree() != 1
                or slope.has(x)
                or not proved(sp.Q.real(slope) & sp.Q.finite(slope) & sp.Q.nonzero(slope))
            ):
                break
            graph = _polynomial(-p.nth(0) / slope, (x,))
            if graph is None or not all(
                proved(sp.Q.real(c) & sp.Q.finite(c)) for c in graph.coeffs()
            ):
                break
            graphs.append(graph.as_expr())
        if len(graphs) != 2:
            continue
        for bottom, top in (graphs, tuple(reversed(graphs))):
            gap = sp.Poly(top - bottom, x)
            m = gap.degree()
            if m < 2:
                continue
            center = sp.cancel(-gap.nth(m - 1) / (m * gap.LC()))
            if not proved(sp.Q.real(center) & sp.Q.finite(center) & sp.Q.positive(gap.LC())):
                continue
            u, v = sp.Dummy("contact_base", real=True), sp.Dummy("contact_normal", real=True)
            width = sp.expand(gap.as_expr().subs(x, u + center))
            if sp.expand(width - gap.LC() * u**m) != 0:
                continue
            if bottom == 0 and center == 0:
                continue  # already the canonical tangent family
            mapping = {x: u + center, y: v + bottom.subs(x, u + center)}
            mapped_formula = formula.subs(mapping, simultaneous=True).replace(
                lambda a: isinstance(a, Relational),
                lambda a: a.func(sp.expand(a.lhs), sp.expand(a.rhs)),
            )
            num, den = sp.sympify(expr).as_numer_denom()
            if _polynomial(num, (x, y)) is None or _polynomial(den, (x, y)) is None:
                continue
            density = sp.cancel(sp.sympify(expr).subs(mapping, simultaneous=True))
            mapped_level = sp.factor(level.subs(mapping, simultaneous=True))
            result = tangent_graph_cutoff(
                density, mapped_formula, mapped_level, epsilon, (u, v), assumptions
            )
            if result is None:
                continue
            value, details = result
            old_targets = details["target_variables"]
            inner = dict(zip((u, v), details["forward_map"], strict=True))
            composed = {
                x: sp.expand(mapping[x].subs(inner, simultaneous=True)),
                y: sp.expand(mapping[y].subs(inner, simultaneous=True)),
            }
            details.update(
                realization="polynomial_contact_cutoff",
                source_formula=formula,
                source_variables=variables,
                source_density=sp.sympify(expr),
                source_level=level,
                forward_map=tuple(composed[w] for w in variables),
                inverse_map=(x - center, (y - bottom) / (gap.LC() * (x - center) ** m)),
                contact_point=center,
                contact_graph=bottom,
                shear_jacobian=sp.S.One,
                transformed_level=sp.factor(level.subs(composed, simultaneous=True)),
                target_variables=old_targets,
                coverage_basis="global polynomial shear and invertible weighted chart off the excluded contact",
            )
            details["transformed_cutoff"] = sp.Abs(details["transformed_level"]) > epsilon
            return value, details
    return None
