# Exact fast paths, automatic cells and singular charts

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

Geometry recognition has a bounded 512-entry cache keyed by normalized ranges and structural-order mode. It is independent of the density. `clear_cache()` clears geometry and result caches. Repeating a box request can reuse its recognition before looking up the result. Cached geometry is immutable.

## Moment and elementary rules

| Density on finite real intervals / boxes | Rule | Required proof |
|---|---|---|
| Sparse polynomial on a Cartesian box | Product of endpoint power moments | Finite coefficients and independent finite bounds |
| Polynomial times an affine exponential, sine or cosine | Integration-by-parts recurrence | Finite coefficients; frequency zero has its own branch |
| Polynomial times `exp(-alpha*x**2 + beta*x + gamma)` | Shifted Gaussian recurrence using erf | `alpha>0`, finite coefficients and endpoints |
| Rational density with linear/quadratic denominator | Polynomial division and real log/atan formulas | Real denominator; strict no-pole proof on the entire interval |
| `c*sin(k/(x-h))/(x-h)` or cosine counterpart | Si / real Ci endpoint limits | Finite real endpoints and h; real nonzero k with proved sign |

Reciprocal-phase integrals touching h have proved ordinary convergence. Absolute convergence is false for a proved nonzero amplitude, true for a proved zero amplitude, and unknown when the amplitude's zero locus is unresolved. Interior h is split into two ordinary one-sided integrals, rather than interpreted as a principal value. Zero frequency declines this rule and follows the ordinary classifier. A denominator pole declines the rational fast path and proceeds through existing singular analysis. Unknown prerequisites are never inferred numerically.

Polynomial conversion checks an expansion estimate first, including nested powered products. Default limits follow `IntegrationPolicy.semialg`: degree eight and at most 64 estimated terms. Over-budget input declines this shortcut; `allow_expensive_exact=True` permits larger exact work. Reversed explicit ranges retain their orientation; geometric boxes use their geometric measure.

```python
import sympy as sp
from multiple_integrate import integrate_result
x, a = sp.symbols("x a", real=True)
r = integrate_result(x**4*sp.exp(-3*(x-1)**2), (x, 0, 1))
assert r.complete and r.absolute_convergent is True
r = integrate_result(x**4*sp.cos(a*x+1), (x, 0, 1))
assert r.value.subs(a, 0) == sp.cos(1)/5
r = integrate_result(sp.sin(1/x)/x, (x, 0, 1))
assert r.value == sp.pi/2-sp.Si(1)
assert r.ordinary_convergent is True and r.absolute_convergent is False
```

For `sin(k*x+b)/x` and `cos(k*x+b)/x` tails, proved nonzero real frequency permits Dirichlet Si/Ci formulas. Absolute sinusoids have positive periodic mean, proving harmonic-tail divergence. Zero frequency and singular lower endpoints require separate prerequisites; a missing proof declines the rule.

## Nonseparable entire densities

On finite Cartesian boxes, polynomial amplitudes times exp/sin/cos of polynomial phases have an exact multi-index `Sum` representation. For example `exp(x*y)` on the unit square integrates to `Sum(1/(factorial(k)*(k+1)**2), (k,0,oo))`. Each phase monomial supplies an index. The exponential absolute-series majorant is uniformly bounded on the compact box, proving termwise integration, absolute convergence and Fubini. On positive Gaussian decay, the recurrence also handles half-lines and the full line by vanishing polynomial-exponential boundary limits.

Trigonometric sums use real even/odd terms to avoid complex hypergeometric numerical evaluation. No numerical truncation contributes to the returned value.

The route follows existing separable strategies and preserves existing coordinate-sum closed forms before constructing a series. Polynomial and dimension budgets still apply; trigonometric parity combinations must fit the piece budget. This extends a broad nonseparable family, but does not solve arbitrary nonseparable densities or improper integrals. For this family, [bounded Taylor evaluation](series-evaluation.md) supplies exact finite moments and a rational analytic error bound without evaluating the multi-index `Sum`. Other series can still need external justified truncation or quadrature.

## Automatic polynomial event cells

Omit `cells` in `integrate_cells(..., parameters=(a,))` to request an exact partition for one real parameter. Axis-wise geometry is quadratic at most in each coordinate, with rational parameter coefficients and rational transition polynomials within the configured degree budget. Coupled linear boundary forms can additionally be tracked through coefficient zeros, parallel-boundary collisions and affine-matrix determinant zeros. Coefficient zeros, degree drops, discriminant zeros and pairwise boundary resultants produce exact transition roots. Every equality fiber is retained alongside open intervals. There is no sampling of generic fibers.

```python
import sympy as sp
from multiple_integrate import integrate_cells
x, a = sp.symbols("x a", real=True)
r = integrate_cells(1, region=x**2 <= a, variables=(x,),
                    bounds={x: (-1, 1)}, parameters=(a,))
assert r.complete
assert r.value.subs(a, -1) == 0
assert r.value.subs(a, sp.Rational(1, 4)) == 1
assert r.value.subs(a, 2) == 2
```

These cells isolate a=0 (empty-to-nonempty transition) and a=1 (boundary contact). With multiple declared real parameters, transition polynomials are split into bounded-degree rational factors. The arrangement is split incrementally into sign and equality cells, pruning only certified empty cells. Coupled quadratic boundaries track selected principal-minor and critical-level events. Sign conditions may be disconnected and do not certify constant topology; arbitrary nonlinear intersections still require proved child reductions. Parameter-coefficient poles have explicit equality fibers retained as unknown, since geometry is undefined there; valid sibling cells still run. Callers can supply proved cells for broader families. The ordinary budget exception signals an oversized partition. Automatic partition evidence proves coverage, not successful integration of every child; inspect each outcome.

## Singular intersections in triangular polynomial charts

For three or more coordinates within the hypersurface budget, selected product singularities admit a global triangular polynomial chart. Each chosen factor is affine in its pivot coordinate, polynomial in earlier coordinates, and has a proved finite real constant nonzero slope. The inverse and constant absolute Jacobian are exact. The mapped domain must admit certified Cartesian pieces, and the transformed density must fit the existing product-cutoff moment engine.

For `u=x`, `v=y-x**2`, `w=z-x*y`, the chart has Jacobian one. On `abs(u),abs(v),abs(w)<=1`, density `1/(u*v*w)**2` with the original shared prescription `abs(u*v*w)>epsilon` has finite-part value -8. Replay verification reconstructs the chart, cutoff and Jacobian; changing stored chart evidence fails verification. It remains algorithm replay, not an independent general convergence theorem.

Repeated factors and asymmetric mapped boxes use the original product cutoff. General tangent intersections, arbitrary implicit hypersurfaces and domains without a certified mapped-box decomposition remain unresolved. A regularized value does not establish ordinary convergence.

Outer simplification preserves certified hypergeometric summands, avoiding per-index Lerch-function quadrature.

Regression evidence: `tests/test_performance_methods.py`; full-suite compatibility includes prior geometry, convergence and regularization tests. Benchmarks accompanying the delivery compare the prior working tree with these rules and report measured cases, rather than promising a universal speedup.

## Proved cutoff shortcuts

A radial polynomial profile with strictly integrable local absolute moments and
no singularities away from its center has the same limit for every shrinking
positive-definite geometric exclusion. After a neighborhood inside the original
domain is certified, its ordinary domain integral avoids a redundant curved CAD
hole. The evidence is local cutoff-shape independence; it does not assert
global convergence on an arbitrary unbounded domain. Threshold equality or any
unproved profile retains the original cutoff construction.

## Full-line shifted quartic moments

The fast rational portfolio also handles `P(x)/(A*t**4+B*t**2+C)` with `t=x-h`, numerator degree at most two, real finite denominator coefficients and `A>0`. It proves positivity using either `C-B**2/(4*A)>0`, or `C>0` and `B>=0`. The first test implies `C>0` itself and avoids expensive reasoning about nested radicals. A proved negative overall denominator sign is normalized along with the numerator.

With `D=B+2*sqrt(A*C)>0`, the zeroth and second moments are
`pi/(sqrt(C)*sqrt(D))` and `pi/(sqrt(A)*sqrt(D))`; the odd first moment vanishes by absolute integrability. The numerator degree bound gives at least quadratic tail decay. Poles, unknown leading signs, non-biquadratic quartics, half-lines and cubic numerators decline this shortcut. A cubic odd numerator cannot acquire an ordinary zero value from this rule.

```python
import sympy as sp
from multiple_integrate import integrate_result
x, b = sp.symbols("x b", real=True)
r = integrate_result(1/(1+(x*x+2*x+b)**2), (x, -sp.oo, sp.oo))
assert r.method == "quartic_rational_moments"
assert r.absolute_convergent is True
assert sp.simplify(r.value.subs(b, 1)-sp.pi/sp.sqrt(2)) == 0
```

Regression coverage: `tests/test_quartic_moments.py`, `tests/test_series_evaluation.py`, `tests/test_parameter_geometry_extensions.py`.

Further supported cases and prerequisites are described in [the extension guide](nonlinear-moments-and-singularities.md). Regression coverage: `tests/test_truncated_nonlinear_horns.py`.
