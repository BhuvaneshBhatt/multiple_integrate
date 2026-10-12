# Verified co-area reductions

The specialized `multiple_integrate.coarea` API integrates along level fibers and
then over their levels. Scalar maps support affine boxes and Euclidean radial
shells. Vector maps support independent affine projections and supplied verified
charts, with level dimension strictly smaller than ambient source dimension.
Selected angular, algebraic and canonical planar polar families have explicit
proof obligations. General implicit-fiber or critical-value stratification is not
implemented.

`reduce_coarea(...)` returns an immutable `CoareaReduction`, its `.pieces`,
formal `.integral`, and geometry coverage. `.evaluate()` or
`integrate_coarea(...)` returns an `IntegrationResult`. Coverage does not prove
convergence: finite signed values require L1 evidence. One-sign densities may use
Tonelli to prove a finite value or signed divergence. Conditional iterated values
remain unknown as coarea values. Failed geometry returns an unknown result;
the reduction constructor raises `UnresolvedIntegration`. Malformed inputs raise
`ValueError`. Coordinate symbols must use `real=True` without additional sign
or arithmetic assumptions; put coordinate restrictions in the domain or ranges.
Parameter symbols may retain their own facts. Extra coordinate flags decline
rather than certify geometry or convergence outside their declared scope.

## Affine maps and the map Jacobian

Use an explicit `level_map`, source `variables`, and independent inner-first
`ranges`. Coupled scalar affine maps currently support finite planar boxes;
a single active coordinate works in arbitrary supported dimension. Rates, corner
ordering and bounds must be proved throughout the supplied parameter cell.

```python
import sympy as sp
from multiple_integrate.coarea import CoareaChart, reduce_coarea, integrate_coarea
x, y, z, t, s, u, a = sp.symbols("x y z t s u a", real=True)
affine = integrate_coarea(sp.exp(-x-y), level_map=x+y, variables=(x, y),
                         ranges=((y, 0, 1), (x, 0, 1)))
assert sp.simplify(affine.value-(1-sp.exp(-1))**2) == 0
assert affine.complete and affine.absolute_convergent is True
```

For `g=3*x+4*y`, the map Jacobian is `5`, rather than a chosen pivot rate.
Solving for `x` gives fiber metric `5/3`, and their ratio is `1/3`, the absolute
chart volume determinant. The implementation checks this identity.

For `g: R^m -> R^k`, the Jacobian is `sqrt(det(Dg*Dg.T))`. The fiber metric
uses the last `m-k` chart columns. The first `k` coordinates are levels, in map
order; all ranges remain inner-first. Independent affine projections are generated
automatically. Coupled vector affine maps require supplied charts and source
coverage proofs; they are not approximated by coordinatewise Jacobians.

```python
vector = reduce_coarea(x+y+z, level_map=(2*x, -3*y), variables=(x, y, z),
                      ranges=((z, 0, 1), (y, 0, 1), (x, 0, 1)))
assert vector.pieces[0].jacobian == 6
assert vector.pieces[0].fiber_metric == 1
assert vector.evaluate().value == sp.Rational(3, 2)
```

## Radial levels and angular amplitudes

For a radius or squared radius, `level_bounds=(lower, upper)` defines the source
shell. Prove `0<=lower<upper`; `upper=oo` is allowed. Equal nonnegative levels
have zero ambient measure. A finite real center is supported. Squared-radius
weight is `S_(n-1)*t**(n/2-1)/2`, where
`S_(n-1)=2*pi**(n/2)/gamma(n/2)`. Radius weight is `S_(n-1)*t**(n-1)`.
The critical center is an ambient null set. Unbounded maps use locally Lipschitz
exhaustion; origin and tail convergence remain separate obligations.

Polynomial angular amplitudes use spherical gamma moments and an absolute
monomial majorant. In the plane, selected finite Fourier polynomials with integer
frequencies of magnitude at most eight are also supported. The angle is the
canonical `atan2(y-cy,x-cx)` around the radial center. Selected rational angular
amplitudes have a denominator `A+B*cos(theta)` with a proved uniform gap
`|A|>|B|`; the zero `B` fiber must be known or specialized separately.

The reciprocal cosine moments obey `J_0=2*pi/sqrt(A*A-B*B)` for positive `A`,
and `J_n=(M_(n-1)-A*J_(n-1))/B`, with `M_n` the ordinary circle cosine moment.
Even sine powers reduce using `sin(theta)**2=1-cos(theta)**2`; odd sine moments
vanish. Negative denominators are normalized by their proved sign. Absolute
majorants divide by the uniform gap, so cancellation cannot certify an
otherwise nonintegrable density.

```python
q = x*x+y*y
angular = integrate_coarea(x*x*sp.exp(-q), level_map=q, variables=(x, y),
                            level_bounds=(0, sp.oo))
assert angular.value == sp.pi/2 and angular.absolute_convergent
fourier = integrate_coarea(sp.cos(sp.atan2(y,x))**2, level_map=q,
                          variables=(x,y), level_bounds=(0,1))
assert fourier.value == sp.pi/2 and fourier.absolute_convergent
reciprocal = integrate_coarea(1/(3+x/sp.sqrt(q)), level_map=q,
                             variables=(x,y), level_bounds=(0,1))
assert sp.simplify(reciprocal.value-sp.pi/sp.sqrt(8)) == 0
```

For example, `x/q**2` and `cos(atan2(y,x))/q` near zero remain unknown, even
though their signed angular moments vanish. A denominator touching zero,
noninteger Fourier frequency, non-Euclidean quadratic level or arbitrary angular
function declines the selected family.

## Supplied charts and excluded sets

`CoareaChart` is a candidate, not a trusted certificate. `coordinates` lists levels
first, `forward` follows source order, `inverse` follows chart-coordinate order,
and `ranges` reverse coordinate order. Rational, selected positive algebraic
branches and canonical planar polar charts are supported. Semialg checks both
inverse identities, the level identities, rank, metric, regular image, coverage
and disjoint chart interiors. Closed boundaries may coincide.

Chart and source exclusions require proved polynomial null-set certificates.
Denominators and determinants must remain nonzero on the regular domains.
An excluded center, seam or branch endpoint does not establish integrability near
it. Independent chart ranges can additionally use separable affine endpoint-order
and Laurent/log comparison proofs; each absolute factor must be integrable.
Unproved denominator gaps, hidden source branches, duplicate interiors and
parameter capture are rejected.

```python
chart = CoareaChart((t,u), (t*u,u), (x/y,y), ((u,0,1),(t,0,1)),
                    chart_exclusions=sp.Eq(u,0), source_exclusions=sp.Eq(y,0))
source = sp.And(x>=0, x<=y, y>=0, y<=1)
charted = integrate_coarea(y**sp.Rational(-3,2), level_map=x/y,
                          variables=(x,y), region=source, charts=(chart,))
assert charted.value == 2 and charted.absolute_convergent
```

## Caller-supplied parameter cells

`integrate_coarea_cells(...)` shares exact partition validation with
`integrate_cells`. Cells must be disjoint, involve declared real parameters only,
and carry substitutions implied by their conditions. Linear equality fibers are
specialized exactly. The uncovered complement is retained as unknown. Child
coverage remains associated with its parameter cell; convergence evidence is
conditioned on that cell.

Semialg's `chart_parameter_events` and `parameter_geometry_events` identify
bounded parameter-only rank, pole and boundary-coincidence candidates. These are diagnostics, not an automatically
certified partition. An equality fiber must not be discarded because it is null in
parameter space. When specialization loses map rank or collapses a chart range,
the ordinary source integral is evaluated separately when available. A constant
map has zero Jacobian but need not have a zero source integral. Undefined density
or geometry poles remain unknown.

```python
from multiple_integrate.coarea import integrate_coarea_cells
from multiple_integrate.parameter_cells import ParameterCell
cells = (ParameterCell(a<0), ParameterCell(sp.Eq(a,0)), ParameterCell(a>0))
parameterized = integrate_coarea_cells(x+y, cells=cells, parameters=(a,),
    level_map=a*x, variables=(x,y), ranges=((y,0,1),(x,0,1)))
assert parameterized.complete
assert all(case.value == 1 for case in parameterized.cases)
```

## Ordinary integration and bounded dispatch

Ordinary `integrate(...)` and `integrate_result(...)` try selected cheap co-area
routes before general fallback. Existing specialized moment and separation rules
keep precedence. Recognition supports polynomial amplitudes times
`exp(c-a*g**k)` with proved finite real coefficients and `a>0`, compact polynomial
and pole-free rational radial profiles, simple radial polynomial-valued Piecewise
profiles, integrable selected rational radial tails, and shifted isotropic
Gaussians. Compact pole-free rational densities that depend only on an affine level are
also candidates; separately dependent amplitudes keep the existing iterated
route.
Unsuitable oscillatory or mixed exponential phases decline before any child
integration.

```python
from multiple_integrate import integrate_result
ordinary = integrate_result(sp.exp(-(x+y)**3), (y,0,1), (x,0,1))
assert ordinary.method == "coarea_affine" and ordinary.absolute_convergent
```

Recognition caches both successes and failures. Semialg certificates are frozen
and cached by candidate, explicit parameter assumptions and proof budget.
Temporary global SymPy assumptions are ignored by geometry certification.
`clear_cache()` clears recognizer and geometry caches. Variable, operation,
degree, expansion and chart-count guards precede expensive proof work.
Regression tests count dispatches and proofs rather than impose machine timing
limits. `tools/benchmark_coarea_certificates.py` reports cold certification and
cached reuse separately; these measurements are not correctness gates.

## Ownership, references and scope

`semialg` owns `Chart`, `ChartCertificate`, `CoverCertificate`, `NullSetCertificate`,
`LinearMetricCertificate`, geometry event candidates, bounded proofs and caches.
The same metric certificate serves affine transformations, intrinsic affine
embeddings and boundary facets. MultipleIntegrate owns transformed densities,
fiber evaluation, convergence, parameter integration and strategy dispatch.

General nonlinear critical-set stratification, implicit fibers, arbitrary singular
contacts and automatic construction of general parameter partitions are possible
future improvements. Layer-cake integrates superlevel/sublevel volumes by Tonelli;
coarea integrates level fibers with the map Jacobian. Both are geometric reductions.

Regressions: `tests/test_coarea.py`, `tests/test_coarea_extensions.py`,
`tests/test_coarea_portfolio.py`, and Semialg `tests/test_chart_certificates.py`.
Every selected family has an executed example in the method catalogue or this guide;
the demo notebook also includes the new vector, angular and parameter-cell families.
