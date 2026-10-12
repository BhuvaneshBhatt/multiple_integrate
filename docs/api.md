# Public API contracts

This page explains the roles of the public API. Exact signatures and member listings are generated
from source docstrings in the [Sphinx API reference](api/index.rst).

## Integration

- `integrate` returns the mathematical result without extra information.
- `integrate_result` evaluates the same problem and adds status, parameter and singularity strata, strategy attempts, limitations, and diagnostics.
- `regularized_result` returns regulator-level provenance for principal-value and finite-part calculations.

Domains can be supplied as inner-first range tuples, canonical `semialg` geometry, or a semialgebraic Boolean formula. Canonical geometry and raw formulas require `variables=` to identify ambient integration coordinates. See the [ranges and regions guide](concepts/ranges-and-regions.rst).

## Measure and lower-dimensional geometry

The default `measure_dimension="ambient"` uses ordinary ambient Lebesgue measure. For curves, surfaces, and other embedded geometry, `measure_dimension="intrinsic"` uses semialg's verified parametric charts and induced Hausdorff metric factors. See [measure and dimension](concepts/measure-and-dimension.rst).

Region measure, centroid, membership, transformations, and sampling are geometry operations and are provided by `semialg`. MultipleIntegrate consumes those regions for integration.

## Policy and status

`IntegrationPolicy` contains named `StrategyBudget` values for potentially expensive exact
semialgebraic, hypersurface, and asymptotic work. `StrategyBudgetExceeded` signals that policy,
rather than mathematics, rejected an expensive strategy. `IntegrationStatus` classifies structured
outcomes.

## Principal-value and finite-part prescriptions

`CauchyPrincipalValue`, `IteratedPrincipalValue`, `GeometricPrincipalValue`, `HypersurfacePrincipalValue`, and `HadamardFinitePart` make nonordinary integral semantics explicit.
`verify_regularization_result` replays locally checkable certificate claims. Read the [regularization guide](regularization.md) before choosing a prescription.

## Domain helpers

`region_from_ranges` inspects nested bounds and returns the structured domain description used by the planner. `Boole` constructs Boolean restrictions used by integration-domain workflows. These helpers do not replace canonical semialg geometry.

## Expert APIs

Specialized strategy modules and structured integration-domain classes are implementation APIs for contributors and advanced diagnostics. They are not exported from the package root. You should use package-level integration functions and public semialg geometry in application code.

## Root export boundary

The package root is the stable application-facing facade. Its 27 exports are grouped by the choices a caller must make; it does not mirror every specialist module.

| Root exports | Why they belong here |
|---|---|
| `integrate`, `integrate_result`, `regularized_result`, `integrate_cells` | Main evaluation entry points |
| `ParameterCell`, `IntegrationPolicy`, `StrategyBudget`, `StrategyBudgetExceeded` | Request configuration and explicit policy handling |
| `CauchyPrincipalValue`, `IteratedPrincipalValue`, `GeometricPrincipalValue`, `HypersurfacePrincipalValue`, `HadamardFinitePart` | Explicit prescription choices |
| `IntegrationResult`, `IntegrationOutcome`, `IntegrationStatus`, `RegularizationResult`, `UnevaluatedRegionIntegral` | Returned values and outcome inspection |
| `CoverageEvidence`, `ConvergenceEvidence` | First-class evidence interpretation |
| `NumericalVerification`, `RegularizationVerification`, `verify_finite_result_numerically`, `verify_regularization_result` | Optional result verification |
| `Boole`, `region_from_ranges`, `clear_cache` | Small domain conveniences and explicit cache control |

Keep nested case, stratum, provenance and strategy-attempt types in their owning specialist modules. Importing a returned object's metadata type for advanced tooling does not require making all such types root exports. The smaller `multiple_integrate.api` facade declares its own limited set; every declared name resolves, including `UnevaluatedRegionIntegral`. Application code should normally use the root facade. The existing root names remain supported.

An exact-looking parameter expression alone is insufficient to establish a uniformly finite result. Unproved denominator exclusions and logarithm/gamma poles produce unknown/incomplete structured outcomes, preserving the symbolic formula. Strict or explicit nonzero cells can resolve them. Negative nonzero logarithm arguments can have finite complex values; finiteness does not assert a real branch. Gamma pole exclusions cover positive arguments and certified noninteger or nonreal arguments; unresolved pole sets remain unknown.

## Calling syntax

```python
import sympy as sp
import semialg
from multiple_integrate import integrate
x, y = sp.symbols("x y", real=True)
# Ranges are innermost first; the inner bound may depend on an outer variable.
assert integrate(1, (y, 0, 1-x), (x, 0, 1)) == sp.Rational(1, 2)
# A Boolean formula specifies the domain directly.
triangle = sp.And(x >= 0, y >= 0, x+y <= 1)
assert integrate(1, region=triangle, variables=(x, y)) == sp.Rational(1, 2)
# Or provide a canonical Semialg region with explicit coordinate order.
assert integrate(1, region=semialg.Box(((0, 1), (0, 2))), variables=(x, y)) == 2
```

The main forms are `integrate(f, *ranges, **options)` and `integrate(f, region=domain, variables=(...), **options)`. `assumptions=` supplies whole-domain parameter prerequisites; `measure_dimension=` chooses ambient or intrinsic measure. Use `integrate_result` with the same inputs to inspect evidence, and `integrate_cells` for a parameter partition. See the generated API reference for the complete keyword signature.

## Indicators and ordered branches

`Boole(condition)` is a MultipleIntegrate helper returning the SymPy indicator `Piecewise((1, condition), (0, True))`; it is not a separate SymPy class.
`integrate` accepts these indicators and SymPy `Piecewise` expressions. Branch conditions follow first-match semantics. Unsupported geometry or unresolved convergence can still leave an unevaluated result; support is not universal. A Piecewise expression without an exhaustive default retains its undefined region.

Verified scalar co-area reductions are a specialist API in `multiple_integrate.coarea`; see [coarea](coarea.md). Their evidence objects and chart candidates are kept out of the application-facing root facade.
