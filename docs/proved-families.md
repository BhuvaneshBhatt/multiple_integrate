# Structured moments and convergence proofs

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

These strategies cover sufficient, explicitly bounded families, rather than general algorithms for all nonseparable densities or singular sets. Executable prerequisites and independent oracles live in `tests/test_gap_improvements.py`.

## Scoped and intrinsic convergence

Parameter-dependent quadratic and radial answers retain ordinary and absolute convergence claims on their individual conditions. For example, on `b >= 0`, the integral of `(b+x**2+y**2)**-2` over the plane is finite for `b > 0` and divergent for `b = 0`. Its positive-branch evidence does not certify the whole parameter domain.

Intrinsic reductions aggregate absolute convergence only after every chart, including zero-dimensional direct values, has been proved finite or absolutely integrable. A chart proof includes its metric density and limits. Nested evaluation suppresses child evidence; complete parent reductions explicitly aggregate it.

Singular diagnostics distinguish `complete` (every budgeted geometry query was attempted) from `geometry_certified` (the algebraic denominator factorization and the required dimension/regularity fields were established). A known singular component can be certified as singular; certification does not mean that it is smooth. Neither field certifies integrability or describes every possible nonalgebraic function singularity. Structured result diagnostics expose `singular_denominator_geometry_certified`.

## Exact parameter specialization

`integrate_cells` infers substitutions from conjunctive, linear equalities among the declared parameters. `ParameterCell(Eq(a, 0))` therefore needs no manually repeated `{a: 0}`. Linear systems and relations such as `a = b + 1` are reduced exactly. Each inferred or caller-supplied replacement still has to follow from the complete cell condition. Generic inequalities do not select sample parameter values. Nonlinear equalities and disjunctions are not converted into substitutions by this route. Full-dimensional affine box images additionally lower through an invertible real linear map with a proved nonzero determinant. A bounded search tries constraint bases, checks the transformed axis-wise formulas, and retains the exact absolute determinant. Lower-dimensional affine images and prescribed PV/finite parts do not use this ordinary-measure shortcut.

## Mixed Gaussian and inverse quadratic moments

Full-space densities of the form

```text
P(x) exp(-a q(x)) / (b + q(x))**s
```

are supported when `q` is a positive-definite translated quadratic form, `a > 0`, `b > 0`, `s > 0`, and all coefficients are finite. Whitening and angular polynomial moments reduce the answer to radial moments. A radial moment with weight `r**(2*alpha-1)` is

```text
b**(alpha-s) G^{2,1}_{1,2}(a*b | 1-alpha ; 0, s-alpha) / (2 Gamma(s)).
```

On finite balls and shells the mixed route supports positive integer `alpha` up to 16, using finite sums of upper incomplete gamma functions. Unsupported fractional radial moments decline this native route. Odd moments vanish only after absolute integrability is established. Integer incomplete-beta moments also simplify to finite power/logarithm sums when the same bound applies.

## Anisotropic moments and local proofs

The full-space family `P(x)/(b + sum(c_i*x_i**m_i))**s` supports strictly positive finite `b,c_i`, positive even integer `m_i`, and finite polynomial coefficients. For a monomial with exponents `p_i`, set `alpha_i=(p_i+1)/m_i`. All terms must satisfy `s > sum(alpha_i)` before evaluation or odd cancellation. The even moment is

```text
b**(sum(alpha_i)-s) Gamma(s-sum(alpha_i))/Gamma(s)
  * product(2 Gamma(alpha_i)/(m_i*c_i**alpha_i)).
```

Near the origin with zero shift, absolute integrability instead requires every local weight to satisfy `sum(alpha_i) > s`. Constant-density critical or supercritical poles receive a negative convergence claim. Cross terms, odd defining powers, and unproved coefficient signs decline this route. These tests are sufficient for polynomial sums and do not claim necessary conditions for every cancellation.

## Shared cutoffs on clipped affine and curved domains

The `|g| > epsilon` prescription handles affine intersections whose transformed domain is an axis-intercept triangle in each occupied orthant. Convex linear inequalities are checked at all three vertices; every remaining inequality must hold on the entire triangle. The implementation integrates rational Laurent moments over the actual clipped triangle and retains the determinant, cutoff window, and source geometry. Additional downward-closed clipped polygon envelopes are supported; see [Geometry extensions](geometry-extensions.md).

A global polynomial graph `y=h(x)` with nonzero constant normal coefficient is straightened by a triangular map. A graph factor alone, or its product with one affine base-coordinate factor, can use the existing shared product cutoff when the transformed domain is one box. Both coordinate orders are tried. The exact constant Jacobian and original defining equation remain in provenance. Repeated factors preserve their prescribed powers. Curved domains that do not map to a box remain unsupported by this route.

For example, `g=x*(y-x**2)` on `-1<=x<=1`, `-1<=y-x**2<=1` gives finite part `4` for `1/g**2`. A different clipped domain can give a different finite part. Replay verification checks these geometry prerequisites and rejects missing regulators; it reuses the same family algorithm, so it is not an independent integration proof.

## Cusp blowup and multivariate convergence

On `0<x<A`, `y**2<=k*x**3` with finite `A>0` and `0<k<=1`, the density `P(x,y)/(x**3-y**2)**s` admits the substitution `y=x**(3/2)*t`. The Jacobian is `x**(3/2)` and the curved boundary pole is excluded by `k<1`. At contact `k=1`, the effective pole order must additionally satisfy `s<1`; exact numerator vanishing reduces that order. A monomial `x**p*y**q` is absolutely integrable if

```text
5/2 + p + 3*q/2 - 3*s > 0.
```

When every term meets this threshold, even moments are exact incomplete-beta expressions and odd terms vanish. Constant numerators at or below the threshold produce signed divergence with explicit ordinary and absolute evidence. Other unproved cases decline the route. This is a concrete multivariate blowup proof; arbitrary cusp neighborhoods, intersecting curved hypersurfaces, and general resolution of singularities remain capability gaps.

## Verification limits

Independent high-precision radial and blowup quadratures test the exact formulas. They are numerical consistency checks rather than rigorous error-bound certificates. Geometry and degree/term budgets still apply. Automatic parameter partitions have the supported scope described in [Parameterized integration](parameterized-integration.md). General curved principal-value evaluation and universal multivariate convergence classification remain unsupported.


## Optional isolated query deadlines

`IntegrationPolicy(backend_timeout_seconds=5.0)` opts into a separate spawned process for each Semialg adapter query (including intrinsic covers) and the general SymPy integration wrapper. A timeout returns an unresolved query or an unevaluated integral and records `backend_deadline`; it does not imply divergence. Workers are terminated and joined. Positive finite durations are required. Startup overhead counts toward each deadline. Normal execution retains direct calls when the option is `None`.

This is a per-call limit, not a total integration deadline. Other native strategy calls, Funcprops queries, direct symbolic calls outside the general wrapper, numerical consistency quadrature, serialization/cleanup overhead, and repeated queries are not bounded by this option. Spawn requires an importable Python main module; applications must use the usual `if __name__ == "__main__":` guard. Tiny limits commonly expire during startup.

## Independent ordinary-result consistency checks

`verify_finite_result_numerically(result, density, *ranges, digits=40, substitutions=None)` checks ordinary exact results or explicitly specialized conditional results on a finite, independently bounded one- or two-dimensional box. Coordinates must be distinct real symbols; all parameters must be resolved and continuity must be proved on the closed box. It performs quadrature at two working precisions and compares both refinement and the claimed result against a relative tolerance. The returned `NumericalVerification` explicitly sets `certified=False`; `consistent=None` means prerequisites or evaluation were unresolved. Changed claimed values are independently rejected. PV/finite-part values and singular boxes require other verification methods.

Native POSIX heuristic timers decline when a caller already owns an active real-time alarm. Otherwise their private interruption bypasses ordinary exception handlers, retries if swallowed, rejects late returns, and temporarily postpones automatic garbage collection. Handler, timer and GC state are restored on return or failure. These remain main-thread heuristic limits, not universal execution deadlines; explicit GC callbacks or non-interruptible native calls can still require process isolation.
