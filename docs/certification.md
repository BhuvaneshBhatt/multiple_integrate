# Certification and result semantics

Use `integrate_result()` to inspect `value`, `status`, `method`, `conditions`,
`parameter_strata`, `complete`, `convergent`, `attempts`, `limitations`, and
`diagnostics`. The flat `integrate()` API returns an expression or raises an
explicit error for an unsupported request. It is not a status protocol.

| Status | Meaning |
|---|---|
| `EXACT` | evaluated symbolic answer under the supplied assumptions and measure convention |
| `CONDITIONAL` | parameter/assumption branches; inspect every branch |
| `DIVERGENT` | the ordinary integral is classified as divergent |
| `UNKNOWN` | unresolved symbolic answer; no proof of finiteness or divergence |
| `UNEVALUATED` | evaluation was refused or unavailable, including a strategy budget |
| `PRINCIPAL_VALUE` | a prescribed principal value, distinct from the ordinary integral |
| `FINITE_PART` | a prescribed finite part, distinct from convergence |

There is no separate public status called “numerically verified”.
Numerical quadrature is independent test evidence, not an exact certificate.
The flat API can raise `NotImplementedError` for unsupported inputs; structured
APIs retain an UNSUPPORTED result. Validation errors still raise; do not rewrite
those errors as divergence. Inspect `limitations` and recorded attempts
for budget refusals. A `None` returned by a specialist geometry adapter is an
unavailable reduction, never an integral value.

Proof queries and refinement use explicit `assumptions=` and intrinsic symbol
facts. Temporary global SymPy `assuming(...)` facts do not become package
certificates or cached parameter prerequisites. Pass a fact explicitly when the
result should carry it. Symbolic backends may still produce a generic expression;
that expression does not prove convergence. A recursion-exhausted backend query
retains the original unevaluated integral, including its original coordinates
and endpoint orientation. See `tests/test_api_audit_fixes.py` for the temporary
fact, conflicting-cell, refinement and recursion regressions.

A generic formula does not settle transition loci. Parameter strata must be read
with their conditions; `complete=False` indicates unresolved cases. The real-parameter theorem for
`exp(-a*x)` over the positive half-line distinguishes convergence for `a>0`
from divergence for `a<=0`; the divergent case may retain an unevaluated value
with an explicit divergence status.

Exact status describes the evaluator's symbolic result. It is not a claim that
arbitrary symbolic expressions carry a replayable formal proof. In particular,
finite iterated values do not by themselves justify a Fubini interchange for a
conditionally convergent double integral. Convergence reasoning and geometry
certification are separate obligations.

See [structured results](structured-results.md) and [worked failures](worked-failures.md).
Regression coverage: `tests/test_convergence_semantic_corpus.py`,
`tests/test_public_contracts.py`, `tests/test_symbolic_conditional_convergence.py`.

## Uniform outcomes

Both `IntegrationResult` and `RegularizationResult` expose `outcome` with the
same `IntegrationOutcome` values: `FINITE`, `DIVERGENT`, `CONDITIONAL`,
`UNSUPPORTED`, and `UNKNOWN`. `finite` is True for a finite evaluated answer,
False for divergence, and None when a single finiteness conclusion is unavailable.
`status` retains detail such as exact, principal value, finite part, or a budget
refusal. A principal-value or finite-part outcome never asserts ordinary convergence.

The structured APIs retain an unevaluated request and a reason for an unsupported
route. A supported geometry route that cannot certify its reduction returns
UNKNOWN instead. A configured budget refusal retains UNEVALUATED status and
UNKNOWN outcome. Invalid inputs still raise validation errors, and the flat
`integrate()` API continues to raise for unsupported operations. Unexpected
backend errors and contract violations are not converted into mathematical outcomes.

An unresolved Integral/Limit or undefined `nan` is UNKNOWN; `nan` alone is not
proof of divergence. Piecewise answers expose disjoint cases and unresolved
complements, including Piecewise expressions nested in arithmetic. Regularized
results use the same classifier: an infinite PV is DIVERGENT with `exact=False`,
not a successful principal value. `complete=True` can mean divergence was fully
resolved; it does not mean a finite answer exists.

## Coverage evidence

Both result types expose a tuple of `CoverageEvidence` records. Each record has
`scope`, `complete`, `method`, and immutable key/value `details`. Scope matters:

- `integration_domain`: a native strategy uses the full requested domain.
- `returned_domain_pieces` / `returned_cutoff_pieces`: every backend-returned
  cylindrical piece was consumed, relying on the backend reduction contract.
- `ambient_null_set`: a certified dimension drop gives zero ambient measure.
- `intrinsic_cover`: a verified Semialg cover has no unresolved target-dimensional
  strata and all its pieces were consumed.
- `atlas_charts`: every requested chart was integrated or independently certified
  empty. Failed/nonexplicit patches without emptiness evidence are refused.
- `angular_domain`: exact sphere moments cover the entire angular domain.
- `cutoff_domain`: the Cauchy path uses all interval segments or the whole-interval
  SymPy PV evaluator.
- `parameter_space`: explicit disjoint cases retain an unresolved complement.
  This can be complete even when some case values remain UNKNOWN.
- `problem_identity`: a cache hit matches the complete structured region,
  coordinate order, integrand, and assumptions. This is cache-identity evidence,
  not a newly reconstructed decomposition certificate.

These are scoped strategy/backend claims, not universal formal proofs. In
particular, consuming all returned pieces is not an independent proof that a
backend's geometric decomposition is correct. Coverage does not prove absolute
convergence, justify Fubini, or turn UNKNOWN values into finite ones. Failed atlas
coverage is retained in the structured result even when no value can be produced.

Regression coverage: `tests/test_uniform_protocol.py`.

## Convergence claims and verification strength

`convergence_evidence` separates the evaluator's `iterated` or `prescribed`
outcome from `ordinary` and `absolute` convergence claims. Every record carries
its kind, method, condition, and a three-valued `established` field. The existing
`convergent` flag describes the selected evaluator/prescription; it must not be
read as an absolute-convergence certificate. `ordinary_convergent` and
`absolute_convergent` report unconditional evidence, or `None` when not proved.

Sufficient rules prove ordinary and absolute convergence for supported continuous
densities on finite boxes, local and full-space radial powers, absolute products,
and polynomial moments of positive definite Gaussians. See `scoped-convergence.md`. Parameter theorem cases supply conditioned
ordinary claims. Other evaluated results retain unknown ordinary/absolute claims
rather than inferring them from a closed expression. For example, the two orders
of `(x**2-y**2)/(x**2+y**2)**2` on `[-1,1]**2` produce `-pi` and `pi`; neither
answer establishes an ordinary domain integral or a Fubini interchange.

SymPy accumulation ranges (`AccumBounds`) are UNKNOWN, never successful finite
answers. An oscillatory divergence theorem is not inferred just from that type.

Regularization verification reports `strength="constant_consistency"` for
constant-only equality checks and `strength="cutoff_replay"` when it replays
cutoff asymptotics and constant extraction. `valid=True` has the scope of that
strength. Neither independently derives the cutoff integral or generally proves
its geometric domain. The regular-density shortcut has a separate
`strength="dominated_convergence"` that replays prerequisites, its polynomial
cutoff geometry, and the ordinary value. Failed results carry `valid=False`.

Auxiliary singular diagnostics use the actual request domain and prune supersets
of certified empty intersections. `IntegrationPolicy(max_singular_queries=128)`
sets their separate query budget; zero disables queries. Partial diagnostics are
marked in `diagnostics` and `limitations` without erasing an evaluated value.
Closed component/intersection loci may overlap and are not a disjoint cover.

Regression coverage: `tests/test_robustness_evidence.py` and
`tests/test_algebraic_pv_and_singular_geometry.py`.

For singular denominator diagnostics, `complete` means the requested queries were attempted within budget; `singular_denominator_geometry_certified` additionally requires certified factorization and the required dimension/regularity fields. Neither certifies convergence. See [proved families](proved-families.md).

## Series and exceptional parameter fibers

Infinity in an exact `Sum` index bound is not an infinite value. The radial series carries a finite-interval absolute-convergence proof; arbitrary unresolved sums remain unknown. Positive-definite Gaussian and intrinsic-dimension claims require strict prerequisites across the complete active cell. Coverage, convergence, finiteness and numerical agreement are independent checks. See [radial series](geometry/radial-series.md) and [parameter geometry](geometry/parameter-geometry.md).

## Convergence consensus and bounded approximations

Consistent unconditional claims are combined. Conditional claims can establish a global summary only after their union is proved to cover the assumption domain, with one consistent established truth value. Unknown complementary cells and conflicting facts keep the summary unknown. Null-set evidence includes the source problem identity before it can justify an ambient convergence claim. Regression coverage: `tests/test_convergence_summaries.py`, `tests/test_parameter_geometry_extensions.py`.

The specialist [series evaluator](series-evaluation.md) returns an exact partial sum and an analytic remainder bound. Its `certified` flag refers to that bound; requested accuracy is met only when `converged=True`. Floating evaluation introduces additional rounding error and is not an interval enclosure.
