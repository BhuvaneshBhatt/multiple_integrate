# Testing

The test suite protects mathematical results, strategy behavior, and package architecture. New tests should fail for a meaningful regression, not merely because an implementation detail changed.

## Running the suite

From an installed development checkout:

```bash
pytest
```

From a source checkout without an editable install:

```bash
PYTHONPATH=src pytest
```

Run the static checks before submitting changes:

```bash
ruff check src tests
ruff format --check src tests
python -m compileall -q src tests
```

## What to test

For a new mathematical capability, prefer a small set of strong contracts:

- an exact success case with a known result;
- a nearby case where the specialized strategy must decline or return a structured unresolved status;
- invalid-input cases for real API preconditions;
- an assumption pair when symbolic parameters control convergence or applicability;
- an independent oracle or metamorphic identity when another evaluation route exists;
- a performance guard only when runtime is part of the behavior being protected.

Avoid assertions such as `result is not None` when the mathematical value, status, method, limits,
dimension, or certificate can be checked directly.

## Range and region contracts

Region-related tests should distinguish integration-domain recognition from canonical geometry.
MultipleIntegrate may inspect its internal structured domain models when testing range recognition or
planner dispatch. Canonical geometry operations belong to public `semialg` APIs.

Useful contracts include:

- integrating `1` over a canonical region equals its measure at the same measure dimension;
- each centroid coordinate equals the corresponding first moment divided by the measure;
- recognized ranges and equivalent canonical geometry integrate the same nonconstant expression;
- lower-dimensional geometry has zero ambient measure and the correct intrinsic measure;
- affine or parametrized reductions preserve the integral after the correct metric factor is applied.

## Lower-dimensional matrix

Embedded points, curves, and surfaces are tested as a dimension matrix rather than as isolated
examples. Each case checks ambient and intrinsic semantics and includes a nonconstant moment where
possible. This catches missing chart conditions and Jacobian/Gram factors that a constant-integrand
test can miss.

## Assumption pairs

When a theorem requires symbolic conditions, test the same problem twice:

1. without enough assumptions, asserting the actual conditional or unresolved structured status;
2. with sufficient assumptions, asserting the exact branch value.

Also test contradictory assumptions at the normalization boundary. Do not encode hidden positivity or
orientation assumptions in fixtures merely to make a strategy succeed.

## Cross-strategy oracles

When two independent routes apply, compare them. Good oracle pairs include:

- recognized ranges versus canonical semialg geometry;
- a structural region shortcut versus semialgebraic reduction;
- an intrinsic chart reduction versus an explicit parameter integral;
- a coordinate transform versus direct integration;
- a symmetry shortcut versus the unreduced integral.

The two routes should be genuinely independent enough that a shared bug is unlikely.

## Sampling tests

Sampling belongs to semialg, but MultipleIntegrate tests may use samples to validate interoperability.
Representative samples should be tested for membership rather than exact coordinates. Random sampling
tests should use fixed seeds, verify membership, and use stable distributional invariants with
comfortable tolerances. Do not use random sampling as the oracle for an exact symbolic result unless
the test is explicitly numerical.

## Unevaluated results

Mathematical tests do not skip dynamically because a result happens to remain unevaluated. Each case
has a permanent contract: exact success, a specific structured unresolved result, or an unsupported
operation. Environment capability skips are appropriate only when the environment lacks a feature
needed by the test, such as a POSIX timer.

## Architecture contracts

The suite statically enforces package ownership:

- canonical geometry, semialgebraic decision procedures, intrinsic charts, measure, centroid, and sampling belong to `semialg`;
- MultipleIntegrate owns integration-domain recognition, strategy selection, convergence, exact integration formulas, and regularization;
- only designated boundary modules may import `semialg` directly;
- strategy modules call small adapter functions instead of private semialg modules;
- integration-domain classes must not grow canonical geometry services;
- canonical geometry constructors must not leak into the MultipleIntegrate package root.

If a strategy needs a new semialgebraic operation, add the narrowest useful adapter function and test that boundary. Don't bypass the boundary from planner code.

## Documentation examples

Core examples in the quick start and region guides have executable regression tests. When changing a public example, update its test at the same time. Documentation should show mathematically checked values rather than illustrative output that is not guaranteed by the current API.

## Test organization

Focused files cover integration families, assumptions, convergence, singularities, regularization,
range recognition, lower-dimensional geometry, strategy oracles, public contracts, architecture, and
performance. Broad end-to-end behavior remains in `test_integrate.py`, while reference cases live in
`test_reference_examples.py` and `test_reference_examples_notebook.py`.

## Geometry, parameter cells, and release gates

`test_geometry_reference_corpus.py` records exact region oracles, measure
conventions, and route names. Its parameter families retain empty fibers,
transition points, disconnected cells, and boundary coincidences.
`test_convergence_semantic_corpus.py` pairs valid exact prerequisites with
nonintegrable neighbors and checks status rather than accepting any unevaluated
expression as evidence of divergence. `test_geometry_properties.py` generates
30 deterministic polynomial-box examples with exact rational oracles;
`test_semialg_documentation_traceability.py` connects claims to tests and executes
the parameter guide example.

Run the complete suite with `python -m pytest`, the normal suite with
`python -m pytest -m 'not slow'`, and the slow suite with
`python -m pytest -m slow`. Run the generated corpus separately with
`python -m pytest tests/test_geometry_properties.py`.

Build documentation with `MI_DOCS_OFFLINE=1 sphinx-build -W -b html docs docs/_build/html`
when remote intersphinx inventories are unavailable. This still checks all local
content and API rendering. Use `python -m build` and `python -m twine check dist/*`
for packaging. Install the wheel into a clean environment, copy tests and their
documentation fixtures outside the source checkout, and run there without a
source `PYTHONPATH`. The Semialg release workflow tests the 1.3.1 floor and latest
published release. The sibling workflow covers unreleased local geometry work.

The installed contract script defaults to the required dependencies. Pass `--with-asymptotic` to additionally require and check the optional real backend.
The optional backend must actually be installed and checked before reporting that gate as passed. Base-dependency tests do not certify optional-backend availability.

## Casebook, generated geometry and notebook gates

Run `python tools/sync_geometry_examples.py --check` and the casebook example script. New generated/mathematical tests live in `tests/test_geometry_casebook_properties.py`; serious boundary regressions live in `tests/test_serious_boundary_regressions.py`. `tests/test_casebook_and_docs.py` executes snippets and validates named claims. The clean-kernel notebook commands and explicit manual-cell reasons are documented in [the casebook](geometry-casebook.md). CI builds documentation strictly in offline mode; its separate online linkcheck gate validates remote links when available. Source and installed-wheel notebook gates use required dependencies plus the notebook-test extra.

## Profiling and slow selections

The slow marker covers improper periodic-tail classification, high-precision
radial-series verification and general geometric cutoffs. Generated tests retain
their full deterministic examples. Use `--durations=20` to find expensive tests
outside that selection, and `python -m cProfile -o tests.prof -m pytest -m slow`
to distinguish collection time, strategy time and numerical oracle time.

Certified infinite summands remain unchanged during outer simplification. This
avoids per-index Lerch-function quadrature while preserving the exact series and
independent radial quadrature tests. Geometric cutoffs can omit a curved CAD hole
only when a proved globally radial profile is locally absolutely integrable and
continuous away from its center. Unproved or threshold-equality cases retain the
original prescription path.

Capability limits and reproducible evidence-summary issues belong in
[limitations and known issues](limitations.md), with precise consequences.

## Exceptional fibers and generated mathematical families

The ordered mathematical regression layers are:

1. `tests/test_exceptional_fiber_matrix.py`: exact endpoint/decay thresholds,
   zero amplitudes, undefined poles, boundary coincidence and dimension drops.
   A complete parameter covering does not imply that every fiber is finite.
2. `tests/test_generated_mathematical_families.py`: 54 deterministic generated
   cases across endpoint vanishing, shifted gamma moments, affine Jacobians,
   oriented parameter cells and shared singular cutoffs. Together with the two
   existing generators, the suite now has seven generators and 104 examples.
3. `tests/test_independent_improper_oracles.py`: integrated sine-series and
   oscillatory quadrature references, hand-derived cutoff constants, and
   precision-escalated short gamma-tail differences. Tiny integrals are scaled
   before quadrature so an absolute error target cannot hide relative error.
4. `tests/test_transformed_evidence_contracts.py`: transformed values **and**
   scoped convergence/coverage, missing cells, and corrupted certificates whose
   final numerical value is unchanged.
5. `tests/test_bounded_dispatch_contracts.py`: pre-expansion work bounds,
   unsuccessful recognition, general-backend avoidance, cache scope/eviction,
   and deadline forwarding/worker cleanup using deterministic call assertions.

Run the five files together with `--hypothesis-show-statistics`. Every generated
example has a bounded degree, dimension and coefficient size, an independent
closed-form reference or invariant, and a fixed mathematical contract. Add new
families here rather than increasing unrestricted random expression complexity.
Numerical agreement supports a regression oracle; it does not establish a
convergence theorem or independent verification of arbitrary provenance.

## Executed environment evidence

A workflow definition is not proof that a matrix ran. Record the interpreter, platform, dependency versions, imported package path, installed-wheel status, exact pytest counts and any capability skips for each executed combination. Keep source runs, wheel runs, published dependency runs and unreleased local sibling runs distinguishable. Offline copied dependency snapshots must be identified, pass `pip check`, and use ABI-compatible binary dependencies.

The release workflow covers the minimum and latest Semialg on Python 3.11-3.14. The portability workflow exercises the focused mathematical contracts on Linux, macOS and Windows, and has a separately enabled manual gate for the optional `asymptotic` extra against its declared dependency floor. Locally executed platforms must be reported separately; configuring Windows/Linux CI is not a local execution claim.

The dependency floors are semialg 1.3.2, funcprops 0.3.0 and optionally asymptotic 0.2.1. The optional job is enabled explicitly through its workflow-dispatch input. Verify that the backend imports successfully before claiming optional-backend coverage: a fallback-only run does not validate `asymptotic` integration. The package Python floor is 3.11, matching required `semialg` releases.

