# Semialg integration boundary

MultipleIntegrate requires `semialg>=1.3.1`. Semialg owns region representations,
semialgebraic decision logic, parameter decomposition, intrinsic covers, and
algebraic geometry predicates. MultipleIntegrate owns integration planning,
coordinate and order selection, Jacobians used by its transformations, iterated
integration, integration-specific convergence reasoning, and result assembly.

Canonical objects include `semialg.Box`, `semialg.Point`, `semialg.Ball`,
`semialg.Sphere`, `semialg.SphericalShell`, and `semialg.Simplex`.
Raw Boolean formulas need explicit `variables=`; parameters must not be included
among the ambient integration coordinates. The adapter infers parameters from
free symbols outside those coordinates and forwards them explicitly to Semialg;
this prevents a symbolic endpoint from becoming an extra integration variable.

Supported specialist imports are legitimate public boundaries. In particular,
`intrinsic_parametric_cover` comes from `semialg.parametric_geometry`, not the
Semialg root. The adapter consumes verified covers, chart mappings, bounds,
Jacobian and Gram matrices, and induced metric factors. It substitutes the
integrand into each chart and feeds chart integrals into the native planner.
Do not apply the metric factor twice.

`reduce_domain()` consumes Semialg's structural region reduction rather than
asking Semialg to own symbolic integration. An absent reduction (`None`) means
that this geometry route did not certify a reduction; it does not mean an empty
region or a divergent integral. Contract violations raise `SemialgContractError`.

For intrinsic/top integration, lower-dimensional strata do not contribute to a
higher-dimensional Hausdorff measure. Zero-dimensional integration uses counting
measure. Singular target-dimensional strata require certification; an unresolved
stratum must not be silently discarded. A verified chart decomposition is geometry
evidence, not a universal proof of convergence of every possible integrand.

Within caller-supplied parameter cells, MultipleIntegrate also recognizes
independent axis-wise linear/quadratic inequalities as integration ranges. This
limited native shortcut requires proved signs and root ordering; it does not
replace Semialg CAD, discover a parameter decomposition, or certify general
symbolic intrinsic geometry. Unsupported parameter geometry stays unknown.
Parameter-dependent native boxes avoid moment delegation when Semialg cannot
resolve their parameters. An auxiliary dimension query failure retains unknown
diagnostic dimension rather than failing an otherwise valid integral.

See [supported regions](../supported-regions.md), [result semantics](../certification.md),
and [parameter cells](../parameterized-integration.md).

Regression coverage: `tests/test_geometry_reference_corpus.py`,
`tests/test_intrinsic_semialgebraic.py`, `tests/test_architecture_contract.py`.


## Chart certificates

`semialg.chart_certificates` owns reusable chart inverses, differential rank,
metric identities, source coverage, overlap and whole-parameter-cell null sets.
Its frozen records are cacheable and replayable. MultipleIntegrate's adapter
maps integration policy limits to `GeometryBudget`, and converts an unproved
prerequisite into `UnresolvedIntegration`. MultipleIntegrate owns transformed
densities, angular moment tables, convergence evidence and integration outcomes.
Vector maps use Semialg's full map Gram determinant and fiber Gram metric.
Semialg supplies bounded parameter-only rank, pole and boundary-event candidates;
MultipleIntegrate validates caller partitions, specializes equality fibers and
evaluates a separate source integral on proved rank-loss or collapsed-chart
fibers. Events do not certify a generic formula on an equality fiber.
A geometry certificate is never an integrability certificate. The shared linear
metric certificate is used by Semialg affine transformations and intrinsic affine
charts, and by MultipleIntegrate 3D boundary-facet reduction.
