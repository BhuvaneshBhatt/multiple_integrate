# How the planner chooses an integration method

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

The planner prefers the cheapest exact argument that is justified by the integrand, domain, and
assumptions. It does not run every method and compare answers.

## Dispatch responsibilities

Explicit prescriptions dispatch first: Cauchy, geometric, hypersurface and Hadamard semantics have separate evaluators and retain the requested cutoff. Ordinary evaluation normalizes ranges and assumptions, recognizes immutable geometry through a bounded cache, and checks the result cache.

Constant volumes and proved polynomial-function moments, pole-free rational formulas, and narrow oscillatory rules precede expensive singularity queries. Structured geometry then supplies Cartesian, simplex, radial, symmetry and coordinate-change formulas. Product separability and inactive finite dimensions preserve whole-domain prerequisites. Coordinate-sum closed forms keep priority over multi-index series.

Finite-box polynomial phases can produce an exact uniformly convergent series. Remaining composition strategies use linear pushforwards, positive-definite quadratic maps, additive and monotone changes, then selected layer-cake reductions. General iterated SymPy integration follows the proved specialized routes. Inconclusive backend output is unknown.

Semialgebraic geometry is owned by the Semialg adapter. Native axis-wise and graph reductions can precede certified decomposition. `integrate_cells` builds an exact supported transition partition or validates caller-supplied cells; each child retains its measure, coverage and convergence claims. A partition alone does not prove every child integral.

Certified infinite sums retain their summands during outer algebra simplification. Rewriting a hypergeometric coefficient into a slower numerical special function does not improve the integration strategy, and symbolic summation is not launched merely to simplify outer factors.

## Why a strategy may be skipped

A method can be inapplicable because orientation cannot be proved, assumptions are insufficient,
a singularity invalidates a symmetry argument, a transformation leaves an unevaluated integral,
or a configured complexity budget predicts expensive exact work. Use `integrate_result()`
to inspect these decisions.

For exact mathematical identities and hypotheses, see [Theory](theory.md). For a family-by-family
support summary, see the [Capability matrix](capabilities.md).

## Geometry extension routing

Certified polygonal domains with polynomial or affine-exponential densities, fixed-normal symbolic convex polygons on proved cells, and rational polynomial convex 3D polytopes can reduce to exact boundary flux before geometric decomposition. See [Polygon and polytope boundary reduction](boundary-reduction.md). Native symbolic boxes and affine cell maps precede globally certified polynomial graph strips. Cusp blowups check vertex and contact thresholds before angular symmetry. Finite mixed radial profiles can return exact convergent series. Shared product cutoffs retain source geometry and scale through affine or polynomial graph maps. Unknown signs or failed budgets decline; the caller can provide additional cells. These routes do not imply automatic CAD, general resolution of singularities, arbitrary polygon cutoffs or a global deadline for all upstream symbolic work.
