# Supported regions and measures

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

Exact evaluation is conditional on certified geometry and an integrable expression;
this matrix describes routes, not a guarantee that every expression is solvable.

| Region | Parameters | Intrinsic measure | Typical route | Limits |
|---|---|---|---|---|
| `Box` | symbolic endpoints; check assumptions | degenerate cases need dimension checks | iterated / moments | orientation and singularities |
| `Point` | symbolic coordinates | counting measure, dimension 0 | direct substitution | ambient positive-dimensional measure is zero |
| `Ball`, `SphericalShell` | radii under valid conditions | full-dimensional volume | radial / moments / Semialg reduction | branch and convergence restrictions |
| `Sphere` | valid radius | surface measure | intrinsic cover / spherical moments | not ball volume |
| `Simplex` | valid vertices | embedded length/area/volume | affine chart / moments | rank drops change dimension |
| Axis-wise linear/quadratic formulas | caller-supplied cells with proved root ordering | points and coordinate-aligned lower-dimensional fibers | native symbolic interval boxes | coupled or uncertain geometry declines |
| Polynomial graph strips | caller-supplied cells with proved envelopes or complete ordered intersection roots | stable dimension must be proved; partitioned route is ambient | Bernstein envelope / exact root subdivision / iterated moments | unproved root order and weak slopes decline |
| Parallel affine ellipsoid slabs | proved SPD quadratic, parallel planes, ordered bounds and finite polynomial density | ambient; full-dimensional intrinsic | conditional covariance / incomplete beta moments | nonparallel clips and intrinsic collapsed slices decline |
| Planar ellipsoid sectors | exact algebraic SPD matrix and center-passing affine clip directions | ambient; full-dimensional intrinsic | whitening / exact angular cells / polynomial moments | offset nonparallel clips and intrinsic collapsed cones decline |
| Tangent graph horns | y*(y-c*x**m), c>0, x>0 and matched horn bounds; proved polynomial shears and translations | prescribed shared level-set cutoff | weighted blowup / rational and dilogarithmic angular moments | fractional radial orders, other poles and unproved implicit charts decline |
| Polygonal Boolean formulas | rational boundaries; convex fixed rational normals allow proved symbolic offsets | full-dimensional only in this route | polynomial and affine-exponential boundary flux | symbolic nonconvex and changing normals decline |
| Convex 3D polytopes | rational halfspaces | full-dimensional only in this route | recursive polynomial facet moments | nonconvex/symbolic 3D and exponential 3D decline |
| Clipped downward-closed polygons | proved finite coefficients and positive cutoff window | ambient prescribed cutoff | exact Laurent moments / shared product cutoff | holes and lower sloping cuts decline |
| Cusp horns | 0<k<=1, strict vertex and contact thresholds | ambient measure | exact blowup / vanishing order | sign-changing divergence and general cusps remain unresolved |
| Polynomial Boolean formulas | specialized or certified parameter cells | supported verified strata | Semialg CAD / intrinsic covers | complexity and certification can stop evaluation |
| Disconnected or mixed-dimensional formulas | cell-specific | select a dimension explicitly | disjoint geometry pieces | no double counting; lower strata have zero higher measure |
| Algebraic boundaries | algebraic coefficients | selected regular charts | certified bounds / induced metric | singular or unresolved geometry may be unsupported |
| Verified scalar/vector coarea domains | supplied parameter cells; chart inverses, rank, coverage and null sets certified by Semialg | ambient dimension at least two | affine/radial reduction, polynomial/Fourier and bounded rational angular moments, selected algebraic and canonical polar charts | geometry alone does not prove convergence; arbitrary critical-fiber stratification declines |
| Nonsemialgebraic formulas | no general promise | no general promise | unsupported geometry route | use explicit ranges if valid |

With `measure_dimension="ambient"` (default), a curve in R² has zero area and a
surface in R³ has zero volume. Use `"intrinsic"`, `"top"`, or an integer dimension
for the desired Hausdorff measure. A point has intrinsic mass one. A mixed union
needs an explicit convention: dimension 1 ignores isolated points, while counting
measure on a positive-dimensional set is not a finite sum of point evaluations.

Regression coverage: `tests/test_geometry_reference_corpus.py`,
`tests/test_lower_dimensional_contracts.py`.

Regression coverage: `tests/test_symbolic_cell_geometry.py`, `tests/test_weighted_quadratic_radial.py`.

Mixed radial density support and exact result representations are described in [radial series](geometry/radial-series.md). Complete geometry coverage does not certify every singular density.

## Parameter-cell geometry extensions

Bounded-degree one-parameter polynomial events and multiple-parameter polynomial sign conditions can include selected coupled linear forms and rational parameter coefficients. Generic affine box images require a proved inverse and Jacobian; graph reductions require proved envelopes. Equality fibers retain their own dimension/measure semantics. Coefficient-pole fibers remain unknown instead of being treated as empty. Single SPD quadratic inequalities support exact polynomial moments on proved cells. Sign conditions are not a general nonlinear CAD, and arbitrary coupled nonlinear geometry remains unresolved.

Regression coverage: `tests/test_parameter_geometry_extensions.py`, `tests/test_parameter_specialization_safety.py`.

Further supported cases and prerequisites are described in [the extension guide](nonlinear-moments-and-singularities.md). Regression coverage: `tests/test_truncated_nonlinear_horns.py`.

Selected parallel affine slabs of SPD ellipsoids have exact polynomial moments; intrinsic collapsed slices are declined by that slab rule. Tangent graph horns support the selected regulator-preserving rational cutoff family. See [Laurent moments, clips and tangencies](laurent-clips-and-tangencies.md) and `tests/test_laurent_clips_and_tangencies.py`.

Planar ellipsoid sectors with exact algebraic nonparallel halfplanes through the center
have polynomial moment coverage. Finite polynomial graph strips may be partitioned
at completely known, proved ordered graph-intersection roots inside caller cells.
See [sector and contact prerequisites](sectors-and-contact-charts.md).

Full real space supports polynomial shear images of weighted even-power profiles
with proved origin/tail thresholds. This does not extend the same chart to untransformed
finite or halfspace domains. Regression: `tests/test_nonseparable_improper.py`.

Selected polynomial or affine-exponential polygons, including rational nonconvex loops and holes, fixed-normal symbolic convex polygons on supplied cells, and rational polynomial convex 3D polytopes support [boundary reduction](boundary-reduction.md), with complete facet and compactness proofs.
