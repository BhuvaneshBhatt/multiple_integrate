# Capability matrix

This page answers **what kind of problem should I expect MultipleIntegrate to solve?** It is a
support matrix, not a list of internal planner functions.

| Problem family | Exact native path | Conditional result | Delegated path | Main limitation |
| --- | --- | --- | --- | --- |
| finite boxes / product ranges | yes | parameter orientation when needed | SymPy fallback | difficult nonseparable antiderivatives |
| simplex / affine simplex moments | yes | exponent conditions | — | unsupported non-Dirichlet structure |
| disks, balls, annuli, shells, ellipsoids | selected moments/radial formulas | parameter conditions when certifiable | SymPy after transforms | general transcendental angular dependence |
| Gaussian full-space integrals | selected exact formulas | positive-definiteness/decay conditions | SymPy fallback | general symbolic matrices/parameters |
| graph / iterated regions | selected recognition | orientation/branch conditions | SymPy iterated integration | difficult dependent bounds |
| semialgebraic Boolean domains | native cheap recognizers first | — | `semialg` CAD/reduction | variable/degree/Boolean complexity |
| lower-dimensional semialgebraic sets | — | — | `semialg` intrinsic charts | singular or uncertified strata |
| 1-D Cauchy PV | yes | assumption-sensitive pole location | SymPy PV fallback | nonisolated/branch singularities |
| isolated-point geometric PV | spheres/ellipsoids | geometry certification | `semialg` for general domains | uncertified local neighborhoods |
| regular polynomial hypersurface PV | graph and certified atlas paths | — | `semialg` geometry | singular hypersurfaces and hard algebraic branches |
| Hadamard finite part | rational 1-D and supported hypersurface cutoffs | — | `asymptotic` multiseries | nonrational infinity finite parts and richer interacting strata |
| intersecting/repeated coordinate planes | Laurent densities on asymmetric finite boxes and certified affine box images | certified widths/scales | exact shared product cutoff | general curved/singular intersections remain unsupported |

## Result meanings

- **Exact**: a supported strategy returned a closed symbolic value under the active assumptions.
- **Conditional**: the parameter space contains distinct convergence/value regimes.
- **Unevaluated**: a configured strategy budget declined expensive exact work.
- **Unsupported**: the requested prescription or strategy has no implemented route.
- **Unknown**: available theorem/backends did not establish a result.
- **Divergent**: divergence was established for the relevant parameter cell.
- **Principal value / finite part**: a nonordinary regularization prescription supplied the value.

A configured complexity guard is not a mathematical failure. Use `integrate_result()` to
see which strategy was declined and why, or raise the relevant budget only when the exact backend
cost is acceptable.

Polynomial-weighted Gaussian and inverse-quadratic profiles support full-space
positive definite quadratic forms, including cross terms and translated centers.
Weighted radial profiles also support balls, shells, and ellipsoids. See
[integration extensions](integration-extensions.md) for precise prerequisites.

## Geometry and density matrix

| Density | Supported geometry | Prerequisites / result |
|---|---|---|
| Polynomial | finite parameter-cell graph strips | globally proved envelopes; exact iterated expression |
| Cusp polynomial / defining-equation power | selected cusp horns including contact | exact vanishing; strict local thresholds; beta or proved divergence |
| Polynomial-weighted mixed Gaussian/shifted power | supported finite balls, shells, quadratic maps | positive shift and supported index; exact convergent series or special function |
| Integer Laurent moments | downward-closed clipped polygons or mapped graphs | shared product regulator; certified envelope and cutoff window; finite-part constants |

See [the casebook](geometry-casebook.md) for executable calls, negative prerequisites and evidence inspection.
