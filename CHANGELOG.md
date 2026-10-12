# Changelog

## Version 2.2.0

Version 2.2 adds geometry-driven symbolic integration methods, including generalized co-area reductions, polygonal and polyhedral boundary integration, and polynomial coordinate transformations, alongwith parameter-dependent regions and expanded handling of improper integrals.

### New and Expanded Integration Capabilities

#### More Geometric Regions

- Added integration of polynomial functions over convex and nonconvex polygons, including polygons with holes and disconnected regions, and over some three-dimensional convex polyhedra.
- Added boundary-based integration methods that reduce integrals over polygons and polyhedra to integrals over their edges or faces.
- Expanded support for regions defined by polynomial inequalities, including some curved boundaries, cusps, and clipped ellipsoids.

#### More Powerful Coordinate Transformations

- Added transformations that simplify certain nonseparable integrals into products of standard one-dimensional integrals.
- Added support for some polynomial coordinate transformations including nonlinear shear transformations.
- Extended coarea integration, which evaluates multidimensional integrals by integrating over the level sets of a function.
- Added coarea reductions for scalar and vector-valued functions, including some affine, radial, rational, algebraic, and polar transformations.
- Improved handling of coordinate singularities, exceptional level sets, and variable-changes that are valid except on measure-zero sets.

For example, MultipleIntegrate can now recognize some integrals involving expressions such as `x**4 + (y - x**2)**2`, where a nonlinear coordinate transformation makes exact integration possible.

#### More Integrand Families

- Expanded support for polynomial-weighted Gaussian, exponential, rational, and inverse-power integrals.
- Added new radial and angular integration formulas, including polynomial angular factors, Fourier terms, and some rational trigonometric expressions.
- Added weighted moments involving higher even powers such as `x**4 + y**6`
- Improved integration of certain functions with endpoint singularities, logarithmic behavior, or integrable point singularities.
- Expanded exact series methods (no truncation) for some non-separable integrals that do not have convenient elementary closed forms.

### Better Parameters Handling

MultipleIntegrate can now handle a wider variety of integrals whose values or integration regions depend on symbolic parameters.

- Added auto-analysis of certain parameter-dependent regions, including cases where a region changes shape, becomes empty, or changes dimension.
- Added `integrate_cells()` function for evaluating integrals separately on different parameter ranges.
- Improved handling of exceptional parameter values where an integration formula changes or becomes invalid.
- Extended exact integration over parameter-dependent polygons, graph regions, and quadratic regions.
- Improved reporting of assumptions/conditions under which a symbolic result is valid.

These make it possible to obtain piecewise results for more parametrized integrals instead of you having to evaluate each parameter case manually.

### Improved Convergence and Singular-Integral Handling

MultipleIntegrate now has stronger mathematical checks for improper and singular integrals.

- Expanded convergence tests for integrals over infinite domains and near singularities.
- Improved recognition of integrable singularities that may initially appear divergent.
- Added stronger checks for absolute vs conditional convergence, particularly when symmetry or cancellation is involved.

### More Informative Results

The expanded result API makes it easier to understand how and under which conditions MultipleIntegrate calculated the result.

- `integrate()` has the usual symbolic integration interface and gives the mathematical result without extra information.
- `integrate_result()` gives detailed results, including convergence info, assumptions, and diagnostic info.
- `integrate_cells()` handles some parameter-dependent integrals.
- `regularized_result()` gives results and supporting information for explicitly regularized integrals.
- Added `Boole()` for constructing indicator functions used in region-restricted integration.

Results can now distinguish more clearly between successful exact evaluation, conditional results, proved divergence, unsupported methods, and unresolved problems.

### Performance Improvements

- Improved strategy selection to avoid unnecessarily expensive symbolic calculations, added computational budgets to limit excessive symbolic expansion.
- Added early-rejection of unsupported expressions before attempting costly geometric or algebraic analysis.

### Testing & Documentation

- Expanded mathematical tests: integration formulas, coordinate transformations, singularities, convergence, parameter-dependent regions.
- Expanded regression tests: invalid assumptions, exceptional cases, numerical consistency, caching, computational budgets.
- Expanded performance regression tests for potentially expensive integration families
- Added executable examples representing all 34 documented method families.

### Requirements

- **Python version:** 3.11+.
- **Required dependencies:** SymPy 1.12+, mpmath 1.3+, semialg 1.3.2+, and funcprops 0.3.0+.
- **Optional:** asymptotic 0.2.1+ for additional supported regularization calculations.

### Current Limitations

Arbitrary nonlinear geometric decompositions, general singularity resolution and general multidimensional oscillatory improper integration are outside the current scope. See the [capability guide](docs/capabilities.md) and [limitations](docs/limitations.md) for details.

## Version 2.1.0

Version 2.1 expands support for geometric integration, improves handling of assumptions and improper integrals, and includes several bugfixes.

### Improved Geometric Integration

- Added recognition and exact integration of translated disks, balls, and axis-aligned ellipsoids.
- Extended polynomial moments, radial integration, symmetry detection, and coordinate transformations to account for the centers of translated regions.
- Improved integration over annulii, spherical shells, ellipsoids, and unions of supported regions.
- Added stricter validation of region dimensions, radii, coordinate centers, and integration bounds.
- Improved detection of overlapping regions to prevent double-counting when integrating over unions.
- Strengthened recognition of nested integration bounds to avoid incorrectly identifying the integration region.

### Bugfixes

- Fixed incorrect treatment of constant integrands over regions with dependent bounds.
- Prevented symmetry arguments from incorrectly assigning zero to divergent singular integrals.
- Made sure that an integrand is genuinely radial before applying radial integration formulas.

### Better Assumptions Handling

- Expanded the `assumptions` argument to accept SymPy relational conditions and predicates, such as `a > 0` and `Q.positive(a)`.
- Added consistency checks that reject contradictory assumptions.
- Improved integration of parameter-dependent Gaussian, exponential, power-law, and simplex integrals.
- Improved convergence checks for integrals where validity depends on parameters.

### Performance & Reliability

- Improved selection and ordering of specialized integration methods.
- Added bounded caching of integration results and intermediate expression decompositions.
- Expanded regression tests for correctness, region classification, assumptions, and performance.

### Documentation

- Updated README and example notebook.
- Added a new quick-start guide
- Reorganized API reference.
- Added a detailed guide to integration ranges, region classification, and geometric integration methods.
- Expanded examples demonstrating supported integration regions.
- Clarified the required ordering of nested integration bounds, which follows the same inner-first convention as SymPy.
