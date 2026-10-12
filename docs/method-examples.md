# Executable method catalogue

Every mathematical family in the [method reference guide](method-references.md) has a checked representative in `examples/method_catalogue.py`. Run `PYTHONPATH=src python examples/method_catalogue.py` from the source tree. The examples also run individually in the full test suite.

This is family-level coverage: a row may group several formulas or backend algorithms. It does not promise an example for every recognizer branch, parameter stratum or upstream SymPy algorithm. Automatic dispatch can choose a faster applicable strategy; theorem/verification examples invoke specialist APIs explicitly. The catalogue checks values or proof claims, and the manifest test requires every reference row to have an entry.

| Method family | Example function |
|---|---|
| Iterated integration, product separation and inactive coordinates | `iterated` |
| Finite polynomial box moments | `box_moments` |
| Polynomial–exponential and polynomial–trigonometric recurrences | `exp_trig` |
| Polynomial Gaussian recurrences and shifted/truncated moments | `gaussian_recurrence` |
| Full-space quadratic Gaussian and halfspace reductions | `quadratic_gaussian` |
| Beta, Mellin and rational power families | `beta_mellin` |
| Laplace, stretched-exponential, endpoint and Laurent moments | `laplace` |
| Pole-free linear/quadratic and selected quartic rational formulas | `rational` |
| Reciprocal-phase and periodic trigonometric tails | `oscillatory` |
| Simplex and affine-simplex moments | `simplex` |
| Balls, shells, ellipsoids and spherical polynomial moments | `spherical_moments` |
| Mixed radial profiles and half-integer moments | `mixed_radial` |
| Weighted even-power and superellipse moments | `even_power` |
| Layer-cake and level-set integration | `layer_cake` |
| Linear pushforwards, monotone and piecewise changes | `pushforward` |
| Intrinsic point, curve and surface measure | `intrinsic` |
| Graph strips, affine cells and quadratic parameter geometry | `graph` |
| Clipped polygons, parallel quadratic clips and angular sectors | `quadratic_clip` |
| Bernstein interval positivity certificates | `bernstein` |
| Automatic transition cells and caller-supplied cells | `parameter_cells` |
| Compact continuity, products, sums and comparison proofs | `compact_proof` |
| Endpoint vanishing, cusp horns and logarithmic tangent orders | `cusp` |
| Nonseparable full-space shear profiles | `shear` |
| Uniform entire-box series and bounded Taylor evaluation | `entire_series` |
| Reflection cancellation | `reflection` |
| Cauchy, geometric and hypersurface principal values | `principal_value` |
| Hadamard finite parts, rational finite parts at infinity | `finite_part` |
| Shared product cutoffs, affine/graph intersections and tangent regulators | `shared_cutoff` |
| Fallback symbolic integration | `symbolic_fallback` |
| Boolean indicators and certified region restrictions | `indicators` |
| Affine exponential boundary flux | `exponential_boundary` |
| Polynomial boundary flux on polygons and 3D polytopes | `polynomial_boundary` |
| Numerical consistency checks | `numerical_check` |
| Verified affine, radial and certified-chart coarea reductions | `coarea` |

The coarea example executes affine box, radial full-space, supplied polynomial/polar charts, vector maps, Fourier and bounded reciprocal angular moments, and caller-supplied exceptional parameter cells. The indicator example uses `Boole` and ordered `Piecewise` branches. Existing README, casebook, guides and notebook offer longer walkthroughs; this catalogue supplies an exhaustive index rather than duplicating them.

Regression: `tests/test_method_catalogue.py`.
