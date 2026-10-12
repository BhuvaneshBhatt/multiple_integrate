# Method references

This guide maps every implemented mathematical strategy family to its foundations
and source modules. The references explain identities and theorems; they do not
certify a particular result returned by the software. Applicability, branches,
parameter conditions, coverage and the requested regulator still need the proofs
recorded by MultipleIntegrate. See [Theory](theory.md), [Certification](certification.md)
and the individual family guides for those contracts.

A row marked as a package-specific derivation uses the cited standard results,
but does not claim that the cited author published the exact recognizer or chart.
Infrastructure such as caching, budgets, result wrappers and dispatch has no
separate mathematical method attribution.

## Methods and implementation map

| Method | Implementation modules (`multiple_integrate.*`) | Foundations and scope |
| --- | --- | --- |
| Iterated integration, product separation and inactive coordinates | `core`, `separable_strategies` | Fubini/Tonelli and ordinary oriented integration [RA], [CALC]; absolute integrability or nonnegative density is required. |
| Finite polynomial box moments | `fast_rules`, `analytic_region_formulas`, `region_types` | Power antiderivatives, linearity and Fubini [CALC], [RA]. The sparse expansion budget is an implementation choice. |
| Polynomial–exponential and polynomial–trigonometric recurrences | `fast_rules` | Repeated integration by parts [CALC]. Zero frequency is evaluated as a polynomial moment, separately from division by frequency. |
| Polynomial Gaussian recurrences and shifted/truncated moments | `fast_rules`, `quadratic_gaussian` | Completing the square, integration by parts, erf [ERF], and incomplete-gamma recurrences [IGR]. Boundary limits need positive decay. |
| Full-space quadratic Gaussian and halfspace reductions | `quadratic_gaussian` | Linear change of variables [RA], one-dimensional Gaussian evaluation [ERF], and completing the square. Positive definiteness is proved first. |
| Beta, Mellin and rational power families | `selected_families`, `convergence` | Euler beta integrals [BETA], with substitutions and endpoint/tail exponent tests [CALC]. |
| Laplace, stretched-exponential, endpoint and Laurent moments | `selected_families`, `fast_rules` | Gamma/incomplete-gamma integral representations [IG]; substitution gives the stretched exponent, and finite Laurent sums use linearity. Negative powers require the recorded cutoff prerequisites. |
| Pole-free linear/quadratic and selected quartic rational formulas | `fast_rules`, `rational_cutoff` | Polynomial division, partial fractions, completion of squares, log/atan antiderivatives [CALC]. Quartic reductions are elementary substitutions; no general rational-integrator claim is made. |
| Reciprocal-phase and periodic trigonometric tails | `fast_rules` | Substitution into Si/Ci [SICI], integration by parts/Dirichlet convergence [CALC]. Positive periodic absolute mean proves harmonic-tail divergence; this last comparison is a package derivation. |
| Simplex and affine-simplex moments | `region_strategies`, `analytic_region_formulas` | Dirichlet simplex identity [DIR] plus affine change of variables [RA]. |
| Balls, shells, ellipsoids and spherical polynomial moments | `sphere_integrals`, `quadratic_radial`, `analytic_region_formulas` | Polar change of variables [RA], angular beta/Dirichlet integrals [BETA], [DIR], and symmetry. Tensor pairing formulas can be derived from the same angular monomial moments; they are not an invocation of a probabilistic independence assumption. |
| Mixed radial profiles and half-integer moments | `quadratic_radial` | Radial substitution and beta/gamma identities [BETA], [IG]; confluent-hypergeometric integral representations [U] and Mellin–Barnes notation [G] explain special-function forms. Formula-specific prerequisites remain in the radial guide. |
| Weighted even-power and superellipse moments | `anisotropic`, `layer_cake_strategies` | Positive-orthant substitutions into Dirichlet integrals [DIR], summed over valid orthants. Weighted exponents determine the radial convergence threshold. |
| Layer-cake and level-set integration | `layer_cake_strategies`, `pushforward_strategies` | Tonelli applied to nonnegative superlevel indicators [RA]; smooth level-set densities use the coarea formula [GMT]. |
| Linear pushforwards, monotone and piecewise changes | `pushforward_strategies`, `transforms` | Change of variables [RA], [CALC], split into proved monotone pieces. Inverse branches, image bounds and Jacobians must all be tracked. Linear combinations of box coordinates also use piecewise polynomial pushforward densities [BOX]. |
| Intrinsic point, curve and surface measure | `semialg_geometry`, `adapters.semialg`, `radial_parametrization` | Area/coarea formulas and Hausdorff measure [GMT]. Parametric metric factors use the Gram determinant; ambient and intrinsic measure differ. |
| Graph strips, affine cells and quadratic parameter geometry | `cell_graphs`, `symbolic_geometry`, `affine_cell_geometry`, `quadratic_cell_geometry` | Fubini, polynomial antiderivatives and proved changes of variables [RA], [CALC]. These are selected native reductions, not a general topology algorithm. |
| Clipped polygons, parallel quadratic clips and angular sectors | `polygon_moments`, `clipped_quadratic`, `quadratic_sectors` | Piecewise domain integration and affine/polar substitution [RA], beta angular moments [BETA]. Clipping recognition and the supported sector formulas are package-specific reductions. |
| Bernstein interval positivity certificates | `cell_proofs` | Nonnegative Bernstein basis and coefficient convex-hull bounds [BERN]. Strict coefficient bounds certify a whole finite interval; a failed sufficient certificate does not prove negativity. |
| Automatic transition cells and caller-supplied cells | `transition_cells`, `parameter_cells`, `cell_proofs` | Real polynomial sign decomposition, coefficient/discriminant/resultant events and equality fibers are motivated by real algebraic decomposition [CAD]. The bounded event partition is not a full CAD implementation or a topology certificate. |
| Compact continuity, products, sums and comparison proofs | `convergence_evidence`, `convergence_comparison` | Comparison, dominated convergence, Fubini and additivity [RA]; endpoint and radial power tests [CALC]. A sufficient existence proof can accompany an unknown value. |
| Endpoint vanishing, cusp horns and logarithmic tangent orders | `cusp_horns`, `contact_charts` | Taylor order and explicit weighted substitutions [CALC], [RA], followed by beta/power/log endpoint tests [BETA]. The proved charts are selected package derivations, not general resolution of singularities. |
| Nonseparable full-space shear profiles | `nonseparable_improper` | Global triangular polynomial change of variables [RA] with determinant one, followed by weighted Dirichlet moments [DIR]. The recognition algorithm and monomial-wise origin/tail proof are package-specific; see the improper guide. |
| Uniform entire-box series and bounded Taylor evaluation | `entire_box_series`, `series_evaluation` | Taylor expansion/remainder [CALC] plus dominated termwise integration [RA]. The rational majorant and operation budgets are package-specific certificates, not numerical convergence guesses. |
| Reflection cancellation | `core`, `quadratic_gaussian` | Measure-preserving substitution and oddness [RA]. Ordinary existence is required; a singular prescription additionally needs cutoff symmetry. |
| Cauchy, geometric and hypersurface principal values | `cauchy_principal_value`, `geometric_principal_value`, `hypersurface_principal_value`, `hypersurface_atlas` | Symmetric limiting prescriptions [CALC], [FP]; local implicit-coordinate charts and coarea [GMT]. Distinct neighborhoods define distinct prescriptions. |
| Hadamard finite parts, rational finite parts at infinity | `hadamard`, `regularization`, `rational_cutoff`, `adapters.asymptotic` | Finite-part regularization and constant-term extraction [FP]; exact rational primitives [CALC]. The recorded regulator, logarithms and scale affect the answer. |
| Shared product cutoffs, affine/graph intersections and tangent regulators | `singular_geometry`, `singular_maps`, `triangular_intersections`, `tangent_cutoff`, `regular_density` | Proved change of variables [RA], regularization [FP] and dominated limits for regular densities [RA]. Shared regulator transformations are derived explicitly in the package; iterated independent cutoffs cannot replace them. |
| Fallback symbolic integration | `core`, `singularities` | SymPy documented integrators [SYMPY]. Risch, heuristic Risch and Meijer-G methods are backend capabilities, not reimplementations in MultipleIntegrate. |
| Boolean indicators and certified region restrictions | `region_formulas`, `regions` | Integrating measurable indicators, additivity on certified pieces and Fubini [RA]. Geometry certification is delegated through Semialg; overlapping pieces cannot be summed without proof. |
| Affine exponential boundary flux | `boundary_reduction` | Divergence theorem [POLY] and finite polynomial–exponential antiderivative recurrences [CALC]; nonzero rates and constant edge phases are proved separately. |
| Polynomial boundary flux on polygons and 3D polytopes | `boundary_reduction` | Euler’s homogeneous identity and the divergence theorem [POLY]; certified oriented polygon edges and recursive rational facets reduce each monomial to endpoint power moments [CALC]. |
| Numerical consistency checks | `numerical_verification` | mpmath quadrature documentation [MP]. Precision escalation supplies a consistency check, not a proof of convergence or domain coverage. |

| Verified affine, radial and certified-chart coarea reductions | `coarea` | Weighted coarea [GMT] §§3.2.11–3.2.12; supported chart inverses, rank, metric identity, null exclusions, coverage and disjoint interiors are certified by Semialg. Polynomial angular moments use spherical gamma identities [IG] with absolute majorants. Evaluated values require absolute convergence or Tonelli. |

## Sources

- **BERN** — Rida T. Farouki, “The Bernstein polynomial basis: A centennial retrospective,” *Computer Aided Geometric Design* 29(6), 2012, pp. 379–419. [Paper and DOI](https://doi.org/10.1016/j.cagd.2012.03.001).
- **BOX** — Carl de Boor, Klaus Höllig and Sherman Riemenschneider, *Box Splines*, Springer, 1993. [Publisher record and DOI](https://doi.org/10.1007/978-1-4757-2244-4). The reference explains box pushforward densities; MultipleIntegrate implements selected scalar reductions.
- **POLY** — Jean B. Lasserre, “Integration on a Convex Polytope,” *Proceedings of the American Mathematical Society* 126(8), 1998, pp. 2433–2441. [Paper and DOI](https://doi.org/10.1090/S0002-9939-98-04454-2).
- **RA** — Gerald B. Folland, *Real Analysis: Modern Techniques and Their Applications*, second edition, Wiley, 1999. Chapter 2 covers Lebesgue integration, convergence theorems, product integration and changes of variables. [Publisher record](https://www.wiley-vch.de/en?isbn=9780471317166&option=com_eshop&view=product).
- **CALC** — NIST Digital Library of Mathematical Functions (DLMF), [§1.4, Calculus of One Variable](https://dlmf.nist.gov/1.4). Integration, integration by parts, improper integrals, Taylor expansions and changes of variable provide the elementary steps used above.
- **BETA** — DLMF [§5.12, Beta Function](https://dlmf.nist.gov/5.12), especially equations 5.12.1–5.12.3 for interval, angular and half-line forms.
- **DIR** — DLMF [§5.14, Multidimensional Integrals](https://dlmf.nist.gov/5.14), equations 5.14.1–5.14.2 for simplex Dirichlet integrals.
- **ERF** — DLMF [§7.2, Definitions](https://dlmf.nist.gov/7.2), error-function and complementary-error-function integrals. Gaussian polynomial recurrences also use integration by parts; the source does not describe the package's recurrence code.
- **IG** — DLMF [§8.6, Integral Representations](https://dlmf.nist.gov/8.6), for incomplete-gamma moments.
- **IGR** — DLMF [§8.8, Recurrence Relations and Derivatives](https://dlmf.nist.gov/8.8), for incomplete-gamma recurrences.
- **SICI** — DLMF [§6.2, Definitions and Interrelations](https://dlmf.nist.gov/6.2), for sine, cosine and exponential integral conventions and endpoint forms.
- **U** — DLMF [§13.4, Integral Representations](https://dlmf.nist.gov/13.4), especially equation 13.4.4 for the confluent hypergeometric U integral.
- **G** — DLMF [§16.17, Definition](https://dlmf.nist.gov/16.17), for the Meijer G Mellin–Barnes representation. This identifies notation and analytic conventions, not a universal convergence proof.
- **GMT** — Herbert Federer, *Geometric Measure Theory*, Springer, originally 1969; Classics in Mathematics reprint 1996. Area and coarea formulas, especially §3.2.11–3.2.12 (pp. 248–249), in the rectifiability chapter are the foundations for intrinsic and level-set reductions. [Publisher record and DOI](https://doi.org/10.1007/978-3-642-62010-2).
- **CAD** — George E. Collins, “Quantifier elimination for real closed fields by cylindrical algebraic decomposition,” *Automata Theory and Formal Languages*, LNCS 33, 1975, pp. 134–183. [Publisher record and DOI](https://doi.org/10.1007/3-540-07407-4_17). General decomposition belongs to the geometry backend; MultipleIntegrate's selected event cells are narrower.
- **FP** — Ricardo Estrada and Ram P. Kanwal, “Regularization, pseudofunction, and Hadamard finite part,” *Journal of Mathematical Analysis and Applications* 141(1), 1989, pp. 195–207. [Paper and DOI](https://doi.org/10.1016/0022-247X(89)90216-3). This is background for finite-part regularization, not an equivalence theorem for all geometric or hypersurface regulators.
- **SYMPY** — [SymPy integration documentation](https://docs.sympy.org/latest/modules/integrals/integrals.html), including algorithm-specific references. Backend return values do not by themselves certify the multiple-integral convergence hypotheses.
- **MP** — [mpmath quadrature documentation](https://mpmath.readthedocs.io/en/latest/calculus/integration.html), for the numerical integration algorithms used as independent checks.

## Maintaining attribution

When adding a strategy, extend this table with its module, underlying identity,
reference and applicability limits. A new derivation should point to its proof
in the family guide instead of assigning credit to a vaguely related paper.
Update `docs/method-reference-coverage.json` as well: the documentation tests
require every production module in the ownership manifest to have either a
method-reference entry or an explicit infrastructure classification.

[RA]: https://www.wiley-vch.de/en?isbn=9780471317166&option=com_eshop&view=product
[CALC]: https://dlmf.nist.gov/1.4
[BETA]: https://dlmf.nist.gov/5.12
[DIR]: https://dlmf.nist.gov/5.14
[ERF]: https://dlmf.nist.gov/7.2
[IG]: https://dlmf.nist.gov/8.6
[IGR]: https://dlmf.nist.gov/8.8
[SICI]: https://dlmf.nist.gov/6.2
[U]: https://dlmf.nist.gov/13.4
[G]: https://dlmf.nist.gov/16.17
[GMT]: https://doi.org/10.1007/978-3-642-62010-2
[CAD]: https://doi.org/10.1007/3-540-07407-4_17
[FP]: https://doi.org/10.1016/0022-247X(89)90216-3
[SYMPY]: https://docs.sympy.org/latest/modules/integrals/integrals.html
[MP]: https://mpmath.readthedocs.io/en/latest/calculus/integration.html
[BERN]: https://doi.org/10.1016/j.cagd.2012.03.001
[BOX]: https://doi.org/10.1007/978-1-4757-2244-4
[POLY]: https://doi.org/10.1090/S0002-9939-98-04454-2
