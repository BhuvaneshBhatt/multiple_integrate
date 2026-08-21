# Integration methods

MultipleIntegrate uses a layered planner. Methods are selected from the mathematical structure of the region and integrand rather than from a fixed numbered sequence.

## Region recognition

Nested bounds are classified into boxes, graph regions, simplices, affine simplices, disks, annuli, balls, spherical shells, ellipsoids, or a general iterated region. Recognized geometry enables exact formulas that are unavailable to a purely syntactic iterated integrator.

## Structured exact formulas

Product regions support inactive-dimension elimination and product separation. Simplex and affine-simplex regions support Dirichlet and polynomial-moment formulas. Disks, balls, annuli, shells, and ellipsoids support selected volume, moment, radial, and transformed integrals.

## Symmetry and separability

Reflection invariance can eliminate odd contributions after singularity safety checks. Symmetry is tested before coordinate transformations so a cheap exact cancellation is not replaced by a harder transformed integral. Product-separable integrands are reduced to independent one-dimensional integrals. Additively separable inner expressions can be treated through pushforward densities when their ranges are independent.

## Coordinate changes

Polar coordinates are used for disks and annuli, spherical coordinates for balls and shells, and affine normalization for affine simplices and ellipsoids. Translated disks, balls, and ellipsoids retain their centers in the coordinate map. A transformation is accepted only when the target ranges and Jacobian can be represented exactly, and a transformed result that still contains an unevaluated `Integral` is treated as an unsuccessful strategy so other methods can continue.

For explicit symbolic ranges, geometric formulas are used only when range orientation can be proved under the active assumptions. This prevents unsigned geometric volume formulas from replacing oriented iterated integrals.

## Composition and level-set methods

For an integrand expressible as `f(g(x))`, the internal composition analysis identifies the outer function and inner expression. Depending on the structure of `g`, the planner may use linear pushforward formulas, monotone substitutions, piecewise-monotone substitutions, polynomial level-set integration, or a general symbolic level-set density.

## Iterated fallback

When no specialized method is justified, MultipleIntegrate evaluates the supplied inner-first ranges with SymPy. Specialized methods are conservative optimizations: an uncertain structural match should fall back rather than change the mathematical domain or return an unsupported closed form.

## Boolean restrictions

Boolean `Or` conditions are converted to a `UnionRegion` only when the resulting pieces can be proved pairwise disjoint. If disjointness is uncertain or branches overlap, the union shortcut is declined and the ordinary symbolic path handles the Boolean expression without double counting.

## Bounded heuristic solving

Critical-point searches are heuristic accelerators rather than correctness requirements. Candidate critical points are used only when their interval membership can be proved. Symbolic solves in this path use a POSIX timer when available; in execution contexts without a safe interval timer, the heuristic is skipped rather than allowed to run without a bound. Existing process timers are preserved.
