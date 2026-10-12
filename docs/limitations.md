# Limitations and known issues

This page describes the supported mathematical contract and reproducible remaining issues. New bug reports belong in the repository issue tracker. Include the exact density, region or ranges, assumptions, measure, regularization prescription, dependency versions and a minimal expected result. A mathematically unsupported family is a capability limitation rather than evidence of a wrong answer.

## Capability limits

- Automatic partitions support bounded-degree one-parameter polynomial events and multiple-parameter polynomial sign conditions, with selected affine and coupled-quadratic degeneracy events. Sign conditions need not be connected or topology-stable. Children certify their geometry independently; polynomial moments on a single SPD ellipsoid are supported. Parallel affine ellipsoid slabs are supported with proved order and covariance moments; selected planar center-passing nonparallel clips are also supported. Offset nonparallel clips, arbitrary clipped nonlinear geometry and general nonlinear CAD remain outside these native recognizers.
- General nonseparable integration is not decidable by a universal symbolic procedure. Finite-box polynomial amplitudes times exp/sin/cos of polynomial phases have exact uniformly convergent series; selected beta, generalized Laplace and Mellin moments, together with radial, quadratic, graph and pushforward families, supply other reductions. Arbitrary densities, domains and improper nonseparable integrals may remain unknown.
- Singular product charts require proved global inverses and mapped Cartesian pieces. Polynomial graph normal crossings are supported in the triangular family. Selected tangent pairs y*(y-c*x**m) on proved one-sided horns have weighted blowup cutoffs with the original regulator retained. Other tangent intersections, arbitrary implicit singular varieties and incompatible regulators do not have a general finite-part procedure.
- A correct multi-index exact `Sum` need not have fast built-in numerical evaluation. The specialist Taylor evaluator covers finite-box polynomial exp/sin/cos phases with exact inputs and a rational remainder bound; budgets and conservative bounds can prevent meeting a tolerance. Other series still need external justified truncation or quadrature. Floating rounding is not enclosed by the analytic bound.
- Complexity and backend-query budgets do not provide a global deadline for every upstream symbolic assumption or transformation. Unsupported parameterized quartic profiles can still require expensive reasoning; an interrupted search supplies no convergence conclusion.
- Real level-set and singularity-dimension reductions require proved real polynomial coefficients. Complex denominator zero sets are split into proved finite real/imaginary polynomial equations; unknown coefficient domains retain unknown dimension/regularity. Complex Gaussian contours have their own narrower analytic rule.
- Convergence rules are sufficient, scoped theorems. Failure to prove convergence is unknown, rather than divergence. Branch-sensitive complex formulas require explicit prerequisites.
- Numerical verification is a consistency check for its supported finite continuous boxes, not a rigorous interval error certificate. Regularization replay reconstructs the same algorithm and prescription, rather than supplying an independent general proof.
- Minimum/latest release checks accompanying development runs can be focused gates rather than full dependency matrices. Read the validation report for exact versions, counts and scope.

## Known smaller issues

**Process isolation from stdin entry points.** With `backend_timeout_seconds` enabled, running from a pipe-fed Python entry point whose `__main__.__file__` is `<stdin>` can make spawned workers fail to reconstruct the main module. The query conservatively returns unavailable/unknown and can print a child traceback. Use a file-backed entry point with the usual main guard for isolated queries. This affects availability and diagnostics, not a finite or divergence certificate. Ordinary execution without the opt-in timeout is unaffected.

**Unscoped scalar Gaussian formulas on coefficient-pole fibers.** The scalar
Gaussian shortcut can return `pi*exp(1/(4*a**2))` for
`exp(-x**2-y**2+x/a)` over the whole plane without proving `a != 0`.
The structured `integrate_result` correctly marks this request unknown with
incomplete results, so it does not falsely certify the exceptional fiber.
Use explicit nonzero assumptions or parameter-cell orchestration. Tightening
scalar shortcut prerequisites would improve consistency and diagnostics.

**Intermittent parameter-cell completeness.** One full Python 3.11 installed-wheel
run left the family `x**2 <= a**3-a` incomplete. Its 47-test module, standalone
result and subsequent complete suite passed without a code change. The cause
has not been established; the validation report preserves the failed run.
This loses a coverage claim rather than supplying a false finite certificate.

**Exceptional-fiber rank work before geometry admission.** Coarea cell
specialization can compute a small map Gram determinant before enforcing the
source-variable geometry budget. A three-dimensional source with a
`max_variables=1` policy still performs a two-by-two determinant when specializing
`(a*x, y)` at `a=0`. The resulting source integral is correct, but larger symbolic
maps need a guard before this diagnostic/fallback decision.

## Reporting and maintenance

The README links here; strategy-specific prerequisites live in the mathematical guides and executable tests. A confirmed value/certificate defect gets a regression test and a fix. Unresolved defects stay here with a reproduction and consequence. Resolved items belong in release notes rather than an accumulating historical bug list. The project does not provide compatibility aliases for removed dependency APIs.

Full-space nonseparable profiles admit selected globally invertible triangular
polynomial shears followed by weighted even-power moments. Nontriangular charts,
changed halfspace/box boundaries and general oscillatory coupled tails need other
proved reductions. See [nonseparable improper integration](nonseparable-improper.md).
