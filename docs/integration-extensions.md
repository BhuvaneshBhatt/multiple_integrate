# Weighted radial and quadratic families

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

The exact strategies recognize `P(x)*exp(-a*q(x))` and
`P(x)*(b+q(x))**p`, where `P` is a finite-coefficient polynomial, `q` is a real
positive definite quadratic after centering, and the profile scale is finite.
Gaussian profiles require positive `a`. Full-space inverse-quadratic profiles
require positive `b` and sufficient tail decay for every nonzero polynomial term.
Gaussian profiles may include a pure radial power, provided it is locally
integrable. Matched Gaussian and shifted inverse-power profiles have a separate mixed-moment route under proved positive rates and shifts; see [Mixed Gaussian and inverse quadratic moments](proved-families.md).

On balls, spherical shells, and ellipsoids, polynomial numerators are integrated
with exact angular moments and scalar radial gamma or incomplete-beta integrals.
Canonical Semialg radial objects retain the native shortcut, including a rotated
ellipsoid shape matrix. The quadratic profile must match the radial geometry;
an arbitrary anisotropic Gaussian on a round ball is not covered by this family.

```python
import sympy as sp
from multiple_integrate import integrate_result

x, y = sp.symbols("x y", real=True)
r = integrate_result(x*x/(1+x*x+y*y)**3,
    (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
assert r.value == sp.pi/4
assert r.absolute_convergent is True
```

For a numerator monomial of degree `d` in dimension `n`, a pure radial power
requires `n+d+2*p > 0` at the origin and `n+d+2*p < 0` at infinity without
Gaussian decay. The local condition is unnecessary when a positive inner shell
radius excludes the origin. All absolute moment prerequisites are checked before
angular cancellation: an odd numerator cannot conceal a nonintegrable pole.
These sufficient term-wise tests can decline integrable densities whose leading
terms cancel. Rank-deficient forms, unproved parameter signs, coordinate-dependent
profile exponents, and nonmatching geometry fall back conservatively.

Regression coverage: `tests/test_weighted_quadratic_radial.py`,
`tests/test_extended_convergence.py`, `tests/test_asymmetric_affine_cutoffs.py`,
`tests/test_symbolic_cell_geometry.py`.

A nonnegative shift does not exclude its zero fiber. Supported quadratic/radial
families retain distinct positive and zero-shift branches. Constant-phase local
power thresholds can establish zero-fiber divergence; otherwise the unresolved
fiber stays unevaluated. Shells may have a finite zero-shift value because their
inner radius excludes the origin. No unconditional absolute certificate is
inferred merely from the positive-shift branch.

Regression coverage: `tests/test_measure_and_parameter_proof_scope.py`.
