# Regularization

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

Ordinary improper integration asks whether a limit exists. Principal-value and finite-part
prescriptions answer a different question: whether a singular integral has a finite value after
a specified cancellation or subtraction rule. The prescription is part of the mathematics, so
MultipleIntegrate never silently changes one regularization into another.

## Choosing a prescription

| Prescription | Singular geometry | Cutoff semantics | Typical use |
| --- | --- | --- | --- |
| `CauchyPrincipalValue` | finite 1-D poles | symmetric intervals around all interior poles | real rational and elementary 1-D singular integrals |
| `IteratedPrincipalValue` | coordinatewise singularities | ordered 1-D PVs | problems where integration order is part of the definition |
| `GeometricPrincipalValue` | isolated point | one shrinking sphere or ellipsoid | multidimensional odd/canceling point singularities |
| `HypersurfacePrincipalValue` | regular polynomial `g(x)=0` | one shared `|g| > epsilon` cutoff | codimension-one singular sets |
| `HadamardFinitePart` | poles or regular hypersurfaces | subtract divergent cutoff terms and retain the constant term | higher-order poles where PV alone is insufficient |

These values need not agree. In particular, an iterated PV is order-sensitive, while a geometric
PV couples all directions through one cutoff.

## Structured regularization results

Use `regularized_result()` when the cutoff and certificate matter:

```python
import sympy as sp
from multiple_integrate import HadamardFinitePart, regularized_result

x = sp.symbols("x", real=True)
result = regularized_result(
    x**-2,
    (x, -1, 1),
    regularization=HadamardFinitePart(),
)

assert result.value == -2
assert result.provenance.cutoff_integral == -2 + 2 / result.provenance.cutoff_parameter
```

The provenance records the regulator, singular locus, cutoff integral, asymptotic expansion,
divergent terms, constant term, backend, and replayable certificate.

## Hypersurface principal value

For a regular polynomial hypersurface `g(x)=0`, the package uses one global regulator
`|g| > epsilon`. A global graph chart is used when possible. Otherwise a certified local atlas
chooses transverse directions patch by patch. Explicitly solving every branch is not required;
a chart may retain the implicit relation `g(x)=t` together with its transverse derivative.

Multiplying a defining equation by a nonzero constant does not change the level-set PV when the
same geometric singular set and shared-cutoff semantics are retained.

## Hadamard finite part

If the cutoff integral has an expansion

\[
I(\varepsilon)=\sum_{k<0} a_k\varepsilon^k + \sum_j b_j(\log\varepsilon)^j + c + o(1),
\]

MultipleIntegrate discards the divergent terms and returns the constant coefficient `c`. When the
optional `asymptotic` package is installed, its public multiseries capability is preferred;
otherwise a local SymPy series fallback is used and recorded in provenance.

## Verification

`verify_regularization_result(result)` independently replays the stored algebraic claims. It
checks the cutoff expansion, divergent-term classification, constant-term extraction, and any
locally replayable geometric conditions. Verification detects tampering with the result value,
expansion, or divergent-term list; it is not a proof of facts supplied only by an external backend.

## Intersecting and repeated coordinate planes

Finite asymmetric boxes support an exact shared product cutoff for polynomial
Laurent densities whose poles lie on coordinate planes. The defining polynomial
may be a nonzero real multiple of a product of shifted coordinates with positive
integer powers; unused coordinates may have regular polynomial dependence.
This includes intersections such as `x*y=0` and repeated factors such as `x**2`.

```python
import sympy as sp
from multiple_integrate import HadamardFinitePart, regularized_result

x, y = sp.symbols("x y", real=True)
result = regularized_result(
    1/(x*y)**2, (x, -1, 1), (y, -1, 1),
    regularization=HadamardFinitePart(hypersurface=x*y),
)
assert result.value == 4
```

All orthants use the same `|g| > epsilon` cutoff. Exact product moments retain
powers and logarithms of epsilon in provenance. A Cauchy PV of this positive
example diverges; its finite part is a separate prescription. In three dimensions,
`1/(x*y*z)**2` on the unit cube has shared-cutoff finite part `-8`, illustrating why
an iterated finite part cannot silently replace the specified product regulator.

Widths must be certified finite positive. Density coefficients must be finite
and their common denominator nonzero; sufficient assumptions can establish
these prerequisites. Shifted planes may cross, touch, or lie outside the box;
signed interval moments and inclusion-exclusion preserve each contribution.
Additional unexcluded poles, non-product curved intersections, and general cusps are declined
by this route. Existing regular chart/atlas paths remain available; unsupported
singular geometry stays explicit. This is a family-specific extension, not a
resolution-of-singularities implementation.

For finite one-dimensional rational cutoffs with fully split real poles, a
canonical real primitive avoids algebraic complex-log limit artifacts. Pole
ordering must be certified from real comparisons and assumptions; structural
expression ordering is never used as a substitute.

## Coordinate bounds and trace scope

`bounds=` intersects the requested region before native integration or cutoff
strategies are selected. Centered box restrictions retain their native path;
other restrictions become explicit semialgebraic domain constraints. A restricted
singular domain outside the supported product-cutoff family can remain unknown
or unsupported, rather than receive the unrestricted region's answer.

Regulator provenance is collected only during `regularized_result()` calls.
Repeated flat `integrate()` calls do not retain an ever-growing history of
cutoff objects in their execution context.

Regression coverage: `tests/test_requested_bounds.py`.

## Regular densities across singular polynomial zero sets

For a continuous density on a finite closed box and a certified nonzero real
polynomial `g`, the zero set has ambient Lebesgue measure zero. Boundedness of the
density supplies an integrable majorant, so dominated convergence identifies the
`|g| > epsilon` limit with the ordinary integral. Both hypersurface PV and
hypersurface finite part can use this shortcut, even for a cusp, isolated zero,
intersecting planes, or a repeated factor. This does not integrate singular
densities across those sets.

```python
import sympy as sp
from multiple_integrate import HypersurfacePrincipalValue, regularized_result, verify_regularization_result

x, y = sp.symbols("x y", real=True)
r = regularized_result(1+x*x, (x,-1,1), (y,-1,1),
    principal_value=HypersurfacePrincipalValue(y*y-x**3))
assert r.value == sp.Rational(16,3)
assert r.provenance.cutoff_integral is None
assert verify_regularization_result(r).strength == "dominated_convergence"
```

The result records the actual regulator and cutoff domain, proved prerequisites,
and ordinary value. It does not fabricate an epsilon-dependent truncated integral.
Verification replays the continuity/nonzero-polynomial prerequisites, regulator
geometry, and ordinary value. A zero polynomial, an unproved nonzero parameter,
a noncompact box, or an unexcluded singular density declines the shortcut.

## Rational finite parts at infinity

One-dimensional real rational densities support one or both infinite endpoints.
The convention is `R=1/epsilon`: truncate a left-infinite interval at `-R` and a
right-infinite interval at `R`. Two infinite endpoints therefore use a symmetric
cutoff centered at zero. Finite poles, including finite endpoint poles, are
excluded with the same epsilon. Every real pole in the interval must be included;
an explicit incomplete list is refused. Polynomial terms are included in this
rational family. Real poles must be fully resolved and ordered, and logarithm
branches must be certified.

```python
import sympy as sp
from multiple_integrate import HadamardFinitePart, regularized_result

x = sp.Symbol("x", real=True)
r = regularized_result(1/x, (x,2,sp.oo), regularization=HadamardFinitePart())
assert r.value == -sp.log(2)
assert "R=1/epsilon" in r.provenance.details["infinity_cutoff"]
```

The cutoff convention is part of the prescription: translating or rescaling the
regulator can change its finite constant. This is not an ordinary improper
integral, an independently regularized endpoint limit, or a coordinate-invariant
finite-part operation. Nonrational infinite-endpoint finite parts remain unsupported.
Decaying terms such as `epsilon*log(epsilon)` are retained as vanishing remainders,
not reported as divergent subtractions.

Regression coverage: `tests/test_regular_density_cutoff.py`,
`tests/test_rational_infinity_finite_parts.py`.

## Asymmetric and affine product cutoffs

Independent affine level factors are supported when their invertible coordinate
map turns the requested domain into one certified finite box. Density, level
polynomial, and domain are transformed together, with the absolute Jacobian.
A tilted singular plane inside an arbitrary axis box does not meet this condition.
The original cutoff domain is retained in provenance. Verification replays the
supported product family and checks the affine matrix, inverse, and Jacobian;
this uses the same family algorithm and is not an independent integration proof.

```python
import sympy as sp
from multiple_integrate import HadamardFinitePart, regularized_result

x, y = sp.symbols("x y", real=True)
u, v = x+y, x-y
r = regularized_result(1/(u*v)**2,
    region=sp.And(u >= -1, u <= 2, v >= -1, v <= 3), variables=(x, y),
    regularization=HadamardFinitePart(hypersurface=u*v))
assert r.value == 1
assert r.provenance.details["affine_jacobian"] == sp.Rational(1, 2)
```

Asymmetry can leave a nonzero PV: `1/(x*y)` on `[-1,2] x [-1,3]` with
`g=x*y` has PV `log(2)*log(3)`. On the positive box `[0,2] x [0,3]`, its shared
product finite part is `(log(2)+log(3))**2/2`, rather than the iterated-cutoff
constant `log(2)*log(3)`. Changing the nonzero scale of `g` can change logarithmic
finite parts. The policy bounds the number of Laurent terms and interval pieces;
exceeding it refuses the route, without dropping any contribution.

Regression coverage: `tests/test_asymmetric_affine_cutoffs.py`.

Finite-part, geometric PV, and hypersurface PV routes currently use ambient
measure. An explicit non-ambient `measure_dimension` is unsupported on these
routes; it must not be silently ignored. Intrinsic ordinary integration uses the
selected chart measure separately.

Regression coverage: `tests/test_measure_and_parameter_proof_scope.py`.
