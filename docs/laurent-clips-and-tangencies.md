# Laurent moments, ellipsoid slabs and tangent cutoffs

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

These selected rules extend the automatic portfolio. Each rule proves its prerequisites; an unmatched family remains unknown or proceeds to the existing fallback. This page is executed by `tests/test_laurent_clips_and_tangencies.py`.

## Finite Laurent amplitudes and comparisons

Generalized Laplace moments accept bounded finite Laurent amplitudes and polynomial amplitudes times a single real power. For example, `(1+t)/t**2` becomes two incomplete-gamma moments. The coordinate `t` must be strictly positive throughout the requested truncated interval when a negative power is not integrable at zero. A cutoff at zero retains the separate `p>-1` threshold for every nonzero term. Full-line formulas require integer powers and absolute integrability before odd cancellation. Expansion and term-count budgets apply; rational amplitudes with other poles are not Laurent polynomials.

```python
import sympy as sp
from multiple_integrate import integrate_result
x = sp.Symbol("x", real=True)
f = (1+x)/x**2 * sp.exp(-x*x)
r = integrate_result(f, (x, 1, sp.oo))
assert r.absolute_convergent is True
expected = sp.exp(-1)-sp.sqrt(sp.pi)*sp.erfc(1)-sp.Ei(-1)/2
assert sp.simplify(r.value-expected) == 0
```

Additional sufficient comparisons cover finite Laurent amplitudes times powers of one affine logarithm. After mapping to `t`, an origin interval must lie in `0<t<B<1`, and a tail must start at `t>L>1`. Every term obeys the relevant power threshold; the borderline `t**-1` term requires logarithmic exponent `<-1`. Crossing the logarithm's zero at `t=1` does not receive this proof. A proved real cosine phase also gives `|1-cos(z)|<=|z|**2/2`. Nonnegative finite powers of bounded multipliers inherit a bound only when the exponent is independent of integration coordinates. These are absolute-convergence proofs, not numerical convergence claims; failed prerequisites give unknown.

## Parallel affine clips of an ellipsoid

A polynomial density on one positive-definite quadratic sublevel can be integrated with parallel affine halfspace clips. This includes rotated/translated ellipsoids and slabs bounded by several parallel planes. The rule proves coefficient finiteness, definiteness, parallelism and the ordering of clipped endpoints; caller-supplied parameter cells can supply those order proofs.

The implementation conditions on the affine clip coordinate. Conditional polynomial moments use a covariance recurrence, and the remaining interval moments use incomplete beta functions. Empty slabs have zero integral; a collapsed slab has zero ambient measure. Intrinsic integration on a collapsed slice is declined. Nonparallel clips, indefinite quadratics and intrinsic curved surfaces are outside this rule. A beta-function result can be exact without being elementary.

```python
import sympy as sp
from multiple_integrate import integrate_cells, ParameterCell
x, y, a = sp.symbols("x y a", real=True)
r = integrate_cells(1, region=sp.And(x*x+y*y<=a*a+1, x>=0),
                    variables=(x,y), parameters=(a,), cells=[ParameterCell(sp.true)])
assert r.complete
assert sp.simplify(r.value-sp.pi*(a*a+1)/2) == 0
```

## Tangent graph intersections with the original regulator

The selected tangent family is `g=K*y*(y-c*x**m)` on `0<x<A, 0<y<c*x**m`, with integer `m>=2`, proved finite `A,c>0`, and finite nonzero real `K`. The chart

```text
x=u, y=c*u**m*t,  u>0, 0<t<1
inverse: u=x, t=y/(c*x**m)
absolute Jacobian: c*u**m
|g|>epsilon  becomes  |K|*c**2*u**(2*m)*t*(1-t)>epsilon.
```

is invertible away from the excluded vertex. The vertex lies on `g=0` and is absent from every positive cutoff. Chart, inverse, Jacobian, source and transformed regulators, and the allowed epsilon interval are recorded in provenance. No independent-coordinate cutoff replaces this regulator.

Supported rational densities pull back to finite Laurent monomials in `u,t,1-t`. Each radial order `(p+1)/(2*m)` must be an integer. Nonzero orders use bounded rational calculations; zero orders use rational and dilogarithmic angular primitives. Polynomial shears of translated monomial graph contacts are supported by a separate lowering chart. Other poles and general implicit contacts are declined. The resulting cutoff can prove divergence of a principal value while still supporting a finite part. Regularization replay checks the same cutoff algorithm; it is not an independent mathematical certification.

```python
import sympy as sp
from multiple_integrate import HadamardFinitePart, regularized_result
x, y = sp.symbols("x y", real=True)
g = y*(y-x*x)
domain = sp.And(x>0,x<1,y>0,y<x*x)
r = regularized_result(x**5/g, region=domain, variables=(x,y),
                       regularization=HadamardFinitePart(hypersurface=g))
assert r.value == sp.Rational(1,2)
assert r.provenance.details["realization"] == "tangent_graph_cutoff"
```

In this example the cutoff expansion is `log(epsilon)/2+1/2+o(1)`. Scaling the prescribed level to `K*g` changes the constant to `(1-log(abs(K)))/2`. This illustrates why the regulator must be transformed exactly and why a finite part is distinct from an ordinary convergent integral.

The [contact extension](sectors-and-contact-charts.md) also supports zero logarithmic
radial order and polynomial shears with translated finite-order contact points.
