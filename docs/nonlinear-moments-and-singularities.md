# Truncated moments, nonlinear cells and weighted singularities

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

## Positive-cutoff Laplace moments

For finite real `p`, positive `L`, `m` a positive integer and `Re(rate)>0`,

```text
integral_L^U t^p exp(-rate*t^m) dt
  = [uppergamma((p+1)/m, rate*L^m)
     - uppergamma((p+1)/m, rate*U^m)] / (m*rate^((p+1)/m)).
```

The second tail is zero at infinity. A positive cutoff permits `p=-1` and
`p<-1`: the upper incomplete gamma is finite at a nonzero right-half-plane
argument even when its order is nonpositive. If `L=0`, the rule still requires
`p>-1`. It does not relax origin convergence or recombine branches on negative
coordinates. Shifted powers use `t=x-center`; intervals crossing the center
remain outside this rule.

```python
import sympy as sp
from multiple_integrate import integrate_result
x = sp.Symbol("x", real=True)
r = integrate_result(sp.exp(-x*x)/x**2, (x, 1, sp.oo))
assert r.absolute_convergent is True
assert sp.simplify(r.value - (sp.exp(-1)-sp.sqrt(sp.pi)*sp.erfc(1))) == 0
```

The exponential multiplier comparison proves a uniform bound when every
nonconstant monomial in `Re(phase)` is nonpositive on the requested Cartesian
coordinate signs. It then transfers an existing absolute-convergence proof
for a majorant to `exp(phase)*majorant`. A constant term may be finite and
positive; its exponential is a finite bound. Symbol assumptions about an
integration coordinate do not replace its requested range. Failure of this
sufficient rule is unknown, rather than divergence.

## Nonlinear parameter event cells

Automatic cells support bounded-degree rational polynomial events. A single
parameter uses exact ordered real roots, including higher-degree events;
multiple parameters use disjoint negative/zero/positive sign conditions for
polynomial factors. These conditions may have disconnected components. They
are not connected CAD cells or a proof that topology is constant on every
condition. Each child still proves its own geometry.

Coupled quadratic boundaries generate principal-minor and augmented-determinant
critical-level events. This detects selected degeneracies; arbitrary
intersections of several nonlinear boundaries still need additional proved
reductions. Event degrees, sign-tree size and algebraic work retain budgets.
No parameter samples stand in for a proof.

For one positive-definite quadratic inequality, polynomial
moments are evaluated directly after completing the square. The critical-level
sign is proved separately on each cell. Negative levels are empty; zero levels
are empty for a strict inequality or a point for a closed inequality. Ambient
measure at a point is zero, while intrinsic/top measure counts the point.
Parallel affine clips have a separate [ellipsoid-slab rule](laurent-clips-and-tangencies.md); intrinsic curved surfaces remain unsupported.

```python
import sympy as sp
from multiple_integrate import integrate_cells
x, y, a, b = sp.symbols("x y a b", real=True)
r = integrate_cells(1, region=x*x+y*y <= a*a+b*b-1,
                    variables=(x,y), parameters=(a,b))
assert r.complete
assert [c.value for c in r.cases] == [0, 0, sp.pi*(a*a+b*b-1)]
```

## Weighted horns and implicit pole geometry

The horn rule covers `0<x<A, y**n<=k*x**m`, with positive integer `m`, positive
even integer `n`, finite `A>0` and `0<k<=1`. For density
`P(x,y)/(x**m-y**n)**s`, the chart `y=x**(m/n)*t` separates radial and angular
singularities. A numerator monomial `x**p*y**q` has radial threshold

```text
h = 1+p+m*(q+1)/n-m*s > 0.
```

At contact `k=1`, it additionally requires `s<1`. For even `q`, its moment is
`2*A**h*beta((q+1)/n,1-s)/(n*h)`; strict `k<1` uses `betainc(...,0,k)`.
Exact defining-equation division proves numerator vanishing before testing
thresholds. Odd terms vanish only after absolute integrability is established.
Definite-sign threshold failures prove divergence; other cases remain unknown.
These are ordinary integrals on proved horn domains, not a universal finite-part
algorithm for arbitrary tangent intersections.

```python
import sympy as sp
from multiple_integrate import integrate_result
x, y = sp.symbols("x y", real=True)
r = integrate_result(1/sp.sqrt(x**4-y*y),
                     region=sp.And(x>0,x<1,y*y<=x**4), variables=(x,y))
assert r.absolute_convergent is True
assert sp.simplify(sp.expand_func(r.value)-sp.pi) == 0
```

Algebraic pole diagnostics split a complex polynomial into real and imaginary
polynomial equations when their coefficient domains are proved finite. They
query real geometry and certify full Jacobian rank on the zero set using a
budgeted simultaneous minor equations. Transverse intersections can receive
regularity evidence; tangent rank failures are retained explicitly. Unknown
coefficient domains, unknown backend answers or exceeded budgets cannot receive
certified geometry. This diagnostic evidence does not itself prescribe a
regularization or prove convergence.

Executable coverage: `tests/test_truncated_nonlinear_horns.py` and
`tests/test_quadratic_real_geometry.py`. The former includes numerical moment
oracles, exceptional fibers, coordinate assumptions, reversals, nonlinear
cell budgets and independent beta/reflection identities.
