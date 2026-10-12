# Polygon and polytope boundary reduction

The boundary engine integrates polynomial densities and polynomial amplitudes times
an affine exponential on selected proved polygonal domains. It reuses exact edge
recurrences and coefficient-independent moment tables. Polynomial densities also
support recursive facet reduction on rational convex three-dimensional polytopes.
See [Lasserre's polytope reduction](https://doi.org/10.1090/S0002-9939-98-04454-2)
and the [method reference guide](method-references.md).

## Polynomial flux and metric factors

For a homogeneous polynomial `p_k` of degree `k` in dimension `n`,
`F_k(x)=x*p_k(x)/(n+k)` satisfies `div(F_k)=p_k`. Its outward facet flux therefore
integrates `p_k`. The domain need not contain the coordinate origin.

For `a*x+b*y+c<=0`, the supporting line is `z(t)=z0+t*(-b,a)`.
The outward normal measure is `(a,b) dt`, and `z(t).(a,b)=-c`.
An edge contributes `-c/(k+2)` times its polynomial moment. A hole reverses
this normal; origin-containing edges can have zero flux but still enter the proof.

In three dimensions, solve a supporting plane for coordinate `j` with `n_j!=0`.
The chart's outward normal measure is `n/abs(n_j) du dv`; hence its flux factor
is `-c/abs(n_j)`. The remaining polygon moments use the same planar evaluator.
This accounts for orientation and induced metric without unit-normal square roots.

## Affine exponential densities

For `P(x,y)*exp(a*x+b*y+c)`, choose a coordinate with a proved nonzero rate `r`.
The finite polynomial `Q=sum((-1)**j*D**j(P)/r**(j+1))` satisfies
`D(Q*exp(phase))=P*exp(phase)`. The field with that component reduces the integral
to polynomial–exponential edge moments. Constant edge phases use direct polynomial
moments; zero rates are never divided by. If both rates vanish, the ordinary
polynomial route applies. An unresolved rate declines; split its zero fiber explicitly.

## Geometry certification and scope

Convex polygons accept conjunctions of affine inequalities with **rational fixed
normals** and offsets proved finite and real. Offsets may depend on parameters.
Every supporting line is intersected with all halfspaces, and every endpoint order
must hold throughout the supplied assumptions or parameter cell. Unknown order
or unbounded edges decline. Duplicate constraints and isolated vertex contacts
produce no extra edge. Axis-aligned boxes keep their Cartesian dispatch.

Rational Boolean combinations of affine inequalities additionally support nonconvex
polygons, holes, disconnected components and overlapping unions. The exact line
arrangement splits every supporting line at all intersections. Rational witnesses
on its two sides identify the source Boolean formula's interior; linear signs are
constant on each arrangement cell. These are exact arrangement certificates, not
numerical samples of a parameter family. Unbounded boundary intervals, an occupied
exterior, unbalanced directed endpoints or nonpositive oriented area decline.
Overlapping unions count each occupied cell once.

Three-dimensional support is currently **bounded rational convex halfspace
intersections with polynomial densities**. Each supporting plane is restricted by
all halfspaces, yielding a certified planar facet. Positive oriented volume,
bounded facet charts and all constraint incidences are required. Nonconvex 3D,
symbolic 3D vertices, changing symbolic normals and curved boundaries are outside
this boundary route. Semialg remains the owner of general canonical geometry.

The routes use ordinary ambient integration, or matching full-dimensional intrinsic
measure. A certified bounded polygon of zero area returns zero only for ambient
area (or dimension 2); intrinsic/top integration on its collapsed fiber declines.
Strict straight boundaries have zero ambient measure. Coefficients must be finite;
a singular density does not qualify merely because its support has zero area.
Principal-value and finite-part prescriptions retain their own routes.

## Parameter cells, caches and budgets

```python
import sympy as sp
from multiple_integrate import integrate_result, integrate_cells, ParameterCell
x, y, z, a, b = sp.symbols("x y z a b", real=True)
polygon = sp.And(x >= 0, y >= 0, x <= 2, y <= 2, x+y <= 3)
result = integrate_result(x*y, region=polygon, variables=(x, y))
assert result.value == sp.Rational(21, 8)
assert result.method == "polynomial_boundary"
assert result.complete and result.absolute_convergent is True

triangle = sp.And(x >= 0, y >= 0, x+y <= 1)
exponential = integrate_result(sp.exp(x+y), region=triangle, variables=(x, y))
assert exponential.value == 1
assert exponential.method == "affine_exponential_boundary"

clipped = sp.And(x >= 0, y >= 0, x+y <= a, y <= b)
family = integrate_cells(1, region=clipped, variables=(x, y), parameters=(a, b),
    assumptions=sp.And(a > 0, b >= 0),
    cells=(ParameterCell(sp.Eq(b, 0)), ParameterCell(sp.And(b > 0, b < a)),
           ParameterCell(sp.Eq(b, a)), ParameterCell(b > a)))
assert family.complete and family.value.subs({a: 2, b: 1}) == sp.Rational(3, 2)
assert family.value.subs({a: 2, b: 0}) == 0

tetrahedron = sp.And(x >= 0, y >= 0, z >= 0, x+y+z <= 1)
assert integrate_result(x*y*z, region=tetrahedron, variables=(x, y, z)).value == sp.Rational(1, 720)
```

Polygon moment tables cache individual monomial moments independently of density
coefficients. Convex geometry cache keys include the whole cell conditions; Boolean
arrangements have a separate bounded cache. `clear_cache()` clears all of them.
Each call rebuilds coverage evidence for its own source formula and assumptions.
Budgets are checked before cache lookup, and constrain polynomial degree/expansion,
constraint count, arrangement intersections, edges and recursive facet-edge totals.

Successful results record `polynomial_boundary` or `affine_exponential_boundary`,
complete domain coverage, facets and source halfspaces, ordinary convergence and
absolute convergence. The proof is a continuous finite density on a certified
bounded closure; zero-area fibers carry their separate ambient-null basis.

Regression coverage: `tests/test_boundary_reduction.py` and
`tests/test_boundary_extensions.py`. Independent iterated, rectangle-difference and
Dirichlet identities check values; cache, budget, cell-threshold, dimension, union,
hole, orientation and no-general-integrator checks guard the prerequisites.
