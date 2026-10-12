# Quadratic sectors, graph crossings and logarithmic contacts

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

The planar sector strategy integrates polynomial densities over one positive-definite
ellipsoid intersected with affine halfplanes through its center. The halfplanes may
be nonparallel. Whitening the ellipsoid and partitioning a tangent-half-angle line
at every real boundary root gives complete angular coverage. Radial polynomial
moments are elementary; angular rational primitives are differentiated before use.
Exact real algebraic matrix entries and clip directions are required. Symbolic
centers and radii require finite real centers and a proved positive squared radius.
Offset nonparallel clips and multiple quadratic boundaries remain outside this rule.

```python
import sympy as sp
from multiple_integrate import integrate_result
x, y = sp.symbols("x y", real=True)
result = integrate_result(x*y, region=sp.And(x*x+y*y <= 1, x >= 0, y >= 0), variables=(x,y))
assert result.value == sp.Rational(1,8)
assert result.complete and result.absolute_convergent
```

Polynomial graph envelopes can be subdivided where their complete intersection
roots cross the base interval. Every root must be real or proved nonreal, ordered,
and proved inside or outside the interval on the entire caller cell. Each surviving
piece then proves its envelope inequalities. There is no parameter sampling.
This extension currently certifies ambient polynomial integrals on finite planar
strips. Lower-dimensional transition fibers need their own measure treatment.

```python
import sympy as sp
from multiple_integrate import ParameterCell, integrate_cells
x, y, a = sp.symbols("x y a", real=True)
result = integrate_cells(1, region=sp.And(x>-1,x<1,y>0,y<x*x-a), variables=(x,y),
    parameters=(a,), cells=[ParameterCell(sp.And(a>0,a<1))], assumptions=sp.And(a>0,a<1))
assert result.complete
expected = sp.Rational(2,3)-2*a+sp.Rational(4,3)*a**sp.Rational(3,2)
assert sp.simplify(result.diagnostics["cell_results"][0].value-expected) == 0
```

Automatic event projection tracks leading coefficients, discriminants and
base-endpoint resultants of selected polynomial graph differences. Sign partitions
are candidate cells, not a full cylindrical algebraic decomposition. They need not
be connected or have constant topology; integration still proves each child.

For two polynomial graphs with common graph P and a gap c*(x-h)**m, c>0 and m>=2,
a global polynomial shear and translation lower the contact to a weighted horn.
The cutoff preserves the original hypersurface level, including its scale.
Logarithmic radial order zero uses verified rational primitives and
real-interval dilogarithms. Fractional radial orders and unproved angular
primitives still decline. Finite parts describe the specified cutoff and do not
certify ordinary convergence.

```python
import sympy as sp
from multiple_integrate import HadamardFinitePart, regularized_result, verify_regularization_result
x, y = sp.symbols("x y", real=True)
p = x**3+x
level = (y-p)*(y-p-(x-1)**2)
result = regularized_result((x-1)/level,
    region=sp.And(x>1,x<2,y>p,y<p+(x-1)**2), variables=(x,y),
    regularization=HadamardFinitePart(hypersurface=level))
assert sp.simplify(result.value-sp.pi**2/12) == 0
assert verify_regularization_result(result).valid
```

The chart records the forward map, inverse map, absolute Jacobian, original density
and region, and transformed regulator. Replay verifies these identities. Coordinate
permutation changes orientation but preserves the measure Jacobian.

General implicit nonlinear parameter geometry and arbitrary singular contacts are
still unresolved when there is no proved chart. This includes gaps with several
contact points or several competing terms, non-graph branches, and arbitrary
intersections of more than two tangent hypersurfaces. Such cases require additional
certified root isolation or resolution charts; the selected rules do not supply
that general machinery.

Tests in `test_sectors_and_contact_charts.py` execute all three examples and cover
independent sector identities, numerical original-coordinate cutoffs, equality
fibers, coordinate permutation, chart replay, and broken prerequisites.
