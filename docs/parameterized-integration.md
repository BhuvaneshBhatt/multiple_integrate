# Parameterized integration and topology changes

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

For the family `R_a = {(x,y): y=0, x²<=a}`, real parameter space has three
exhaustive cells: `a<0`, `a=0`, and `a>0`. The first is empty, the second is a
point, and the third is the segment `[-sqrt(a),sqrt(a)]` on the x axis.
Ambient area is zero on every cell. One-dimensional measure is zero on the
first two cells and `2*sqrt(a)` on the positive cell. Intrinsic counting measure
at the transition point is one, so “intrinsic measure” is not a fixed measure
across a dimension drop.

```python
import sympy as sp
from multiple_integrate import integrate
from multiple_integrate.adapters.semialg import reduce_intrinsic_domain

x, y, a = sp.symbols("x y a", real=True)
formula = sp.And(sp.Eq(y, 0), x**2 <= a)
positive_fiber = formula.subs(a, 4)
cover = reduce_intrinsic_domain(1, positive_fiber, variables=(x, y))
assert cover is not None and cover.certified
assert cover.dimension == 1
assert cover.pieces[0].limits == ((x, -2, 2),)
assert integrate(1, region=positive_fiber, variables=(x, y), measure_dimension=1) == 4
length = sp.Piecewise((0, a <= 0), (2*sp.sqrt(a), True))
```

The piecewise length above is assembled from the mathematical cells, not a claim
that MultipleIntegrate automatically derives every parameter decomposition.
Specialize geometry on each certified parameter cell or supply justified
assumptions and inspect the returned conditions. The Semialg intrinsic-cover
API supplies the verified fiber charts. Never infer a global family cover from
one generic fiber. A failed empty-fiber cover can return `None`; ambient
integration over the certified empty formula still returns zero.

A second example, `1-a <= x² <= 1`, is empty for `a<0`, two isolated points at
`a=0`, two intervals for `0<a<1`, and one interval for `a>=1`. At `a=1` the inner
boundaries coincide. Its length is `0`, `2-2*sqrt(1-a)`, and `2` on the respective
cells. Boundary points have zero length, but must be retained for counting
measure. Tests compare equivalent polynomial inequalities, coordinate renaming,
and the transition fibers, and inspect chart bounds rather than only totals.

For symbolic convergence cells, use `integrate_result(exp(-a*x), (x,0,oo))` and
inspect `parameter_strata`; geometry cells and convergence cells need not agree.

Regression coverage: `tests/test_geometry_reference_corpus.py`,
`tests/test_convergence_semantic_corpus.py`.

## Caller-supplied cell orchestration

`integrate_cells()` validates a supplied partition, evaluates each cell under its
condition, and retains per-cell outcomes and convergence evidence. Omit `cells`
for the budgeted polynomial-event sign partition described in
[the fast-method guide](performance-methods.md); broader families still require
a caller-supplied proved partition. `ParameterCell` accepts optional simultaneous substitutions
for exceptional fibers, but each substitution must follow from the cell
condition. Sample points on a generic cell are rejected.

```python
import sympy as sp
from multiple_integrate import ParameterCell, integrate_cells

x, a = sp.symbols("x a", real=True)
r = integrate_cells(
    x, (x, 0, a), parameters=(a,),
    cells=(ParameterCell(a < 0), ParameterCell(sp.Eq(a, 0), {a: 0}),
           ParameterCell(a > 0)),
)
assert r.complete
assert r.value.subs(a, 0) == 0
assert r.value.subs(a, 2) == 2
```

This example uses oriented integration ranges: a negative upper endpoint reverses
the interval, rather than defining an empty set. For geometric empty/nonempty
transitions, pass an explicit region formula. Parameter-only predicates are
refined under the cell condition before geometry is measured. An exceptional
fiber can be specialized exactly, including its density, bounds, and regulator.

Declare distinct real symbols in `parameters=`. Cells must be certified disjoint;
overlap or uncertain disjointness raises before any child is evaluated. The
policy's Semialg `max_pieces` bounds the number of supplied cells, and its query
complexity limits apply to partition certification. An uncovered complement is
retained as an unknown branch. Coverage of parameter space is separate from
coverage and completeness of each child integration result. Inspect
`diagnostics['cell_results']` for the latter.

Within a supplied cell, the native symbolic recognizer supports independent
axis-wise polynomial inequalities and equalities of degree at most two, including
Boolean unions and intersections. Coefficient signs, discriminants, root ordering,
and bound intersections must follow from the cell assumptions. It preserves
endpoint openness for empty and point fibers, combines overlapping intervals,
and counts isolated roots only for the selected measure. The orchestration can
discover the narrow transition partition described in the fast-method guide;
it does not perform general symbolic CAD. Selected coupled affine box images and polynomial graph strips have certified
reductions. Arbitrary coupled nonlinear domains
and uncertain root ordering can still remain unknown.

```python
import sympy as sp
from multiple_integrate import ParameterCell, integrate_cells

x, a = sp.symbols("x a", real=True)
r = integrate_cells(
    1, region=sp.And(x*x >= 1-a, x*x <= 1), variables=(x,),
    parameters=(a,), cells=(ParameterCell(a < 0), ParameterCell(sp.Eq(a, 0), {a: 0}),
        ParameterCell(sp.And(a > 0, a < 1)), ParameterCell(a >= 1)),
)
assert r.complete
assert r.value.subs(a, 0) == 0
assert r.value.subs(a, sp.Rational(3, 4)) == 1
assert r.value.subs(a, 2) == 2
```

Fixed intrinsic measure dimension must be chosen deliberately across topology
changes and dimension drops. A point has intrinsic mass one but zero length;
counting measure on a positive-dimensional interval is not a finite point sum.
The symbolic path declines such a request rather than asserting a finite value.

Regression coverage: `tests/test_parameter_cells.py`.

Regression coverage: `tests/test_symbolic_cell_geometry.py`.

Conjunctive linear equalities infer exact exceptional-fiber substitutions automatically; for example `ParameterCell(Eq(a, 0))` specializes `a` to zero. No sample point is chosen on a generic cell. See [proved families](proved-families.md) for scoped branch evidence and the remaining geometry limits.

If an exceptional cell makes an explicitly prescribed hypersurface identically zero, undefined, or nonfinite, that cell is retained as unsupported with its unevaluated request. Valid sibling cells still run, and parameter-space coverage remains separate from integration completeness. Regression coverage: `tests/test_heuristic_alarm_ownership.py`.

## Topology and measure transitions

Split equality/contact fibers explicitly. A generic intrinsic interval can collapse to a point with counting mass one; substituting the boundary into its length formula would give zero. Distinct point branches can also merge. A caller cell containing such transitions must be split or remain unresolved. The graph example and equality cell are in [parameter geometry](geometry/parameter-geometry.md).

## Multiple parameters and coupled affine events

Automatic orchestration supports bounded-degree rational polynomial sign conditions in multiple declared real parameters. Only certified empty sign cells are pruned. Equality fibers remain explicit, and the piece budget applies during construction. Parameter-space coverage does not imply a complete answer on every cell.

```python
import sympy as sp
from multiple_integrate import integrate_cells
x, y, a, b = sp.symbols("x y a b", real=True)
r = integrate_cells(x*x, region=sp.And(x >= a, x <= b),
                    variables=(x,), parameters=(a, b))
assert r.complete and r.absolute_convergent is True
assert r.value.subs({a: 1, b: 2}) == sp.Rational(7, 3)
assert r.value.subs({a: 2, b: 1}) == 0

formula = sp.And(x >= 0, x <= 1, y-a*x >= 0, y-a*x <= b)
r = integrate_cells(1, region=formula, variables=(x, y), parameters=(a, b))
assert r.complete
assert r.value.subs({a: 2, b: 3}) == 3
```

Coupled linear forms track possible affine rank drops through determinant zeros. Generic cells use a proved real inverse and absolute Jacobian; a collapsed cell is measured in its own dimension rather than by substituting into a generic volume formula. Selected graph reductions can provide the same guarantee inside caller-supplied cells. No sample point certifies a generic family.

Rational geometry coefficients are allowed when their pole events fit the supported partition. For `0<=x<=1/a`, the positive cell has length `1/a`, the negative cell is empty, and `a=0` is an unknown **undefined geometry** fiber. An oriented explicit range `(x,0,1/a)` instead has signed value `1/a` on either nonzero cell. The undefined fiber is neither an empty region nor a divergence proof.

Likewise, specializing a density such as `1/a` at `a=0` retains an unknown original request. A Piecewise formula giving an explicit removable value at zero remains eligible for finite integration. A parameter-only pole is different from a genuine coordinate singularity whose ordinary integral diverges. Prescribed regulators that collapse to an improper hypersurface retain the separate unsupported outcome described above.

Regression coverage: `tests/test_parameter_geometry_extensions.py`, `tests/test_parameter_specialization_safety.py`.

Further supported cases and prerequisites are described in [the extension guide](nonlinear-moments-and-singularities.md). Regression coverage: `tests/test_truncated_nonlinear_horns.py`.
