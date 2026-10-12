# MultipleIntegrate

MultipleIntegrate evaluates definite multiple/multivariate integrals by combining SymPy-based evaluation with geometric reductions, moment rules, coordinate changes and some selected regularization prescriptions. It is useful when the structure of a density or domain supplies more information than a generic antiderivative search.

## Installation and dependencies

```sh
pip install multiple-integrate
```

Python 3.11 or newer is required. SymPy supplies symbolic expressions and fallback integration, `funcprops` supplies function-property queries, `semialg` supplies canonical region geometry and decompositions, and `mpmath` is used for numerical consistency checks. The package requires `semialg>=1.3.2`, `funcprops>=0.3.0`. The optional package `asymptotic>=0.2.1` handles supported asymptotic regularization work.

## Quick start

```python
import sympy as sp
from multiple_integrate import integrate, integrate_result
x, y, a = sp.symbols("x y a", real=True)
assert integrate(x**2 * y, (x, 0, 1), (y, 0, 2)) == sp.Rational(2, 3)
result = integrate_result(sp.sin(1/x)/x, (x, 0, 1))
assert result.value == sp.pi/2-sp.Si(1)
assert result.ordinary_convergent is True
assert result.absolute_convergent is False
```

Boolean formulas use the same `region=` argument:

```python
triangle = sp.And(x >= 0, y >= 0, x+y <= 1)
assert integrate(sp.exp(x + y), region=triangle, variables=(x, y)) == 1
```

Integration ranges are listed from innermost to outermost, not with the outer-to-inner ordering used, for example, by Mathematica. Dependent bounds use explicit triples; reversed independent ranges preserve orientation. Canonical `semialg` objects use `region=` and explicit `variables=`. Select `measure_dimension` for points, curves or surfaces, because ambient volume and intrinsic measure are distinct.

```python
import semialg
from multiple_integrate import integrate_cells
assert integrate(1, region=semialg.Box(((0, 1), (0, 2))), variables=(x, y)) == 2
family = integrate_cells(1, region=x**2 <= a, variables=(x,),
                         bounds={x: (-1, 1)}, parameters=(a,))
assert family.complete and family.value.subs(a, sp.Rational(1, 4)) == 1
```

Extensions include finite Laurent–Laplace moments on proved positive cutoffs, polynomial ellipsoid moments clipped by parallel affine planes, and weighted tangent-graph cutoffs that preserve the requested regulator. See [the formulas and prerequisites](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/laurent-clips-and-tangencies.md).

## Capabilities and prerequisites

| Family | Exact strategy and required proofs |
|---|---|
| Boxes and product densities | Cached geometry, bounded sparse polynomial moments, separability; independent bounds and finite coefficients |
| Polynomial times exponential/trigonometric functions | Affine-frequency recurrences with explicit zero-frequency branches |
| Polynomial Gaussian moments | Shifted erf recurrence on finite intervals and, with proved positive decay, half-lines/full lines |
| Endpoint and stretched-exponential moments | Exact endpoint factor orders; shifted/truncated gamma moments with proved decay and branch prerequisites |
| Rational densities | Pole-free linear/quadratic interval formulas and positive shifted-biquadratic full-line moments |
| Oscillatory improper densities | A few narrow Si/Ci and Dirichlet rules; conditional convergence is recorded separately from absolute convergence |
| Nonseparable improper profiles | Global triangular polynomial shear charts and weighted moments with proved origin and tail thresholds |
| Nonseparable polynomial phases on finite boxes | Exact uniformly convergent series; specialized bounded Taylor evaluation with analytic remainder |
| Parametrized geometry | Budgeted polynomial event cells, proved quadratic ellipsoid moments, coupled affine/graph reductions |
| Simplex, balls, shells and quadratic forms | Dirichlet, radial and Gaussian moments under sign, rank and dimension prerequisites |
| Polygonal domains and convex 3D polytopes | Boundary flux, cached polynomial moments; affine exponentials on polygons and fixed-normal polygons on proved cells |
| Graph domains, cusps and clipped polygons | Geometric reductions and proved contact/vanishing thresholds |
| Singular product intersections | Affine or polynomial graph charts with proved inverses, Jacobians and the original shared cutoff |

Use assumptions to express the prerequisites for the entire requested parameter cell. Sample points cannot certify a generic cell. There are complexity budgets that will decline expensive routes.

## Results, evidence and regularization

`integrate` returns the mathematical result without extra information. `integrate_result` adds outcomes, parameter cases, coverage, convergence evidence, diagnostics and limitations. Outcomes include finite, divergent, conditional, unsupported and unknown. An exact convergent series representation is finite; an unresolved arbitrary series is unknown. Coverage, ordinary convergence, absolute convergence and numerical consistency are separate claims.

Assumptions can be provided via the option `assumptions=` or via symbol-attached assumptions. Temporary SymPy `assuming(...)` facts are excluded from package proof queries and refinement, so cached evidence cannot escape its parameter scope.

Consistent convergence evidence is summarized across conditions only when they provably cover the requested parameter domain. Missing equality fibers, conflicting facts and unknown cells would prevent a global conclusion.

If you need the principal value or Hadamard finite part for a divergent integral, ask for them specifically. Obviously a regularized value does not establish ordinary convergence.

Selected affine/radial co-area dispatch, vector-valued maps, polynomial/Fourier and bounded rational angular amplitudes, caller-supplied parameter cells, and supplied rational, algebraic or polar charts use `multiple_integrate.coarea`; see [verified coarea reductions](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/coarea.md).

`Boole(condition)` creates a SymPy indicator `Piecewise((1, condition), (0, True))`. Both forms are accepted by `integrate` within the supported geometry and convergence rules.

## Reading guide

The [executable method example catalog](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/method-examples.md) has a representative example for every documented method.

The [method reference guide](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/method-references.md) maps mathematical methods to sources and implementation modules.

- [Supported regions and measures](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/supported-regions.md)
- [Bounded series evaluation](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/series-evaluation.md)
- [Selected symbolic families](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/selected-symbolic-families.md)
- [Exact fast methods and budgets](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/performance-methods.md)
- [Parameterized integration](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/parameterized-integration.md)
- [Result semantics and certfication](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/certification.md)
- [Regularization](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/regularization.md)
- [Polygon and polytope boundary reduction](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/boundary-reduction.md)
- [Nonseparable improper integration](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/nonseparable-improper.md)
- [Laurent moments, clips and tangencies](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/laurent-clips-and-tangencies.md)
- [Nonlinear moments and singularities](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/nonlinear-moments-and-singularities.md)
- [Sectors and contact charts](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/sectors-and-contact-charts.md)
- [Limitations and known issues](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/limitations.md)

Executable examples live in `examples/`, and `notebooks/multiple_integration.ipynb` is a demo notebook.

## Worked examples

### Layer-cake identity

For a nonnegative density `f`, the layer-cake identity is
`integral(f) = integral_0^infinity volume({f > t}) dt`, with infinite values
allowed by Tonelli's theorem. For `f=exp(-g)` and `g>=0`, an equivalent form is
`integral(exp(-g)) = integral_0^infinity exp(-t)*volume({g < t}) dt`.
For the positive-quadrant superellipse `g=x**4+y**4`, the sublevel area is
`Gamma(1/4)**2*sqrt(t)/(8*sqrt(pi))`:

```python
import sympy as sp
from multiple_integrate import integrate_result
x, y = sp.symbols("x y", real=True)
t = sp.Symbol("t", positive=True)
area = sp.gamma(sp.Rational(1, 4))**2 * sp.sqrt(t)/(8*sp.sqrt(sp.pi))
layer_value = sp.integrate(sp.exp(-t)*area, (t, 0, sp.oo))
result = integrate_result(sp.exp(-(x**4+y**4)), (x, 0, sp.oo), (y, 0, sp.oo))
assert sp.simplify(result.value-layer_value) == 0
assert sp.simplify(layer_value-sp.gamma(sp.Rational(1,4))**2/16) == 0
```

This demonstrates the identity and its value; the automatic planner may choose
another proved route for the same density. See the selected-symbolic-family
guide for endpoint factors and shifted or truncated gamma moments.

<!-- geometry-casebook-start -->
## Geometry casebook

The examples share these imports:

```python
from dataclasses import replace
import mpmath as mp
import semialg
import sympy as sp
from multiple_integrate import (HadamardFinitePart, IntegrationOutcome, ParameterCell,
    integrate_cells, integrate_result, regularized_result, verify_regularization_result)
```

### Parameter cells and graph contact

```python
x, y, a, b = sp.symbols("x y a b", real=True)
result = integrate_cells(
    1,
    region=sp.And(x >= 0, x <= a, y >= 0, y <= b - x**2),
    variables=(x, y),
    parameters=(a, b),
    assumptions=sp.And(a > 0, b >= a**2),
    cells=(ParameterCell(b > a**2), ParameterCell(sp.Eq(b, a**2), {b: a**2})),
)
assert sp.simplify(result.cases[0].value - (a * b - a**3 / 3)) == 0
assert sp.simplify(result.cases[1].value - 2 * a**3 / 3) == 0
```

### Cusp contact and divergence

```python
x, y = sp.symbols("x y", real=True)
base = x**3 - y**2
horn = sp.And(x > 0, x < 1, y**2 <= x**3)
result = integrate_result(base ** -sp.Rational(1, 2), region=horn, variables=(x, y))
divergent = integrate_result(x**2 / base, region=horn, variables=(x, y))
assert sp.simplify(sp.expand_func(result.value) - sp.pi) == 0
assert result.absolute_convergent is True
assert divergent.outcome is IntegrationOutcome.DIVERGENT
```

### Exact finite radial series

```python
x, y, z = sp.symbols("x y z", real=True)
rho = x**2 + y**2 + z**2
result = integrate_result(
    sp.exp(-rho) / (2 + rho), region=semialg.Ball((0, 0, 0), 1), variables=(x, y, z)
)
assert result.value.has(sp.Sum) and result.absolute_convergent is True
with mp.workdps(35):
    reference = 4 * mp.pi * mp.quad(lambda r: r**2 * mp.exp(-r * r) / (2 + r * r), [0, 1])
    assert abs(mp.mpf(str(result.value.evalf(30))) - reference) < mp.mpf("1e-25")
```

### Clipped polygon finite part

```python
x, y = sp.symbols("x y", real=True)
polygon = sp.And(x >= 0, y >= 0, x <= sp.Rational(1, 2), x + y <= 1)
result = regularized_result(
    1 / (x * y),
    region=polygon,
    variables=(x, y),
    regularization=HadamardFinitePart(hypersurface=x * y),
)
assert sp.simplify(result.value + sp.pi**2 / 12 - sp.log(2) ** 2) == 0
assert verify_regularization_result(result).valid
assert not verify_regularization_result(replace(result, value=result.value + 1)).valid
```

For an ordinary result inspect `result.outcome`, `result.complete`, `result.coverage`, and `result.convergence_evidence`; for a family inspect each child in `result.diagnostics['cell_results']`. A finite-part answer describes its specified cutoff and does not prove ordinary convergence.
<!-- geometry-casebook-end -->

## Contributing and project information

Contributions – whether they are bug reports, suggestions, or code for bugfixes or improvements – are welcome!

For development setup, architecture and extension guidance, see [Contributing](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/contributing.md). The [testing guide](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/testing.md) describes the mathematical, property, slow and packaging checks.

See the [release notes](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/CHANGELOG.md) for changes and the [limitations and known issues](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/docs/limitations.md) for unresolved cases. MultipleIntegrate is licensed under the GNU General Public License version 3 only (GPL-3.0-only). See [LICENSE](https://github.com/BhuvaneshBhatt/multiple_integrate/blob/main/LICENSE).
