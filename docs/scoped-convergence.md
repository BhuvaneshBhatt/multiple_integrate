# Scoped convergence proofs

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

A finite iterated answer alone does not prove an ordinary multiple integral or
justify changing integration order. Inspect `ordinary_convergent`,
`absolute_convergent`, and `convergence_evidence` on `integrate_result()`.
Each claim is valid under the result's assumptions and its own condition.
`None` means that the implemented rules did not establish the claim.

| Rule | Prerequisites | Evidence |
| --- | --- | --- |
| Compact continuity | independent finite closed box; finite polynomial coefficients or a supported continuous composition | ordinary and absolute convergence |
| Local radial power | `(sum(x_i**2))**p`, real `p`, finite box containing the origin in its interior | convergence iff `2*p + n > 0` |
| Full-space radial tail | `(1 + sum(x_i**2))**p` on all of real n-space | convergence iff `2*p + n < 0` |
| Absolute product | independent ranges; each factor is a certified endpoint/tail power or polynomial Gaussian moment; finite coefficient | absolute convergence and justified Fubini interchange |
| Comparison and scaling | bounded real sine/cosine or unit-modulus phase times a certified majorant; positive even polynomial remainders; SPD affine radial powers | sufficient absolute convergence |
| Finite sums and decomposition | all summands or every certified domain piece absolutely integrable | ordinary and absolute convergence, without relying on cancellation |
| Local polynomial vanishing | finite polynomial numerator over an SPD point-singular power on a finite box | lowest Taylor degree improves the local threshold |
| Coupled Gaussian moments | polynomial times one exponential of a real quadratic on all of real n-space; finite coefficients; negative Hessian positive definite | absolute convergence and justified Fubini interchange |

Compact composition rules include sums, products, exponentials, sine, cosine,
absolute values, and supported powers/logarithms with certified real-domain
prerequisites. Positive polynomial bases may be checked through Semialg, within
the supplied policy. This is a sufficient-rule corpus, not a general continuity
classifier. Reusable sign, symmetry, range, and singularity queries elsewhere
continue to use the Funcprops adapter.

```python
import sympy as sp
from multiple_integrate import integrate_result

x, y = sp.symbols("x y", real=True)
f = sp.exp(-2*x*x + 2*x*y - 2*y*y)
r = integrate_result(f, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
assert r.absolute_convergent is True
assert r.ordinary_convergent is True
assert sp.simplify(r.value - sp.pi/sp.sqrt(3)) == 0
```

Radial divergence establishes ordinary divergence because these densities are
nonnegative. The implication from absolute divergence to ordinary divergence is
not applied to signed or oscillatory densities. Convergence can be proved even
when evaluation remains unknown; proving existence does not manufacture a value.
A proven ordinary divergence can classify an unevaluated value as divergent.

For `(x*x-y*y)/(x*x+y*y)**2` on the centered unit square, the two iterated
values are `-pi` and `pi`. The available rules retain unknown ordinary and
absolute evidence. Singular/rank-deficient Gaussian exponents, dependent bounds,
and unproved parameter thresholds are also declined by these rules.

Regression coverage: `tests/test_scoped_convergence.py`,
`tests/test_robustness_evidence.py`, `tests/test_parameter_cells.py`.

Comparison work is recursive to depth three and respects the configured piece
budget. Power-log rules handle selected zero and infinity endpoints, including
`1/(x*log(x)**2)` away from the other logarithmic zero. A real sine multiplier
also satisfies `abs(sin(h)) <= abs(h)`, allowing certified polynomial vanishing
to control a local singularity. Complex exponential phases require real phase;
a growing real exponential cannot use that bound.

Products of real powers of `log(x)` and `-log(x)` use the sum of every
logarithmic exponent in the absolute-integrability threshold. Opposite branches
can change a constant complex phase, but do not remove a divergent magnitude.
For example, `sqrt(-log(x))/(x*log(x)**(3/2))` has magnitude
`1/(x*abs(log(x)))`; the power-log rule supplies no convergence certificate
at zero or infinity.

For a complete Semialg domain decomposition, each transformed piece must have
its own absolute proof before that evidence is propagated to the original domain.
An uncertain piece prevents propagation. Native unions receive the same check.
No ordinary-convergence claim follows from merely cancelling divergent pieces.

Regression coverage: `tests/test_extended_convergence.py`,
`tests/test_weighted_quadratic_radial.py`.

## Summaries across parameter cells

Repeated consistent unconditional facts no longer obscure a convergence conclusion. A scoped summary is established only when the known facts have a single truth value and their conditions provably cover the result's assumption domain. Contradictions, missing equality fibers and unresolved complements retain `None`. An evaluated finite expression alone supplies no absolute-convergence theorem.

Moving-point integration records branch finiteness before attempting a parent summary. For example, the intrinsic point density `Piecewise((1/a,a>0),(0,True))` on `Eq(x,a)` is finite on the supplied cell `a>=0`, including its zero fiber. Both summary accessors are True. The same summary contract applies to regularization results; a finite part itself does not prove ordinary convergence. Scoped coverage work uses a retained request budget, and the result caches the summary for repeated inspection.

Regression coverage: `tests/test_convergence_summaries.py`.

## Additional cheap sufficient rules

The [selected family guide](selected-symbolic-families.md) describes beta endpoint,
generalized Laplace, Mellin, coercive exponential and positive even-tail proofs.
Recognition and exact evaluation share the same prerequisites. These sufficient
rules add absolute-convergence evidence without inferring divergence from failure.
Divergence theorems also require a defined external parameter domain; undefined
coefficient or phase fibers remain unknown.
