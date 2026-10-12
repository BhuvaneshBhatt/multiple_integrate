# Selected symbolic families and cheap existence proofs

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

The automatic planner tries these bounded recognizers before general singularity
analysis and symbolic integration. The specialist implementation is
`multiple_integrate.selected_families`; it adds no package-root exports.
Recognition uses exact polynomial coefficients, strict sign/endpoint tests and
existing expansion/degree/coordinate budgets. A missing prerequisite declines
the rule. It does not imply divergence.

## Finite-interval beta moments

For real finite `lo < hi`, the density
`P(x)*(x-lo)**p*(hi-x)**q` has absolutely convergent endpoint moments when
`p > -1` and `q > -1`, with real finite exponents and finite polynomial
coefficients. Affine rescaling gives a finite sum of beta functions. The bases
are nonnegative inside the interval, so fractional powers use the real positive
branch. Reversed explicit limits negate the result. Unknown exponent signs or
endpoint orders do not receive a certificate. Exact polynomial division extracts
proved endpoint factors: if their orders are `m` and `n`, the thresholds become
`p+m > -1` and `q+n > -1`. Symbolic coefficients that vanish on exceptional
parameter fibers may leave the proof conservative; no sampled fiber supplies an order.

```python
import sympy as sp
from multiple_integrate import integrate_result
x = sp.Symbol("x", real=True)
r = integrate_result((x+2)**sp.Rational(-1,2)*(3-x)**sp.Rational(-1,2), (x, -2, 3))
assert sp.simplify(sp.expand_func(r.value)-sp.pi) == 0
assert r.absolute_convergent is True
```

## Generalized Laplace moments

For positive integer `m`, `P(x)*exp(b-a*x**m)` on `[0,infinity)` has exact gamma
moments if `Re(a)>0`, all coefficients and `b` are finite, and the polynomial
amplitude has finite coefficients. One real power `x**p` is also supported if
`p>-1`. The monomial moment is
`exp(b)*gamma((p+1)/m)/(m*a**((p+1)/m))`.

Strictly positive truncated cutoffs also support bounded finite Laurent amplitudes and polynomial amplitudes times a real power. Each term uses an upper incomplete gamma difference; a zero cutoff retains its origin threshold. See [Laurent moments, clips and tangencies](laurent-clips-and-tangencies.md).

Complex decay rates stay in the open right half-plane. This is the gamma-Laplace
identity after the positive real substitution `t=x**m`, with its principal
complex power; a purely imaginary rate does not satisfy the absolute-decay
prerequisite. On the whole real line, even `m` and polynomial amplitudes have
zero odd moments and twice the even half-line moments. Absolute convergence is
proved before odd cancellation. A growing exponential or odd full-line phase
must decline this shortcut.

```python
import sympy as sp
from multiple_integrate import integrate_result
x = sp.Symbol("x", real=True)
r = integrate_result((1+x*x)*sp.exp(-x**4), (x, 0, sp.oo))
expected = (sp.gamma(sp.Rational(1,4))+sp.gamma(sp.Rational(3,4)))/4
assert sp.simplify(r.value-expected) == 0
assert r.absolute_convergent is True
```

## Rational Mellin moments

For positive finite real `a,c`, positive integer `m`, real finite `q`, and a
monomial numerator `x**p`, the rule for `x**p/(c+a*x**m)**q` on `[0,infinity)`
requires both `p>-1` and `m*q>p+1`. The exact moment is
`c**((p+1)/m-q)*beta((p+1)/m,q-(p+1)/m)/(m*a**((p+1)/m))`.
A finite polynomial numerator is summed term by term after checking every
monomial's tail threshold. The denominator is positive throughout the domain;
zero constants, unproved signs, real poles, and equality thresholds decline.

```python
import sympy as sp
from multiple_integrate import integrate_result
x = sp.Symbol("x", real=True)
r = integrate_result(sp.sqrt(x)/(1+x**3)**2, (x, 0, sp.oo))
assert sp.simplify(sp.expand_func(r.value.rewrite(sp.gamma))-sp.pi/6) == 0
assert r.absolute_convergent is True
```

## Comparison proofs without exact evaluation

Polynomial amplitudes times `exp(-Q)` are absolutely integrable on full real
space when `Q` has a strictly positive pure even-power term in every coordinate
and all other nonconstant terms are nonnegative even monomials. Discarding mixed
terms leaves separable stretched-exponential decay dominating every polynomial
moment. Missing confinement, negative remainder coefficients, or odd powers
must decline this sufficient proof.

Positive even-polynomial denominators have a related comparison: keep a positive
constant and a proved positive highest pure even-power term in every coordinate,
and discard other nonnegative terms. The existing anisotropic theorem checks
each numerator monomial's weighted tail order. For example, the growth of
`1+x**2+x**6` gives a sharper tail comparison than replacing it by a quadratic.
These proofs also propagate through the existing finite sums, bounded multipliers
and absolute-product rules. Positive-orthant rational tails can use the same
whole-space majorant.

```python
import sympy as sp
from multiple_integrate.convergence_evidence import absolute_rule
x,y = sp.symbols("x y", real=True)
f = (1+x*x+y**4)*sp.exp(-x**4-y**6-x*x*y*y)
claim,_ = absolute_rule(f, ((x,-sp.oo,sp.oo),(y,-sp.oo,sp.oo)))
assert claim is True
claim,_ = absolute_rule((1+x*x+x**6)**sp.Rational(-1,3), ((x,-sp.oo,sp.oo),))
assert claim is True
```

A proved zero amplitude supplies a cheap absolute-convergence rule even on an
improper domain. External parameter poles must still be excluded. Elementary
power, exponential and Gaussian divergence theorems retain explicit zero and
nonzero amplitude cells, including numeric exponent/rate cases. Structured
values agree with these cases rather than retaining an upstream infinity-times-zero
expression.

An existence certificate is separate from an evaluated answer. A coupled
exponential can have a proved finite integral while its exact value remains
unknown. Numerical quadrature in tests is an independent consistency check,
not the source of the convergence proof.

Regression coverage: `tests/test_selected_families.py`,
`tests/test_divergence_parameter_domains.py`, `tests/test_exact_graph_reversal.py`.

## Shifted and truncated stretched-exponential moments

For a real finite center `c`, a phase exactly equal to `b-a*(x-c)**m`
uses the same gamma moments after translation. On `c <= lo < hi <= infinity`,
polynomial amplitudes are translated exactly and upper incomplete gamma
functions subtract the two endpoint tails. The rate must satisfy `Re(a)>0`;
this sufficient prerequisite also applies to finite truncations. Full-line
moments require even `m` and polynomial amplitudes. Intervals crossing the
center and unrelated polynomial phases decline this rule. Real-power amplitudes
are supported when translation leaves one recognized power of `x-c`.

```python
import sympy as sp
from multiple_integrate import integrate_result
x = sp.Symbol("x", real=True)
r = integrate_result((1+x)*sp.exp(-(x-2)**4), (x, 2, sp.oo))
expected = (3*sp.gamma(sp.Rational(1,4)) + sp.sqrt(sp.pi))/4
assert sp.simplify(r.value-expected) == 0
```

Gamma is nonzero wherever its finite parameter domain is proved. Exponentials
of finite arguments and real hyperbolic cosines are nonzero. A principal
logarithm is nonzero when both its argument and its difference from one are
proved nonzero. These facts prune impossible amplitude-zero cells; gamma poles,
`log(0)`, and `log(1)=0` remain distinct exceptional cases.

Regression coverage: `tests/test_moment_extensions.py` checks exact endpoint
orders at three scales, broken thresholds, shifted finite and infinite ranges,
reversal, whole-line moments, and special-function parameter domains.

Further supported cases and prerequisites are described in [the extension guide](nonlinear-moments-and-singularities.md). Regression coverage: `tests/test_truncated_nonlinear_horns.py`.
