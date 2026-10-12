# Bounded evaluation of entire polynomial-phase integrals

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

`multiple_integrate.series_evaluation.evaluate_entire_box` evaluates the supported density family `P(x)*exp(Q(x))`, `P(x)*sin(Q(x))` or `P(x)*cos(Q(x))` on an independent, strictly ordered finite real box. It avoids asking SymPy to evaluate a multi-index infinite sum. The function lives on a specialist module; the package root remains focused on integration and result inspection.

```python
import sympy as sp
from multiple_integrate.series_evaluation import evaluate_entire_box
x, y = sp.symbols("x y", real=True)
r = evaluate_entire_box(sp.exp(x*y), (x, 0, 1), (y, 0, 1),
                        tolerance=sp.Rational(1, 10**20))
assert r.converged and r.certified
assert r.error_bound <= sp.Rational(1, 10**20)
assert r.terms < 40
assert not r.value.has(sp.Sum, sp.Float)
approximation = r.value.evalf(30)
```

The returned `SeriesApproximation.value` is an **exact finite partial sum**. `error_bound` bounds its difference from the integral, before floating-point rounding. `converged` means that this bound meets `tolerance`; `certified` refers to this analytic bound, not to the rounding error of `.evalf()`. A term or sparse-expansion budget exhaustion returns the current partial sum, a valid bound and `converged=False`. This is different from an unknown exact integration result. Unsupported prerequisites raise `NotImplementedError`; malformed options raise `ValueError`.

## Error theorem

Write `Q=c+q`, with zero constant term in `q`. On the box, let `M` be an integer upper bound on `sum(abs(q_m)*prod(R_i**m_i))`, where `R_i=max(abs(lo_i),abs(hi_i))`. The implementation also bounds the absolute amplitude and volume by integers `A` and `V`, and `abs(c)` by an integer `C`.

After `n` Taylor terms, the integral error is bounded by

`V*A*3**(C+M)*M**n/n!`.

The inequality `e<3` makes the bound rational and makes its comparison with an exact positive rational tolerance exact. The same bound applies to complex polynomial phases and all three entire functions: their derivatives have magnitude at most `exp(abs(c)+M)` on the relevant line segment. Polynomial moments of each finite term are computed exactly. No numerical quadrature supplies the certificate.

Default budgets are 128 Taylor terms, 4,096 estimated sparse products per step and 1,024 for the majorant exponent. The input amplitude/phase also obey the active polynomial and dimension budgets. The majorant can be conservative; `converged=False` does not mean the integral diverges. The exponent budget declines excessively large majorants before constructing a huge integer power.

Resolve parameters with exact `substitutions=`. Floating input coefficients/endpoints and unresolved parameters are refused, because the implementation does not enclose input or arithmetic rounding errors. Use exact rational or algebraic input when requesting this bound. Reversed or degenerate boxes are rejected by this specialist evaluator; the integration API retains its oriented-range behavior.

Regression coverage: `tests/test_series_evaluation.py`.
