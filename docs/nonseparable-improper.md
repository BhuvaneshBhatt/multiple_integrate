# Nonseparable improper integration

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

Full-space profiles can be evaluated after a proved triangular polynomial shear.
For example, the change u=x, v=y-x² turns a coupled quartic profile into u⁴+v².
The chart is a global bijection of real space with determinant one. Positive even
powers and finite real chart coefficients are required. This preserves the complete
integration domain; the same substitution is not valid for a halfspace or box
without transforming its boundaries.

```python
import sympy as sp
from multiple_integrate import integrate_result
x,y=sp.symbols("x y",real=True)
q=x**4+(y-x*x)**2
result=integrate_result(1/(1+q)**2,(x,-sp.oo,sp.oo),(y,-sp.oo,sp.oo))
assert sp.simplify(result.value-sp.sqrt(sp.pi)*sp.gamma(sp.Rational(1,4))**2/8)==0
assert result.complete and result.absolute_convergent
```

The supported profile is a finite polynomial density times an exponential of the
weighted power sum, an inverse shifted-power factor, or both with the same power
sum. Translation is included. A positive shifted inverse power removes the origin
singularity. Mixed Gaussian and shifted inverse-power moments use gamma/Meijer-G
representations. Polynomial moment expansions obey the existing degree and term
budgets.

For a transformed monomial with powers p_i and power-sum degrees m_i, let
A=sum((p_i+1)/m_i). With zero shift and an additional radial power h, the origin
requires A+h>0. Without an exponential tail, infinity requires A+h<0. These strict
thresholds apply to every nonzero monomial, including odd terms whose signed
angular moment is zero. Odd cancellation is used only after absolute convergence
has been established. Unproved or violated prerequisites cause this rule to decline;
other strategies determine whether the overall outcome is divergent or unknown.

```python
import sympy as sp
from multiple_integrate import integrate_result
x,y=sp.symbols("x y",real=True)
q=x**4+(y-x*x)**2
result=integrate_result(sp.exp(-q)/sp.sqrt(q),(x,-sp.oo,sp.oo),(y,-sp.oo,sp.oo))
expected=sp.sqrt(sp.pi)*sp.gamma(sp.Rational(1,4))**2/(2*sp.gamma(sp.Rational(3,4)))
assert sp.simplify(result.value-expected)==0
assert result.absolute_convergent
```

This is an absolutely convergent improper integral, not a principal value or finite
part. Coverage evidence records the global chart and full orthant decomposition;
convergence evidence records the origin/tail argument. Coefficient poles require
explicit exclusion or separate parameter cells. The recognizer tries the supplied
coordinate order and its reverse. General nontriangular charts, oscillatory coupled
tails, mismatched profiles and arbitrary singular domains remain unresolved.

Regression and independent Laplace-transform oracle coverage:
`tests/test_nonseparable_improper.py`.
