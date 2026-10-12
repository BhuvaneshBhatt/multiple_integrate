# Worked failures and conservative outcomes

A failed exact route is useful information. Keep mathematical divergence,
unsupported geometry, unresolved certification, and resource refusal separate.

* `integrate_result(1/x, (x,0,1))` is `DIVERGENT`. A boundary pole does not
  acquire a finite value from antiderivative subtraction.
* `integrate_result(1/x, (x,-1,1))` is divergent, whereas the same request with
  `principal_value=True` has value zero and status `PRINCIPAL_VALUE`.
* A budget with `StrategyBudget(max_variables=0)` refuses a one-variable
  semialgebraic reduction. The structured result is `UNEVALUATED` and retains
  the domain; this is not an empty-region answer.
* A region defined by `sin(x)>=0` is not a general supported semialgebraic
  input. A reducer may return `None`; an intrinsic request can raise
  `NotImplementedError`. Use certified explicit intervals if appropriate.
* The fiber `a*x=0` is a point for nonzero `a` and a line at `a=0`. A finite-point
  argument valid generically cannot be applied at zero. Choose the measure and
  certify each fiber separately.
* The change `u=x²` is not globally invertible on `[-1,1]`. Split into monotone
  branches and include both Jacobian contributions. Integrating over one branch
  is not the integral over the original domain.
* `sqrt(x)` on `[-1,0]` has principal complex value `2*I/3`; `log(x)` there has
  integral `-1+I*pi`. A real-power identity cannot remove these branches.
* If Semialg cannot verify a required intrinsic cover, the adapter returns
  `None`. Do not interpret this as zero or drop a target-dimensional singular
  stratum. Unsupported intrinsic geometry may raise `NotImplementedError`.

Structured results distinguish finite, divergent, conditional, unsupported, and
unknown outcomes through `IntegrationOutcome`. Flat evaluation may still raise
`NotImplementedError` for unsupported operations. Resource refusal stays unknown;
a finite PV or finite part does not imply ordinary convergence.

Regression coverage: `tests/test_convergence_semantic_corpus.py`,
`tests/test_intrinsic_semialgebraic.py`, `tests/test_geometry_reference_corpus.py`.

## Geometry extension failures

A weak graph slope can vanish, so it cannot be divided by uniformly. A weak cusp contact cell with s=1 includes a divergent curved boundary even if the vertex is integrable. A Gaussian matrix can lose rank on its parameter boundary. Intrinsic point and curve measures change when branches collapse. A polygon outside the proved downward-closed envelope family can decline; that does not establish divergence. An unresolved series is unknown, while a certified radial series is finite. Each failure is paired with valid neighboring cases in `tests/test_serious_boundary_regressions.py` and `tests/test_geometry_casebook_properties.py`.

## Parameter poles and complex Gaussian branches

A cell with `a >= 0` does not prove that the polynomial `a*x` has degree one:
at `a = 0`, the condition `a*x >= 0` holds throughout the domain. On `[-1,1]`,
the ambient integral of one is two at zero and one at positive `a`. Supply the
separate cells `Eq(a,0)` and `a > 0`; a weak cell without a stable leading
coefficient remains unknown.

Parameter-dependent densities need their function domains proved even when the
integration variable does not occur in the density. For example, `tan(a)` has
poles where `cos(a)=0`, `atanh(a)` at `a=1` and `a=-1`, `zeta(a)` at `a=1`, and
`Ei(a)` at zero. A weak cell containing one of these poles cannot receive an
unconditional finite certificate. This also applies to counting measure on
moving points such as `Eq(x,a)` with density `1/a`. The guards cover selected
recognized functions; they are not a universal special-function domain prover.
Checks retain the original density and bounds: symmetry simplifying `x/a` to
zero does not remove the undefined coefficient at `a=0`. Coordinate singularities
still use the integration and convergence routes; `1/sqrt(x)` on `[0,1]` remains
finite. Parameter-only Piecewise conditions supply the scope for each density.

For complex Gaussian coefficients, `sqrt(det(A))` need not equal the analytic
product of the individual square-root branches. The full-space shortcut uses
the usual determinant formula for real positive-definite forms and independent
right-half-plane branches for diagonal complex forms. Coupled complex forms
remain with the fallback until their analytic branch can be certified.

Regression coverage: `tests/test_gaussian_branches_and_parameter_domains.py`.

## Positive quartics and undefined parameter specializations

A generic backend can return a spurious zero for `1/(1+(x*x+2*x+b)**2)` over the real line with real `b`. The density is strictly positive and has quadratic-or-better decay. The shifted-biquadratic rule supplies a proved positive exact moment instead, under its explicit coefficient prerequisites. Non-biquadratic or pole-bearing denominators decline that rule; its proof cannot be reused for an unsupported neighbor. Regression coverage: `tests/test_quartic_moments.py`.

Specializing `1/a` at `a=0` produces an undefined density, not a theorem of ordinary divergence. Parameter-cell orchestration retains an unknown request on that fiber and preserves valid sibling cells. An explicitly supplied removable extension is evaluated; a surviving coordinate pole is still classified by ordinary integration. Regression coverage: `tests/test_parameter_specialization_safety.py`.

## Reversed intervals and geometric boxes

Explicit integration limits carry orientation: `integrate(1, (x, 0, -2))`
returns `-2`. A geometric `BoxRegion(((x, 0, -2),))` instead denotes the
unsigned interval between its endpoints and has volume `2`. Constant
strategies previously multiplied an already signed width by orientation
a second time; this is corrected. Geometric boxes normalize their endpoints
consistently for constant and nonconstant densities. Endpoints that cannot
be proved real retain an unknown geometric request rather than silently
becoming a complex contour. Regression coverage: `test_constant_orientation.py`.

## Parameter domains of divergence theorems

`x**p/b` on `[0,1]` with `p<-1` cannot have a global divergence certificate while
`b=0` remains allowed: the density is undefined there. Previously the divergent
branch was checked less strictly than the finite branch. Both now require their
source parameter domain. Supplying `b != 0` retains the valid divergence claim.
Exponential phases containing `1/b` have the same requirement. Backend divergence
results, including numeric-power densities without a registered symbolic case,
now undergo the same source-domain check and retain an unevaluated request when
the parameter domain is not established.

## Exact graph reversal at small scales and crossing bounds

A triangle with `0<=y<=x` and `0<=x<=10**(-20)` has area `10**(-40)/2`.
Tolerance-based boundary comparisons previously replaced it by a rectangle of
twice that area. Reversal now uses exact signs and order comparisons. It also
declines crossing limits: `y` from `x` to `1-x`, with `x` from zero to one,
has signed cancellations and cannot be replaced by only its positive triangle.
Regression coverage: `tests/test_exact_graph_reversal.py`.

## Zero amplitudes in divergent families

`b/x**2` near zero and `b*exp(x)` or `b*exp(x**2)` on improper intervals
are zero densities at `b=0`, although their nonzero-amplitude branches diverge.
The elementary theorem registry now preserves that zero fiber even with numeric
exponents/rates. Structured values are reconstructed from their proved cases,
so evaluating the zero fiber cannot produce an infinity-times-zero artifact.
Other fallback divergence results require a proved nonzero independent amplitude;
unproved cases retain an unknown request. Zero-amplitude absolute proofs still
check the external parameter domain to avoid erasing undefined coefficients.
