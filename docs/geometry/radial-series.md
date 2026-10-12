# Exact half-integer radial series

See the [method reference guide](../method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

Polynomial-weighted radial integration reduces supported balls, shells and quadratic coordinate maps to moments

`M = integral(r**(2*alpha-1)*exp(-a*r**2)*(b+r**2)**(-s), r=R0..R1)`.

The finite mixed route requires a finite proved interval, b>0, real finite profile parameters and positive supported radial index alpha. Positive half-integer indices up to 16 are included. Polynomial weights and dimension determine alpha. A quadratic map additionally needs a proved positive-definite form and its Jacobian.

## Series derivation and numerical use

On a finite interval, the exponential Taylor series converges absolutely and uniformly. Its absolute sum is bounded by `exp(abs(a)*R1**2)`, and the shifted-power weight is bounded for b>0. Dominated convergence therefore permits termwise integration. The kth endpoint moment is

`R**(2*(alpha+k))*hyper((s,alpha+k),(alpha+k+1,),-R**2/b)/(2*b**s*(alpha+k))`.

Subtract the inner endpoint, multiply by `(-a)**k/k!`, and sum k from zero to infinity. An unevaluated infinite `Sum` is an exact representation when accompanied by this convergence proof; infinity in its index bound does not imply a divergent value. An arbitrary unresolved series without that proof remains unknown. Exactness does not imply a finite closed form or guarantee that low-precision evaluation is well conditioned. Large `abs(a)*R1**2` can cause alternating cancellation: increase precision and compare independent radial quadrature.

## Executable public example

```{literalinclude} ../../examples/geometry_casebook.py
:start-after: example-start: radial
:end-before: example-end: radial
:dedent: 4
:language: python
```

The example uses 35-digit independent quadrature, a 30-digit symbolic evaluation and absolute tolerance 1e-25. Convergence proofs and numerical agreement are distinct evidence. Shifted-power tails at infinity must retain the lower endpoint. General infinite-radius mixed profiles beyond existing recognized families, invalid shifts and unsupported indices have no promise of a native series result.

Claims: `finite-series-protocol` → `tests/test_serious_boundary_regressions.py::test_finite_radial_series_is_not_divergent`; shells, weights, limits and precision → `tests/test_geometry_casebook_properties.py`; public example → `tests/test_casebook_and_docs.py`.
