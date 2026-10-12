# Structured results and certificates

`integrate()` is the convenient flat API. Use `integrate_result()` when you need
to know **why** a value is valid, conditional, unavailable, or expensive.

## Flat versus structured

```python
import sympy as sp
from multiple_integrate import integrate, integrate_result

x, a = sp.symbols("x a", real=True)

flat = integrate(sp.exp(-a*x), (x, 0, sp.oo))
result = integrate_result(sp.exp(-a*x), (x, 0, sp.oo))

assert result.value == flat
print(result.status)
print(result.parameter_strata)
print(result.explain())
```

The structured form adds status, theorem/strategy name, conditions, parameter strata,
singular-locus strata, attempted strategies, limitations, and diagnostics.

## Parameter strata

A symbolic parameter can split one integral into mathematically different cells. Rather than hide
that information inside a single expression, `parameter_strata` records mutually exclusive
conditions with their value and convergence status. Supplying sufficient assumptions can collapse
a conditional problem to one exact cell.

## Strategy budgets

When an exact CAD, hypersurface, or asymptotic calculation exceeds the configured
`IntegrationPolicy`, the flat API raises `StrategyBudgetExceeded`. The structured API instead
returns `IntegrationStatus.UNEVALUATED` and records the rejected strategy and limit. This makes
resource policy distinguishable from mathematical nonexistence.

For explicit ranges, ``result.value`` is an ordinary unevaluated ``sympy.Integral``. For a
geometric or semialgebraic ``region=`` request, it is an ``UnevaluatedRegionIntegral``. That
placeholder retains the region formula, bound variables, and requested measure dimension, so an
intrinsic surface integral is not misrepresented as an ordinary integral over bare coordinates.

## Singular strata

For algebraic singularities the structured result may include decomposed singular components and
their intersections. These records are diagnostics for understanding which singular geometry a
regularization or convergence decision had to handle.

## Regularization certificates

`regularized_result()` is the corresponding structured API for PV and finite-part
calculations. Call `verify_regularization_result()` when you want to replay the stored cutoff and
constant-term claims independently of planner history.

Use structured results in applications that log symbolic decisions, expose diagnostics to users,
or need stable release-contract tests. Use the flat API when only the final SymPy expression is
needed.

## Exact series results

A certified finite radial series can be exact without evaluating its infinite sum to a finite closed form. Its infinity index is not a divergent value. An arbitrary unresolved symbolic sum is unknown. Inspect convergence evidence and active parameter assumptions separately from expression exactness; see [the radial guide](geometry/radial-series.md).
