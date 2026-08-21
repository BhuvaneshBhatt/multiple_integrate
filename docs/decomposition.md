# Integrand composition analysis

MultipleIntegrate analyzes some integrands as a composition

$$
F(x_1,\ldots,x_n)=f(g(x_1,\ldots,x_n)).
$$

The private `_IntegrandDecomposition` container stores the outer callable `f_outer`, the inner expression `g_inner`, and whether `g_inner` is polynomial in the integration variables. This representation is used only by the planner.

## Recognized structures

### Polynomial inner expressions

A polynomial integrand can use the identity outer function, with the polynomial itself as `g_inner`. This representation enables linear pushforward, quadratic full-space, simplex, and polynomial level-set formulas.

### Single-argument functions

Expressions such as `exp(g)`, `sin(g)`, `cos(g)`, and `log(g)` are represented by replacing the function argument with a fresh dummy variable. For example, `exp(x**2 + y)` becomes an outer function `exp(t)` and inner expression `x**2 + y`.

### Powers with integration-independent exponents

An expression such as `(x**2 + 1)**(3/2)` is represented by the outer function `t**(3/2)` and inner expression `x**2 + 1`.

### Constant factor and addend peeling

Integration-independent factors and addends are separated before recursively analyzing the variable-dependent expression. For example, `3*sin(x)` can be represented with inner expression `x` and outer function `3*sin(t)`.

### Single active variables

Any expression depending on exactly one integration variable can use the identity outer function. This supports monotone and piecewise-monotone change-of-variable methods for expressions such as `tan(x)`, `1/x`, and `x*log(x)`.

## When composition analysis declines

A multivariable expression that has no supported compositional structure is left to separability checks, region-specific formulas, or ordinary iterated integration. Declining a decomposition is intentional: the planner should not force an artificial `f(g)` representation when doing so would obscure variable dependence or make a structural method unsound.

## Polynomial flag

The `is_polynomial` flag determines whether polynomial-only formulas are eligible. Non-polynomial inner expressions skip coefficient extraction and polynomial geometry tests, then proceed directly to methods that support general symbolic inner functions.
