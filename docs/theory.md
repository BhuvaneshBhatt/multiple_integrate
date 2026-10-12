# Mathematical foundations

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

This doc page collects the identities and hypotheses used by MultipleIntegrate. Planner behavior is
in [How the planner chooses](strategies.md), API contracts are in [Public API](api.md).

## Iterated integrals and orientation

For nested ranges, MultipleIntegrate follows SymPy's inner-to-outer convention (not the structural outer-to-inner ordering used, for example, by Mathematica). A two-variable integral

\[
\int_a^b\int_{u(x)}^{v(x)} f(x,y)\,dy\,dx
\]

is written with the `y` range first. Bounds are oriented: reversing a finite bound changes the sign.
Geometric volume formulas are therefore valid replacements only when the orientation implied by
symbolic parameters is known.

## Product separation

On a Cartesian product `A × B`, if the integrand factors as `f(x)g(y)`, Fubini/Tonelli conditions
allow

\[
\int_{A\times B} f(x)g(y)\,d(x,y)
=\left(\int_A f(x)\,dx\right)\left(\int_B g(y)\,dy\right).
\]

If an integrand is independent of one coordinate, that coordinate contributes the corresponding
region measure.

## Simplex and Dirichlet integrals

On the standard simplex

\[
\Delta_n=\{x_i\ge0:\;x_1+\cdots+x_n\le1\},
\]

Dirichlet's formula gives

\[
\int_{\Delta_n}\prod_{i=1}^n x_i^{\alpha_i}
(1-\textstyle\sum_i x_i)^{\alpha_0}\,dx
=
\frac{\prod_{i=0}^n\Gamma(\alpha_i+1)}
{\Gamma(\sum_{i=0}^n\alpha_i+n+1)},
\]

provided `Re(alpha_i) > -1` for every exponent. Polynomial simplex moments are the nonnegative
integer special case.

## Affine change of variables

For an invertible affine map `x = A u + b`,

\[
\int_R F(x)\,dx
=|\det A|\int_{A^{-1}(R-b)}F(Au+b)\,du.
\]

This identity underlies affine-simplex and ellipsoidal normalization. A valid symbolic use requires an invertible map, a correctly transformed domain, and the appropriate absolute Jacobian for
ordinary measure.

## Polar and spherical coordinates

In two dimensions,

\[
x=r\cos\theta,\qquad y=r\sin\theta,\qquad dx\,dy=r\,dr\,d\theta.
\]

In three dimensions,

\[
x=\rho\sin\phi\cos\theta,\quad
y=\rho\sin\phi\sin\theta,\quad
z=\rho\cos\phi,
\]

with volume element `rho**2 sin(phi) d rho d phi d theta`. In dimension `n`, radial measure contains
the factor `r**(n-1)` times unit-sphere measure.

## Spherical tensor moments

Rotational invariance implies

\[
\int_{S^{n-1}}\omega_i\omega_j\,d\omega
=\frac{|S^{n-1}|}{n}\delta_{ij}.
\]

Higher even moments are sums over pairings of indices with the corresponding isotropic denominator
`n(n+2)...`; odd moments vanish. These contraction identities avoid unnecessary coordinate
expansion for polynomial angular factors.

## Gaussian integrals

For real `a > 0`,

\[
\int_{-\infty}^{\infty}e^{-a x^2}\,dx=\sqrt{\frac{\pi}{a}}.
\]

For a real symmetric positive-definite matrix `A`,

\[
\int_{\mathbb R^n}e^{-x^T A x}\,dx
=\frac{\pi^{n/2}}{\sqrt{\det A}}.
\]

Positive definiteness is a convergence hypothesis, not just a simplification assumption.

## Symmetry

If a domain is invariant under a reflection and the integrand is odd under the same reflection, the ordinary integral vanishes when it exists. For singular integrals, the cutoff prescription must
preserve the symmetry before the same cancellation can be used.

## Level-set reduction

For suitable `g`, a co-area/layer-cake viewpoint rewrites an integral of `F(g(x))` in terms of the
measure or density of level sets of `g`. In one transverse variable this reduces to an ordinary
substitution; for regular hypersurfaces the local Jacobian contains `1/|partial_i g|` on a chart
where `partial_i g != 0`.

## Principal values

A one-dimensional Cauchy principal value around an interior singularity `c` uses a symmetric cutoff:

\[
\operatorname{PV}\int_a^b f(x)\,dx
=\lim_{\varepsilon\to0^+}
\left(\int_a^{c-\varepsilon}f(x)\,dx+
\int_{c+\varepsilon}^b f(x)\,dx\right).
\]

A geometric principal value replaces the interval cutoff by one shrinking centered geometric
neighbourhood. A hypersurface PV uses one shared level-set cutoff `|g| > epsilon`. These are distinct
mathematical prescriptions and can give different answers.

## Hadamard finite part

When a cutoff integral has negative powers or logarithms of `epsilon`, the Hadamard finite part is
the coefficient of `epsilon**0 * log(epsilon)**0` after the divergent terms are separated. This
requires an asymptotic expansion with enough information to identify the constant term.

## Convergence and assumptions

Exact formulas are valid only under their theorem hypotheses. Common conditions include positive
Gaussian quadratic forms, endpoint exponents greater than `-1`, sufficient decay at infinity, and
nonvanishing Jacobians or transverse derivatives. When parameters leave these conditions undecided, the natural mathematical result is parameter-stratified or conditional rather than a unqualified formula.
