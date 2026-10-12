"""Exact fast paths and automatic topology cells; run from an installed package."""

import sympy as sp

from multiple_integrate import integrate_cells, integrate_result

x, y, a = sp.symbols("x y a", real=True)
gaussian = integrate_result(x**4 * sp.exp(-3 * (x - 1) ** 2), (x, 0, 1))
assert gaussian.absolute_convergent is True
oscillatory = integrate_result(sp.sin(1 / x) / x, (x, 0, 1))
assert oscillatory.value == sp.pi / 2 - sp.Si(1)
assert oscillatory.absolute_convergent is False
nonseparable = integrate_result(sp.exp(x * y), (x, 0, 1), (y, 0, 1))
assert nonseparable.value.has(sp.Sum) and nonseparable.complete
cells = integrate_cells(1, parameters=(a,), region=x**2 <= a, variables=(x,), bounds={x: (-1, 1)})
assert cells.complete and cells.value.subs(a, sp.Rational(1, 4)) == 1
print(gaussian.value, oscillatory.value, nonseparable.value, cells.value, sep="\n")
