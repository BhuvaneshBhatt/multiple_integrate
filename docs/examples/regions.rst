Worked region examples
======================

These examples emphasize both **classification** and **correctness of the
resulting integral**. All ranges use the public inner-first convention.

Rectangles and boxes
--------------------

A rectangle has independent constant bounds and is classified as
``BoxRegion``:

.. code-block:: python

   import sympy as sp
   from multiple_integrate import (
       BoxRegion,
       multiple_integrate,
       region_from_ranges,
   )

   x, y, z = sp.symbols("x y z", real=True)

   ranges = [
       (y, 0, 2),
       (x, -1, 1),
   ]

   region = region_from_ranges(ranges)

   assert isinstance(region, BoxRegion)
   assert region.constant_volume() == 4
   assert multiple_integrate(1, *ranges) == 4

A three-dimensional box works the same way: its volume is the product of its
side lengths.

Graph regions
-------------

A graph region has an outer constant interval and an inner interval bounded by
functions of the outer variable:

.. code-block:: python

   from multiple_integrate import GraphRegion

   ranges = [
       (y, x, 2*x + 1),
       (x, 0, 1),
   ]

   region = region_from_ranges(ranges)

   assert isinstance(region, GraphRegion)
   assert region.constant_volume() == sp.Rational(3, 2)
   assert multiple_integrate(1, *ranges) == sp.Rational(3, 2)

For supported affine graph boundaries, ``GraphRegion.reversed_pieces()`` can
derive a piecewise description with the integration order reversed. This can
be useful when one order exposes an easier antiderivative.

Disks
-----

The disk

.. math::

   x^2 + y^2 \le R^2

can be represented with Cartesian nested bounds:

.. code-block:: python

   from multiple_integrate import DiskRegion

   R = sp.Integer(2)

   ranges = [
       (y, -sp.sqrt(R**2 - x**2), sp.sqrt(R**2 - x**2)),
       (x, -R, R),
   ]

   region = region_from_ranges(ranges)

   assert isinstance(region, DiskRegion)
   assert region.radius == 2
   assert region.constant_volume() == 4 * sp.pi
   assert multiple_integrate(1, *ranges) == 4 * sp.pi

For radial or separable polar expressions, ``DiskRegion.radial_integral`` can
use polar coordinates. Polynomial moments use disk-specific exact formulas.

Balls
-----

A radius-2 three-dimensional ball is represented by three nested Cartesian
ranges:

.. code-block:: python

   from multiple_integrate import BallRegion

   R = sp.Integer(2)

   ranges = [
       (
           z,
           -sp.sqrt(R**2 - x**2 - y**2),
           sp.sqrt(R**2 - x**2 - y**2),
       ),
       (
           y,
           -sp.sqrt(R**2 - x**2),
           sp.sqrt(R**2 - x**2),
       ),
       (x, -R, R),
   ]

   region = region_from_ranges(ranges)

   assert isinstance(region, BallRegion)
   assert region.dimension == 3
   assert region.constant_volume() == sp.Rational(32, 3) * sp.pi
   assert multiple_integrate(1, *ranges) == sp.Rational(32, 3) * sp.pi

The same ``BallRegion`` abstraction also supports recognized higher-dimensional
standard balls.

Ellipsoids
----------

Unequal semi-axes distinguish an ellipsoid from a ball. In two dimensions this
is an ellipse, represented by ``EllipsoidRegion``:

.. code-block:: python

   from multiple_integrate import EllipsoidRegion

   ranges = [
       (
           y,
           -3 * sp.sqrt(1 - x**2 / 4),
           3 * sp.sqrt(1 - x**2 / 4),
       ),
       (x, -2, 2),
   ]

   region = region_from_ranges(ranges)

   assert isinstance(region, EllipsoidRegion)
   assert set(region.axes) == {sp.Integer(2), sp.Integer(3)}
   assert region.constant_volume() == 6 * sp.pi
   assert multiple_integrate(1, *ranges) == 6 * sp.pi

Ellipsoid shortcuts are obtained by scaling to a unit ball, which also supports
selected polynomial moments and radial-type expressions.

Affine simplices
----------------

The standard two-simplex is the triangle

.. math::

   u \ge 0,\qquad v \ge 0,\qquad u+v \le 1.

Now apply the affine map

.. math::

   x = 1 + 2u,\qquad y = 2 + 4v.

The resulting region has the inner-first bounds:

.. code-block:: python

   from multiple_integrate import AffineSimplexRegion

   ranges = [
       (y, 2, 8 - 2*x),
       (x, 1, 3),
   ]

   region = region_from_ranges(ranges)

   assert isinstance(region, AffineSimplexRegion)
   assert region.constant_volume() == 4
   assert multiple_integrate(1, *ranges) == 4

The affine map has scale factors 2 and 4, so its Jacobian magnitude is 8.
Multiplying by the standard triangle's area :math:`1/2` gives area 4.
Polynomial moments can then be computed by transforming the integrand back to a
standard simplex.

Classification fallback
-----------------------

Not every valid iterated domain belongs to a named family. Such domains remain
``IteratedRegion`` instances and can still be integrated by the general engine.
Code that inspects region types should therefore treat classification as
informative rather than assume every valid region receives a specialized
class.
