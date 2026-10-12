Region examples
===============

These examples use canonical ``semialg`` regions because the domain is useful independently of the
chosen integration coordinates. Use nested ranges instead when the iterated order is part of the
problem.

Boxes
-----

.. code-block:: python

   import semialg
   import sympy as sp
   from multiple_integrate import integrate

   x, y = sp.symbols("x y", real=True)
   box = semialg.Box(((0, 2), (-1, 3)))

   assert box.measure() == 8
   assert box.centroid() == (1, 1)
   assert integrate(x + y, region=box, variables=(x, y)) == 16

Balls and disks
---------------

A two-dimensional ``Ball`` is a disk:

.. code-block:: python

   disk = semialg.Ball((1, -2), 3)

   assert disk.measure() == 9 * sp.pi
   assert disk.centroid() == (1, -2)
   assert integrate(1, region=disk, variables=(x, y)) == 9 * sp.pi

Simplexes
---------

.. code-block:: python

   triangle = semialg.Simplex(((0, 0), (1, 0), (0, 1)))

   assert triangle.measure() == sp.Rational(1, 2)
   assert triangle.centroid() == (sp.Rational(1, 3), sp.Rational(1, 3))
   assert integrate(x + y, region=triangle, variables=(x, y)) == sp.Rational(1, 3)

Ellipsoids
----------

The shape matrix below gives semi-axis lengths 2 and 3:

.. code-block:: python

   ellipse = semialg.Ellipsoid((0, 0), ((4, 0), (0, 9)))

   assert ellipse.measure() == 6 * sp.pi
   assert integrate(1, region=ellipse, variables=(x, y)) == 6 * sp.pi

Nested ranges
-------------

For a triangle written as an iterated integral, the dependent inner range comes first:

.. code-block:: python

   result = integrate(
       x + y,
       (y, 0, 1 - x),
       (x, 0, 1),
   )
   assert result == sp.Rational(1, 3)

   from multiple_integrate import region_from_ranges

   domain = region_from_ranges([(y, 0, 1 - x), (x, 0, 1)])
   assert domain.constant_volume() == sp.Rational(1, 2)

Intrinsic measure
-----------------

A circle has zero ambient area but nonzero intrinsic length:

.. code-block:: python

   circle = semialg.Sphere((0, 0), 2)

   assert integrate(1, region=circle, variables=(x, y)) == 0
   assert integrate(
       1,
       region=circle,
       variables=(x, y),
       measure_dimension="intrinsic",
   ) == 4 * sp.pi

Sampling the same geometry
--------------------------

Sampling is provided by semialg, so no conversion is needed before integrating the region:

.. code-block:: python

   samples = disk.random_points(20, seed=42)
   exact_second_moment = integrate(x**2 + y**2, region=disk, variables=(x, y))

See :doc:`../concepts/sampling` for the distinction between representative and random samples.
