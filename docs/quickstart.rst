Quick Start
===========

Create real SymPy symbols and import the primary integration function plus the
region classifier:

.. code-block:: python

   import sympy as sp
   from multiple_integrate import multiple_integrate, region_from_ranges

   x, y, z = sp.symbols("x y z", real=True)

One-dimensional integration
---------------------------

``multiple_integrate`` also handles one-dimensional definite integrals:

.. code-block:: python

   result = multiple_integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo))
   assert result == sp.sqrt(sp.pi)

A rectangle
-----------

Ranges follow the same **inner-first** convention as ``sympy.integrate``.
For a rectangular domain, constant independent bounds make either integration
order mathematically equivalent:

.. code-block:: python

   result = multiple_integrate(
       x**2 * y**3,
       (x, 0, 1),
       (y, 0, 1),
   )
   assert result == sp.Rational(1, 12)

A triangular domain
-------------------

When a bound depends on another integration variable, order matters. The
dependent inner range comes first:

.. code-block:: python

   ranges = [
       (y, 0, 1 - x),
       (x, 0, 1),
   ]

   result = multiple_integrate(1, *ranges)
   assert result == sp.Rational(1, 2)

The same ranges can be classified geometrically:

.. code-block:: python

   region = region_from_ranges(ranges)

   assert type(region).__name__ == "SimplexRegion"
   assert region.constant_volume() == sp.Rational(1, 2)

A disk
------

Structured regions can trigger exact geometric shortcuts:

.. code-block:: python

   disk_ranges = [
       (y, -sp.sqrt(4 - x**2), sp.sqrt(4 - x**2)),
       (x, -2, 2),
   ]

   region = region_from_ranges(disk_ranges)

   assert type(region).__name__ == "DiskRegion"
   assert region.constant_volume() == 4 * sp.pi
   assert multiple_integrate(1, *disk_ranges) == 4 * sp.pi

Where to go next
----------------

Read :doc:`concepts/ranges-and-regions` before constructing nonrectangular
domains. Then see :doc:`examples/regions` for the principal named region
families. Complete call signatures and source docstrings are available in
:doc:`api/index`.
