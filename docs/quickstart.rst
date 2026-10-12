Quick Start
===========

MultipleIntegrate is most useful when a definite integral has exploitable structure: repeated
product ranges, a recognizable region, conditional convergence, or a nonordinary regularization
prescription. For a routine antiderivative, use ``sympy.integrate`` directly.

Start with real SymPy symbols and the two primary entry points:

.. code-block:: python

   import sympy as sp
   from multiple_integrate import integrate, integrate_result

   x, y = sp.symbols("x y", real=True)

A one-dimensional definite integral
-----------------------------------

``integrate`` also handles one-dimensional definite integrals:

.. code-block:: python

   result = integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo))
   assert result == sp.sqrt(sp.pi)

Product ranges
--------------

Ranges use the same inner-first convention as ``sympy.integrate``:

.. code-block:: python

   result = integrate(
       x**2 * y**3,
       (x, 0, 1),
       (y, 0, 1),
   )
   assert result == sp.Rational(1, 12)

Dependent ranges
----------------

When a bound depends on another integration variable, order matters. For

.. math::

   \int_0^1\int_0^{1-x} 1\,dy\,dx,

write the dependent inner range first:

.. code-block:: python

   triangle_ranges = [
       (y, 0, 1 - x),
       (x, 0, 1),
   ]

   assert integrate(1, *triangle_ranges) == sp.Rational(1, 2)

``region_from_ranges`` is useful when you want to inspect what structure the planner recognized:

.. code-block:: python

   from multiple_integrate import region_from_ranges

   domain = region_from_ranges(triangle_ranges)
   assert domain.constant_volume() == sp.Rational(1, 2)

Canonical geometry
------------------

When the domain itself is reusable, construct it with ``semialg`` and pass it through ``region=``:

.. code-block:: python

   import semialg

   disk = semialg.Ball((0, 0), 2)

   assert disk.measure() == 4 * sp.pi
   assert integrate(1, region=disk, variables=(x, y)) == 4 * sp.pi

The same object can be used for membership, centroid, sampling, transformations, and other geometry
operations in semialg.

Intrinsic integration
---------------------

For lower-dimensional geometry, request the region's own measure dimension:

.. code-block:: python

   circle = semialg.Sphere((0, 0), 2)

   assert integrate(1, region=circle, variables=(x, y)) == 0
   assert integrate(
       1,
       region=circle,
       variables=(x, y),
       measure_dimension="intrinsic",
   ) == 4 * sp.pi

Assumptions and structured results
----------------------------------

Use ``integrate_result`` when you need to distinguish an exact answer from a conditional theorem,
a policy refusal, divergence, or an unresolved result:

.. code-block:: python

   a = sp.symbols("a", real=True)
   result = integrate_result(sp.exp(-a * x), (x, 0, sp.oo))

   print(result.status)
   print(result.parameter_strata)
   print(result.explain())

With sufficient assumptions, the flat API returns the corresponding exact branch:

.. code-block:: python

   assert integrate(
       sp.exp(-a * x),
       (x, 0, sp.oo),
       assumptions=a > 0,
   ) == 1 / a

Where to go next
----------------

Read :doc:`concepts/ranges-and-regions` for domain input, :doc:`concepts/measure-and-dimension` for
curves and surfaces, and :doc:`concepts/sampling` when you want points from the same canonical
geometry. Use :doc:`strategies` to understand planner choices, :doc:`regularization` for principal
values and finite parts, and :doc:`structured-results` for provenance and diagnostics.
