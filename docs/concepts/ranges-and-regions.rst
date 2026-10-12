Ranges and regions
==================

Why there are two domain forms
------------------------------

MultipleIntegrate accepts a domain in two complementary forms:

* **nested ranges** describe an iterated integral and make integration order explicit;
* **canonical semialg regions** describe geometry independently of any particular coordinate order.

Use ranges when the bounds are part of the problem. Use a ``semialg`` region when you want one
reusable geometric object for integration, membership, measure, centroid, sampling, or affine
transformation.

This distinction keeps geometry in ``semialg`` and integration strategy in MultipleIntegrate.
Range recognition inside MultipleIntegrate is planner data: it helps choose exact formulas without
creating a second public geometry API.

Range order
-----------

MultipleIntegrate follows the same inner-first convention as ``sympy.integrate``. For

.. math::

   \int_0^1 \int_0^{1-x} f(x,y)\,dy\,dx,

the dependent inner range comes first:

.. code-block:: python

   result = integrate(
       f,
       (y, 0, 1 - x),
       (x, 0, 1),
   )

Reordering dependent ranges without deriving new bounds changes the domain.

Recognizing structure in ranges
-------------------------------

``region_from_ranges`` classifies useful integration structure such as independent boxes,
simplexes, affine simplexes, radial quadrics, graph-bounded domains, and general iterated domains.
The returned object is useful when inspecting planner behavior or testing recognition:

.. code-block:: python

   import sympy as sp
   from multiple_integrate import region_from_ranges

   x, y = sp.symbols("x y", real=True)
   domain = region_from_ranges([
       (y, 0, 1 - x),
       (x, 0, 1),
   ])

   assert domain.variables == (y, x)
   assert domain.constant_volume() == sp.Rational(1, 2)

Most application code does not need to construct these internal domain models directly.

Using canonical geometry
------------------------

When the geometry is known, construct it with ``semialg`` and pass it through ``region=``:

.. code-block:: python

   import semialg
   import sympy as sp
   from multiple_integrate import integrate

   x, y = sp.symbols("x y", real=True)
   triangle = semialg.Simplex(((0, 0), (1, 0), (0, 1)))

   result = integrate(x + y, region=triangle, variables=(x, y))
   assert result == sp.Rational(1, 3)

The same ``triangle`` object can be used with semialg's geometry operations. MultipleIntegrate
uses public semialg lowering and intrinsic-chart APIs when those representations are needed for
integration.

Boolean semialgebraic domains
-----------------------------

A raw Boolean formula can also be passed as a region. Supply ``variables=`` so free parameters are
not mistaken for integration coordinates:

.. code-block:: python

   condition = sp.And(x >= 0, y >= 0, x + y <= 1)
   result = integrate(1, region=condition, variables=(x, y))
   assert result == sp.Rational(1, 2)

For reusable geometry, prefer a canonical semialg object when one exists. Formulas are most useful
for domains defined naturally by inequalities or Boolean combinations.

Disjoint decompositions
-----------------------

The planner can represent a finite disjoint decomposition as a sum of integration domains. This is
an integration device. General geometric union, intersection, difference, topology, and membership
belong to ``semialg``.

See also
--------

* :doc:`measure-and-dimension` for ambient versus intrinsic integration;
* :doc:`sampling` for representative and random points;
* :doc:`../examples/regions` for complete examples.
