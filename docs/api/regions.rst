Regions and domain input
========================

Why use a region?
-----------------

Use a canonical ``semialg`` region when the domain is a reusable mathematical object rather than
just a set of nested integration bounds. The same object can provide membership, dimension,
measure, centroid, sampling, transformations, and an integration domain.

.. code-block:: python

   import semialg
   import sympy as sp
   from multiple_integrate import integrate

   x, y = sp.symbols("x y", real=True)
   disk = semialg.Ball((0, 0), 2)

   assert disk.measure() == 4 * sp.pi
   assert disk.centroid() == (0, 0)
   assert integrate(x**2 + y**2, region=disk, variables=(x, y)) == 8 * sp.pi

Use nested ranges when the integration order itself matters. ``region_from_ranges`` recognizes
useful structure in those ranges for planner dispatch, but its returned domain model is not a second
canonical geometry API.

Accepted domain forms
---------------------

``integrate`` supports three main domain forms:

* inner-first ``(variable, lower, upper)`` ranges;
* canonical ``semialg`` geometry supplied with ``region=`` and ambient ``variables=``;
* a semialgebraic Boolean formula supplied with ``region=`` and ``variables=``.

See :doc:`../concepts/ranges-and-regions` for choosing among them and
:doc:`../concepts/measure-and-dimension` for lower-dimensional geometry.

Range recognition
-----------------

.. autofunction:: multiple_integrate.region_from_ranges
   :no-index:

Boolean restrictions
--------------------

.. autofunction:: multiple_integrate.Boole
   :no-index:

Ownership
---------

``semialg`` owns canonical geometry, formula lowering, intrinsic charts, measure, centroid, and
sampling. MultipleIntegrate owns integration planning, convergence, exact integration strategies,
and regularization. New geometry operations should therefore be added to semialg and consumed here
through the public semialg boundary.
