Measure and dimension
=====================

Why dimension needs to be explicit
-----------------------------------

A curve in the plane has zero two-dimensional area but a nonzero length. A surface in three-space
has zero three-dimensional volume but a nonzero area. MultipleIntegrate therefore distinguishes the
ambient measure from the intrinsic measure of a canonical region.

Use ``measure_dimension="ambient"`` for ordinary volume in the coordinate space. Use
``measure_dimension="intrinsic"`` when the integral should use the region's own dimension and the
corresponding Hausdorff metric factor.

Full-dimensional regions
------------------------

For a full-dimensional ball, ambient and intrinsic dimensions agree:

.. code-block:: python

   import semialg
   import sympy as sp
   from multiple_integrate import integrate

   x, y = sp.symbols("x y", real=True)
   disk = semialg.Ball((0, 0), 2)

   assert disk.measure() == 4 * sp.pi
   assert integrate(1, region=disk, variables=(x, y)) == 4 * sp.pi

Lower-dimensional regions
-------------------------

A circle in the plane has intrinsic dimension one:

.. code-block:: python

   circle = semialg.Sphere((0, 0), 2)

   assert circle.measure() == 4 * sp.pi
   assert integrate(1, region=circle, variables=(x, y)) == 0
   assert integrate(
       1,
       region=circle,
       variables=(x, y),
       measure_dimension="intrinsic",
   ) == 4 * sp.pi

The ambient integral is zero because a one-dimensional set has zero two-dimensional Lebesgue
measure. The intrinsic integral is its circumference.

Metric factors
--------------

For intrinsic integration, semialg supplies verified parametric charts. MultipleIntegrate pulls the
integrand back to each chart and integrates it with the induced metric factor

.. math::

   \sqrt{\det(J^T J)},

where ``J`` is the chart Jacobian. This formula reduces to the ordinary absolute Jacobian
determinant for full-dimensional square coordinate changes.

Embedded simplexes and surfaces
-------------------------------

Intrinsic integration is useful for geometry embedded in a higher-dimensional ambient space. For
example, a triangle in three dimensions is integrated with surface area rather than three-dimensional
volume when ``measure_dimension="intrinsic"`` is requested.

Dimension errors
----------------

``variables=`` names the ambient coordinates of canonical geometry and Boolean formulas. It is not
valid alongside ordinary explicit range tuples. This prevents an explicit iterated integral from
being silently reinterpreted as a region problem.

Use :func:`semialg.region_measure` or a canonical region's ``measure()`` when you only need the
measure. Use MultipleIntegrate when the goal is an integral of a nonconstant expression or when you
need integration strategy, convergence, or regularization information.
