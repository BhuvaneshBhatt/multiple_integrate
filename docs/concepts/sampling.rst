Sampling regions
================

Why sample a region?
--------------------

Sampling is useful for inspecting a domain, checking membership-sensitive code, visualizing a
region, constructing numerical experiments, or validating an exact result independently. Sampling
is a geometry operation, so it belongs to ``semialg`` rather than MultipleIntegrate.

MultipleIntegrate accepts the same canonical semialg region for exact integration, which means one
object can be sampled and integrated without maintaining two domain descriptions.

Representative points
---------------------

``sample_point`` and ``sample_points`` find valid representative points. They are appropriate when
you need witnesses rather than a probability distribution:

.. code-block:: python

   import semialg

   disk = semialg.Ball((0, 0), 2)
   point = disk.sample_point()
   points = disk.sample_points(5)

For semialgebraic formulas, representative sampling can use exact witness machinery when supported.
Do not assume that representative points are uniformly distributed.

Random points
-------------

``random_point`` and ``random_points`` are distributional samplers:

.. code-block:: python

   point = disk.random_point(seed=42)
   points = disk.random_points(100, seed=42)

A fixed seed makes a run reproducible. For supported bounded canonical regions, uniform sampling is
with respect to the region's intrinsic measure. That distinction matters for balls, spheres,
shells, simplexes, and lower-dimensional geometry.

Sampling and exact integration together
---------------------------------------

The same region can be used by both packages:

.. code-block:: python

   import sympy as sp
   from multiple_integrate import integrate

   x, y = sp.symbols("x y", real=True)
   disk = semialg.Ball((0, 0), 2)

   samples = disk.random_points(100, seed=7)
   exact = integrate(x**2 + y**2, region=disk, variables=(x, y))

Sampling does not select or approximate MultipleIntegrate's symbolic strategy. It is a separate
geometry tool that can provide an independent numerical sanity check.

Unbounded and general formula regions
-------------------------------------

There is no canonical uniform probability distribution on an unbounded region. For general formula
regions, random rejection sampling requires finite bounds. Representative sampling remains useful
when a uniform distribution is undefined or unavailable.

For the exact sampling signatures and supported canonical families, use the semialg API reference.
