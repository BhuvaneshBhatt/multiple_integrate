Nested ranges and region classification
=======================================

The range convention
--------------------

MultipleIntegrate uses the same iterated-integral ordering as
``sympy.integrate``: the **first range is integrated first**, so it is the
innermost integral.

For example,

.. math::

   \int_0^1 \int_0^{1-x} f(x,y)\,dy\,dx

is written as:

.. code-block:: python

   multiple_integrate(
       f,
       (y, 0, 1 - x),
       (x, 0, 1),
   )

The upper bound of ``y`` may depend on ``x`` because ``x`` remains free while
the inner ``y`` integral is evaluated. Reversing those tuples without deriving
new bounds generally describes a different iterated integral.

Ranges as geometry
------------------

A list of nested ranges describes more than an evaluation order: it also
describes a geometric domain. ``region_from_ranges`` analyzes that structure
and returns the most specific region type it can justify.

Conceptually, classification proceeds from informative geometric patterns
toward a generic fallback:

.. code-block:: text

   nested ranges
       |
       +-- independent constant bounds --------> BoxRegion
       |
       +-- standard simplex bounds ------------> SimplexRegion
       |
       +-- affine simplex bounds --------------> AffineSimplexRegion
       |
       +-- circular / spherical bounds --------> DiskRegion / BallRegion
       |
       +-- unequal quadratic semi-axes --------> EllipsoidRegion
       |
       +-- dependent graph bounds -------------> GraphRegion
       |
       `-- otherwise --------------------------> IteratedRegion

Classification is deliberately conservative. A specialized region should be
returned only when the bounds establish the corresponding geometry. This
matters because specialized objects can provide exact volume, moment, radial,
symmetry, or change-of-variables shortcuts.

Structural orientation versus public ordering
----------------------------------------------

Geometric recognition is often easiest when bounds are inspected
outer-to-inner, whereas the public API follows SymPy's inner-first convention.
The classifier may therefore inspect plausible structural orientations
internally. Once a region is recognized, its stored ranges are canonicalized
back to the package's public inner-first convention.

Users should continue to supply ranges exactly as they would to
``sympy.integrate``.

Why classification matters
--------------------------

Region recognition can replace a difficult symbolic iterated integral with an
exact formula. Specialized paths include:

* constant volume for boxes, disks, balls, ellipsoids, annuli, spherical
  shells, and simplices;
* polynomial moments on several structured regions;
* cancellation of odd terms on reflection-invariant domains;
* polar or spherical changes of variables for suitable radial integrands;
* affine normalization of an affine simplex to a standard simplex;
* selected graph-region order reversal.

If no specialized path applies, ``multiple_integrate`` retains a general
iterated SymPy integration fallback. Region recognition is therefore an
optimization and reasoning layer rather than a requirement for every valid
integral.

Inspecting a classification
---------------------------

.. code-block:: python

   import sympy as sp
   from multiple_integrate import region_from_ranges

   x, y = sp.symbols("x y", real=True)

   ranges = [
       (y, 0, 1 - x),
       (x, 0, 1),
   ]

   region = region_from_ranges(ranges)

   print(type(region).__name__)
   print(region.variables)
   print(region.normalized_ranges())
   print(region.constant_volume())

The region classes and classification functions are documented in
:doc:`../api/regions`.

Translated quadrics
-------------------

Disk, ball, and axis-aligned ellipsoid recognition also accepts coordinate shifts.
For example, bounds equivalent to ``(x-cx)**2 + (y-cy)**2 <= R**2`` produce a
``DiskRegion`` with a nonzero ``center``.  The center tuple follows the same variable
ordering as ``region.variables``. Polynomial moments and radial shortcuts translate to
the stored center before applying the standard centered formulas.

Union semantics
---------------

``UnionRegion`` represents an additive disjoint decomposition, not a general
inclusion-exclusion set union. Pieces must use the same ordered variables and be
disjoint up to measure-zero boundaries. The constructor rejects overlaps it can prove
for supported concentric radial pieces; when symbolic overlap is undecidable, the
caller is responsible for supplying disjoint pieces.
