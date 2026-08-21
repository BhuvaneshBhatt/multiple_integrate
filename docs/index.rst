MultipleIntegrate
=================

**MultipleIntegrate** is a symbolic multiple-integration library built on SymPy.
It combines ordinary iterated integration with recognition of structured
geometric regions, exact region formulas, symmetry, coordinate transformations,
and specialized integration strategies.

The documentation is organized in four layers:

1. :doc:`quickstart` — a short path from installation to useful integrals.
2. :doc:`concepts/ranges-and-regions` — how nested ranges encode iterated
   integrals and geometric regions.
3. :doc:`examples/regions` — checked examples for rectangles, graph regions,
   disks, balls, ellipsoids, and affine simplices.
4. :doc:`api/index` — API documentation generated from the package's source
   docstrings.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/ranges-and-regions
   theory
   strategies
   decomposition

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/regions
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   testing
   contributing
