MultipleIntegrate
=================

MultipleIntegrate evaluates definite multiple integrals with SymPy, proved
geometric reductions, exact moment rules and selected regularization prescriptions.

Start with installation and the quick start. Use the domain guides to specify
regions and parameter cells, and the result guides to interpret convergence,
coverage and unresolved outcomes. The mathematical family guides give formulas
and prerequisites; the API reference provides signatures. Contributor material
is collected at the end.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Domains and parameter families

   concepts/ranges-and-regions
   concepts/measure-and-dimension
   supported-regions
   decomposition
   parameterized-integration
   geometry/parameter-geometry

.. toctree::
   :maxdepth: 2
   :caption: Results, convergence and regularization

   structured-results
   certification
   scoped-convergence
   regularization
   worked-failures
   concepts/sampling

.. toctree::
   :maxdepth: 2
   :caption: Exact methods and mathematical families

   theory
   method-references
   method-examples
   strategies
   performance-methods
   boundary-reduction
   coarea
   selected-symbolic-families
   proved-families
   series-evaluation
   nonseparable-improper
   geometry/cusp-proofs
   geometry/radial-series
   geometry/clipped-cutoffs
   laurent-clips-and-tangencies
   sectors-and-contact-charts
   nonlinear-moments-and-singularities

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/regions
   examples
   geometry-casebook

.. toctree::
   :maxdepth: 2
   :caption: Capabilities and limits

   capabilities
   limitations
   integration-extensions
   geometry-extensions

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index
   api

.. toctree::
   :maxdepth: 2
   :caption: Contributor guides

   architecture/semialg-integration
   testing
   contributing
