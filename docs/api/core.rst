Core integration API
====================

Most users should call :func:`multiple_integrate.integrate` for a flat SymPy result or
:func:`multiple_integrate.integrate_result` for status, conditions, and planner provenance.

.. autofunction:: multiple_integrate.integrate
   :no-index:

.. autofunction:: multiple_integrate.integrate_result
   :no-index:

.. autofunction:: multiple_integrate.regularized_result
   :no-index:

Internal engine
---------------

The following module documentation is primarily for contributors working on strategy orchestration.
Application code should prefer the package-level functions above.

.. automodule:: multiple_integrate.core
   :members:
   :show-inheritance:
   :member-order: bysource
