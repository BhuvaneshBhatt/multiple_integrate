Installation
============

From PyPI
---------

After a release is published, install MultipleIntegrate with:

.. code-block:: console

   python -m pip install multiple-integrate

The distribution name contains a hyphen, while the Python import package uses
an underscore:

.. code-block:: python

   import multiple_integrate

Development installation
------------------------

Clone the repository and install it in editable mode with the test dependencies:

.. code-block:: console

   python -m pip install -e ".[test]"

For documentation work, install the documentation extra:

.. code-block:: console

   python -m pip install -e ".[docs]"

Then build the documentation locally:

.. code-block:: console

   sphinx-build -W -b html docs docs/_build/html

The ``-W`` option turns Sphinx warnings into build failures. This mirrors the
strict Read the Docs configuration and catches broken cross-references or
autodoc failures before they reach the hosted site.
