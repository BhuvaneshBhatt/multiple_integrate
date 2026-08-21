# Testing

Run the complete suite from the project root with:

```bash
pytest
```

For a source checkout without an editable installation:

```bash
PYTHONPATH=src pytest
```

## Test organization

Region classification and geometry are covered by focused modules for boxes, simplices, graph regions, disks, balls, translated regions, advanced radial shapes, classification boundaries, range ordering, invariants, and method matrices.

Integration methods are covered by tests for simplex/Dirichlet formulas, coordinate changes, supported symbolic families, assumptions, conditional convergence, singular and divergent cases, Boolean/Piecewise integrands, caching, timeouts, and performance guards.

Reference examples are maintained in `test_reference_examples.py` and `test_reference_examples_notebook.py`. Broad end-to-end behavior is exercised in `test_multiple_integrate.py`.

## Writing tests

For a mathematical capability, include:

- a direct exact success case;
- a structurally similar case where the specialized method must decline;
- invalid-input cases when the API has structural preconditions;
- exact symbolic comparisons whenever practical;
- a performance guard only when runtime is part of the behavior being protected.

Region methods should be exercised through both direct region objects and the top-level `multiple_integrate` dispatcher when both paths are supported.
