# Executable geometry casebook

`examples/geometry_casebook.py` is the source for the README and the demo's geometry section. Run it with `python examples/geometry_casebook.py` after installing MultipleIntegrate. It asserts exact values, distinguishes divergence from a finite regularized value, compares the radial series with independent quadrature, and rejects a tampered finite part. The first summation bound is a representation detail, not a claim of divergence.

Shared imports:

```python
from dataclasses import replace
import mpmath as mp
import semialg
import sympy as sp
from multiple_integrate import (
    HadamardFinitePart, IntegrationOutcome, ParameterCell, integrate_cells,
    integrate_result, regularized_result, verify_regularization_result,
)
```

See [parameter geometry](geometry/parameter-geometry.md), [cusp proofs](geometry/cusp-proofs.md), [radial series](geometry/radial-series.md), and [clipped cutoffs](geometry/clipped-cutoffs.md) for prerequisites, derivations and counterexamples. The [supported-region matrix](supported-regions.md) distinguishes measure conventions.

## What the evidence means

Inspect an ordinary result's `outcome`, `complete`, `coverage` and `convergence_evidence`. For a parameter family inspect each entry in `diagnostics['cell_results']`: the parent can be conditional even when every supplied cell is resolved. Conditional parameter-space coverage is scoped to the supplied assumptions. Finite, divergent, conditional, unsupported and unknown are distinct outcomes. A prescribed principal value or finite part is not an ordinary convergence certificate. Replaying a prescription is not independent mathematical verification.

## Synchronization and notebook execution

Run `python tools/sync_geometry_examples.py --check` to detect snippet drift; rerun without `--check` after editing the source examples. The notebook source stores no executed output, so stale output cannot masquerade as a validated result. Clean-kernel executed notebooks and machine-readable execution reports are separate artifacts.

Install the `notebook-test` extra and run `python tools/run_notebook.py --section geometry --output /tmp/geometry-demo.ipynb --cell-timeout 60 --total-timeout 240`. The geometry section is self-contained. To audit all supported demo cells use `--section supported --total-timeout 600`. Three manual timing/showcase cells have stable IDs and explicit reasons in notebook metadata; they are recorded individually in the execution report. No other code cell is silently skipped. Per-cell and whole-notebook budgets fail the gate rather than converting errors into successful output.

Claims and executable references: `tests/test_casebook_and_docs.py` checks all four public examples, named claims, snippet execution and notebook synchronization. `tests/test_geometry_casebook_properties.py` supplies independent geometry, convergence, numerical and asymptotic oracles. `tests/test_serious_boundary_regressions.py` guards the finite-series protocol and exceptional parameter fibers.
