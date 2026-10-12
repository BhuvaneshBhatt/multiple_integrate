# Contributing

This guide explains the project layout and the main ideas to keep in mind when extending MultipleIntegrate.

## Development setup

Use Python 3.11 or newer. From the repository root, create a virtual environment
and install the package with its development dependencies:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the complete suite and static checks before submitting a change:

```sh
pytest
ruff check .
ruff format --check .
```

The complete suite includes slow and generated tests. To run the slow selection
separately, use `pytest -m slow`. See [Testing](testing.md) for mathematical
contracts, dependency gates and packaging checks.

---

## Repository layout

```text
src/multiple_integrate/
  __init__.py
  api/
  adapters/
  analytic_strategies.py
  assumptions.py
  conditional.py
  core.py
  ownership.py
  planner.py
  recognition.py
  region_formulas.py
  region_recognition.py
  region_types.py
  regions.py
  results.py
  transforms.py
tests/
notebooks/
  multiple_integration.ipynb
docs/
  index.rst
  api.md
  strategies.md
  theory.md
  decomposition.md
  examples.md
  testing.md
  contributing.md
```

---

## Architecture

The package uses the following layered flow:

```text
integrate(expr, *ranges)
│
├─ parse ranges using the SymPy convention
├─ classify the region if possible
├─ try exact structured region methods
├─ try symmetry / separability reductions
├─ try selected coordinate changes
├─ try selected decomposition-based methods
└─ fall back to sympy.integrate
```

Responsibilities are split by semantic ownership:

- `core.py` — integration orchestration and recursive execution
- `assumptions.py` — assumption normalization, querying, and refinement
- `recognition.py` — integrand and range recognition used by strategy guards
- `analytic_strategies.py` — exact analytic integration methods
- `transforms.py` — structured coordinate changes
- `region_types.py` — region value types and exact geometric formulas
- `region_recognition.py` — recognition of structured regions from ranges
- `region_formulas.py` — Boolean indicators and exact region restriction logic
- `regions.py` — stable region-facing import facade
- `adapters/` — narrow public-API boundaries to geometry, function-property and optional asymptotic packages
- `planner.py` and `results.py` — provenance and structured result semantics
- `conditional.py` — parameter-conditioned result normalization

---

## Range convention

Use the same public convention everywhere:

- range tuples are in **inner-first iterated order**, matching `sympy.integrate`

When writing tests, docs, examples, or new code paths, do not switch back to outer-first structural order at the public interface.

---

## Main extension areas

### 1. Region recognition

Canonical region types and certified geometric operations belong to Semialg. Add MultipleIntegrate recognition and integration strategies behind its public-API adapter; see the [package boundary](architecture/semialg-integration.md). Internal structured range models live in `region_types.py`, and range recognition lives in `region_recognition.py`.

Typical tasks:

- recognize a canonical Semialg region or an internal range model
- teach `region_from_ranges(...)` how to recognize it
- add any useful exact methods, such as `constant_volume`, `polynomial_moment`, or `radial_integral`

### 2. Exact structured families

If a class of integrals has a stable closed form, prefer implementing that directly rather than relying on generic SymPy definite integration.

Examples include:

- simplex / Dirichlet formulas
- selected disk / ball / shell formulas
- selected Gaussian structured formulas

### 3. Coordinate changes

Selected coordinate transforms are implemented in `transforms.py` using structured transform objects.

When extending this area, focus on transformations that are:

- mathematically safe
- easy to validate
- easy to map to a standard transformed region

### 4. Decomposition-based methods

`_IntegrandDecomposition` and its helpers support monotone, piecewise-monotone, separable, and selected layer-cake methods.

This is a useful extension area, but it is best treated as one family inside the broader solver.

---

## Adding a capability

A good pattern is:

1. decide whether the idea is primarily about **region structure**, **integrand structure**, or a **coordinate change**
2. add the implementation to the module that owns that responsibility
3. add focused tests in `tests/`
4. update docs if the capability changes public expectations

---

## Coding guidance

- prefer exact formulas for recognized families
- avoid unnecessary delegation to raw SymPy for structured definite integrals
- keep public examples in SymPy-style inner-first order
- add comments for genuinely tricky code paths, especially region reversal and coordinate splitting
- keep helper names descriptive and stable

---

## Documentation updates

Keep user workflows in the README and task guides; keep setup, architecture and
release procedures in contributor pages. Add a new page to the matching task
group in the documentation index rather than appending another general section.

When public behavior changes, update the relevant reference and guide pages, starting with:

- `docs/index.rst`
- `docs/api.md`
- `docs/strategies.md`
- `docs/theory.md`
- `docs/testing.md` if the test layout changed

Keep the README, notebook, and reference pages consistent with the same public API and ownership model.
