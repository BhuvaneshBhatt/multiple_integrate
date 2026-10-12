"""Every named method family has an executed, independently checked example."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CATALOGUE = runpy.run_path(str(ROOT / "examples/method_catalogue.py"))["METHOD_EXAMPLES"]


def test_method_reference_example_coverage_is_exhaustive():
    guide = (ROOT / "docs/method-references.md").read_text()
    methods = {
        line.split("|")[1].strip()
        for line in guide.splitlines()
        if line.startswith("| ") and "`" in line and not line.startswith("| Method |")
    }
    assert set(CATALOGUE) == methods
    manifest = json.loads((ROOT / "docs/method-example-coverage.json").read_text())
    assert manifest == {method: example.__name__ for method, example in CATALOGUE.items()}
    page = (ROOT / "docs/method-examples.md").read_text()
    for method, example in CATALOGUE.items():
        assert method in page and f"`{example.__name__}`" in page
    assert "method-examples" in (ROOT / "docs/index.rst").read_text()
    assert "method-examples.md" in (ROOT / "README.md").read_text()


@pytest.mark.parametrize("method", CATALOGUE)
def test_each_method_example_executes(method):
    CATALOGUE[method]()


def test_Boole_is_the_public_indicator_construct():
    import sympy as sp

    import multiple_integrate as mi

    x = sp.symbols("x", real=True)
    assert isinstance(mi.Boole(x > 0), sp.Piecewise)
    assert mi.Boole(x > 0) == sp.Piecewise((1, x > 0), (0, True))
    assert "Boole" in mi.__all__ and not hasattr(mi, "boole")
    assert mi.integrate(mi.Boole(x > 0), (x, -1, 1)) == 1
