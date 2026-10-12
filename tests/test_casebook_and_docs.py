"""Executable public examples, claim traceability and generated notebook integrity."""

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["parameter", "cusp", "radial", "cutoff"])
def test_casebook_public_examples(name):
    casebook = load(ROOT / "examples/geometry_casebook.py")
    result = casebook.EXAMPLES[name]()
    if name != "cutoff":
        assert result.complete


@pytest.mark.parametrize("name", ["parameter", "cusp", "radial", "cutoff"])
def test_shared_snippets_execute(name):
    namespace = vars(load(ROOT / "examples/geometry_casebook.py")).copy()
    synchronizer = load(ROOT / "tools/sync_geometry_examples.py")
    exec(compile(synchronizer.snippets()[name], f"casebook:{name}", "exec"), namespace)
    assert "result" in namespace


def test_readme_notebook_and_source_examples_are_synchronized():
    assert load(ROOT / "tools/sync_geometry_examples.py").generated(check=True) == []
    notebook = json.loads((ROOT / "notebooks/multiple_integration.ipynb").read_text())
    ids = [c["id"] for c in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))
            if "manual-expensive" in cell.get("metadata", {}).get("tags", []):
                assert cell["metadata"]["multiple_integrate_skip_reason"]
    assert (
        sum(
            "geometry-casebook" in c.get("metadata", {}).get("tags", [])
            and c["cell_type"] == "code"
            for c in notebook["cells"]
        )
        == 6
    )


@pytest.mark.parametrize(
    "page", ["parameter-geometry", "cusp-proofs", "radial-series", "clipped-cutoffs"]
)
def test_named_geometry_claims_reference_real_tests(page):
    text = (ROOT / "docs/geometry" / f"{page}.md").read_text()
    assert page in (ROOT / "docs/index.rst").read_text()
    references = re.findall(r"`(tests/[^`]+\.py)(?:::(\w+))?`", text)
    assert references
    for path, name in references:
        file = ROOT / path
        assert file.exists()
        if name:
            definitions = {
                n.name
                for n in ast.walk(ast.parse(file.read_text()))
                if isinstance(n, ast.FunctionDef)
            }
            assert name in definitions
    assert "literalinclude" in text and "geometry_casebook.py" in text


def test_diagram_assets_are_valid_and_accessible():
    import xml.etree.ElementTree as ET

    for path in (ROOT / "docs/_static").glob("*.svg"):
        image = ET.parse(path).getroot()
        assert image.attrib["role"] == "img" and image.attrib["aria-label"]


def test_notebook_gate_and_optional_dependencies_are_explicit():
    assert "notebook-test" in (ROOT / "pyproject.toml").read_text()
    assert "--total-timeout" in (ROOT / "tools/run_notebook.py").read_text()
    assert "run_notebook.py" in (ROOT / ".github/workflows/geometry-docs.yml").read_text()
