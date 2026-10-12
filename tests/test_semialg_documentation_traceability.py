"""Keep user-visible claims linked to runnable regression tests."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    "architecture/semialg-integration.md",
    "supported-regions.md",
    "certification.md",
    "parameterized-integration.md",
    "worked-failures.md",
)
TRACE = {
    "architecture/semialg-integration.md": (
        "test_geometry_reference_corpus.py",
        "test_geometry_unknown_is_not_empty",
    ),
    "supported-regions.md": (
        "test_geometry_reference_corpus.py",
        "test_intrinsic_surface_vs_ambient_and_explicit_chart",
    ),
    "certification.md": (
        "test_convergence_semantic_corpus.py",
        "test_budget_unknown_is_not_divergence",
    ),
    "parameterized-integration.md": (
        "test_geometry_reference_corpus.py",
        "test_parameter_transition_cover",
    ),
    "worked-failures.md": (
        "test_geometry_reference_corpus.py",
        "test_noninvertible_change_requires_both_branches",
    ),
}


@pytest.mark.parametrize("page", PAGES)
def test_documented_claim_has_regression(page):
    content = (ROOT / "docs" / page).read_text()
    filename, function = TRACE[page]
    assert filename in content
    tree = ast.parse((ROOT / "tests" / filename).read_text())
    assert function in {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for reference in re.findall(r"`(tests/[^`]+\.py)`", content):
        assert (ROOT / reference).is_file(), reference
    assert page.removesuffix(".md") in (ROOT / "docs/index.rst").read_text()


def test_parameterized_guide_example_executes():
    content = (ROOT / "docs/parameterized-integration.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", content, re.S)
    assert blocks
    for block in blocks:
        exec(compile(block, "parameterized-integration.md", "exec"), {})


def test_current_semialg_api_in_documentation():
    for path in [
        ROOT / "README.md",
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "docs").rglob("*.rst"),
    ]:
        content = path.read_text()
        assert not re.search(r"semialg\.(BoxRegion|PointRegion|SphericalShellRegion)\b", content), (
            path
        )
        assert not re.search(r"from semialg import[^\n]*intrinsic_parametric_cover", content), path
