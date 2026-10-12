"""Execute new documented claims and enforce links to regression coverage."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "page",
    [
        "scoped-convergence.md",
        "series-evaluation.md",
        "selected-symbolic-families.md",
        "parameterized-integration.md",
        "regularization.md",
        "integration-extensions.md",
    ],
)
def test_convergence_documentation_executes_and_links_regressions(page):
    content = (ROOT / "docs" / page).read_text()
    for block in re.findall(r"```python\n(.*?)```", content, re.S):
        exec(compile(block, page, "exec"), {})
    references = re.findall(r"`(tests/[^`]+\.py)`", content)
    assert references
    for reference in references:
        path = ROOT / reference
        assert path.is_file()
        assert any(
            isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
            for n in ast.walk(ast.parse(path.read_text()))
        )
    assert page.removesuffix(".md") in (ROOT / "docs/index.rst").read_text()
