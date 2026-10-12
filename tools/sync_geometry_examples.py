"""Synchronize public examples; --check fails if README or notebook copies drift."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("parameter", "cusp", "radial", "cutoff")
TITLES = (
    "Parameter cells and graph contact",
    "Cusp contact and divergence",
    "Exact finite radial series",
    "Clipped polygon finite part",
)


def snippets():
    source = (ROOT / "examples/geometry_casebook.py").read_text()
    return {
        name: textwrap.dedent(
            re.search(
                rf"    # example-start: {name}\n(.*?)    # example-end: {name}", source, re.S
            )[1]
        ).rstrip()
        + "\n"
        for name in (*NAMES, "evidence")
    }


def generated(check=False):
    code = snippets()
    readme = (ROOT / "README.md").read_text()
    blocks = []
    for name, title in zip(NAMES, TITLES, strict=True):
        blocks.append(f"### {title}\n\n```python\n{code[name]}```\n")
    section = (
        "\n## Geometry casebook\n\nThe examples share these imports:\n\n```python\nfrom dataclasses import replace\nimport mpmath as mp\nimport semialg\nimport sympy as sp\nfrom multiple_integrate import (HadamardFinitePart, IntegrationOutcome, ParameterCell,\n    integrate_cells, integrate_result, regularized_result, verify_regularization_result)\n```\n\n"
        + "\n".join(blocks)
        + "\nFor an ordinary result inspect `result.outcome`, `result.complete`, `result.coverage`, and `result.convergence_evidence`; for a family inspect each child in `result.diagnostics['cell_results']`. A finite-part answer describes its specified cutoff and does not prove ordinary convergence.\n"
    )
    marker = "<!-- geometry-casebook-start -->"
    end = "<!-- geometry-casebook-end -->"
    if marker in readme:
        readme = re.sub(
            re.escape(marker) + r".*?" + re.escape(end), marker + section + end, readme, flags=re.S
        )
    else:
        readme += "\n" + marker + section + end + "\n"
    path = ROOT / "notebooks/multiple_integration.ipynb"
    notebook = json.loads(path.read_text())
    notebook["cells"] = [
        c
        for c in notebook["cells"]
        if "geometry-casebook" not in c.get("metadata", {}).get("tags", [])
    ]

    def cell(kind, name, text):
        value = {
            "cell_type": kind,
            "id": "geometry-" + name,
            "metadata": {"tags": ["geometry-casebook"]},
            "source": text.splitlines(True),
        }
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        return value

    notebook["cells"].append(
        cell(
            "markdown",
            "intro",
            "## Proved geometry casebook\n\nThese examples are generated from `examples/geometry_casebook.py`. Run this section by itself in a fresh kernel. Region coverage, convergence and cutoff replay are different evidence; the result table keeps them separate. A convergent infinite `Sum` is an exact value.\n\n![Graph contact](../docs/_static/graph-contact.svg)\n![Cusp contact](../docs/_static/cusp-contact.svg)\n![Polygon cutoff](../docs/_static/polygon-cutoff.svg)\n",
        )
    )
    notebook["cells"].append(
        cell(
            "code",
            "setup",
            "from dataclasses import replace\nimport mpmath as mp\nimport semialg\nimport sympy as sp\nfrom multiple_integrate import (HadamardFinitePart, IntegrationOutcome, ParameterCell, integrate_cells, integrate_result, regularized_result, verify_regularization_result)\nfrom examples.geometry_casebook import summary, cusp_contact\n",
        )
    )
    for name, title in zip(NAMES, TITLES, strict=True):
        notebook["cells"].append(cell("markdown", name + "-title", "### " + title + "\n"))
        presentation = (
            "summary(result)\n"
            if name != "cutoff"
            else '{"value":str(result.value), "cutoff_replay":verify_regularization_result(result).valid}\n'
        )
        notebook["cells"].append(cell("code", name, code[name] + presentation))
    notebook["cells"].append(cell("code", "evidence", code["evidence"]))
    changes = []
    if readme != (ROOT / "README.md").read_text():
        changes.append(ROOT / "README.md")
    original = json.loads(path.read_text())
    if notebook != original:
        changes.append(path)
    if check and changes:
        raise SystemExit(
            "Out-of-date generated examples: "
            + ", ".join(str(p.relative_to(ROOT)) for p in changes)
        )
    if not check:
        (ROOT / "README.md").write_text(readme)
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return changes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated(args.check)
