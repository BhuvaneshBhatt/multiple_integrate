"""Keep mathematical attribution aligned with the production ownership contract."""

import json
import re
from pathlib import Path

from multiple_integrate.ownership import MODULE_OWNERSHIP

ROOT = Path(__file__).parents[1]


def test_every_owned_module_has_explicit_method_or_infrastructure_attribution():
    coverage = json.loads((ROOT / "docs/method-reference-coverage.json").read_text())
    text = (ROOT / "docs/method-references.md").read_text()
    rows = {}
    for line in text.splitlines():
        if line.startswith("| ") and "`" in line and not line.startswith("| Method |"):
            method, modules, foundations = line.strip("|").split("|")
            rows[method.strip()] = (re.findall(r"`([^`]+)`", modules), foundations)
    assert set(coverage) == set(MODULE_OWNERSHIP)
    for name, entry in coverage.items():
        if "infrastructure" in entry:
            assert entry == {"infrastructure": MODULE_OWNERSHIP[name]}
        else:
            assert set(entry) == {"methods"} and entry["methods"]
            for method in entry["methods"]:
                modules, foundations = rows[method]
                assert name.removeprefix("multiple_integrate.") in modules
                assert re.search(r"\[[A-Z]+\]", foundations)
    for method, (modules, _) in rows.items():
        for module in modules:
            assert method in coverage[f"multiple_integrate.{module}"]["methods"]


def test_reference_keys_resolve_and_guides_link_to_the_canonical_reference_page():
    text = (ROOT / "docs/method-references.md").read_text()
    used = set(re.findall(r"\[([A-Z]+)\](?!:)", text))
    defined = set(re.findall(r"^\[([A-Z]+)\]: https://", text, re.M))
    assert used and used == defined
    assert "method-references" in (ROOT / "docs/index.rst").read_text()
    assert "method-references.md" in (ROOT / "README.md").read_text()
    for page in ("theory.md", "strategies.md", "regularization.md", "nonseparable-improper.md"):
        assert "method-references.md" in (ROOT / "docs" / page).read_text()
