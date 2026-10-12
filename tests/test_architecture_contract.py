from pathlib import Path

from multiple_integrate.ownership import MODULE_OWNERSHIP, PACKAGE_BOUNDARIES
from multiple_integrate.planner import STRATEGY_OWNERS


def test_every_production_module_has_an_architectural_owner():
    root = Path(__file__).parents[1] / "src" / "multiple_integrate"
    modules = set()
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).with_suffix("")
        parts = rel.parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = "multiple_integrate" + ("." + ".".join(parts) if parts else ".__init__")
        if name == "multiple_integrate.__init__":
            pass
        modules.add(name)
    declared = set(MODULE_OWNERSHIP)
    assert modules == declared


def test_cross_package_boundaries_are_explicit():
    assert PACKAGE_BOUNDARIES == {
        "integration_strategy": "integrate",
        "function_properties": "funcprops",
        "semialgebraic_geometry": "semialg",
        "asymptotic_integration": "asymptotic",
    }


def test_every_planner_strategy_has_an_owner():
    assert STRATEGY_OWNERS
    assert all(owner for owner in STRATEGY_OWNERS.values())


def test_large_modules_are_split_by_responsibility():
    import ast

    root = Path(__file__).parents[1] / "src" / "multiple_integrate"
    core_defs = {
        node.name
        for node in ast.parse((root / "core.py").read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert "_normalize_assumptions" not in core_defs
    assert "_decompose_integrand" not in core_defs
    assert "CoordinateTransform" not in core_defs
    assert "_try_additive_separable" not in core_defs

    region_nodes = ast.parse((root / "regions.py").read_text()).body
    assert not any(isinstance(node, ast.ClassDef) for node in region_nodes)


def test_internal_modules_follow_dependency_direction():
    import ast

    root = Path(__file__).parents[1] / "src" / "multiple_integrate"
    forbidden = {
        "assumptions.py": {"multiple_integrate.core", "multiple_integrate.analytic_strategies"},
        "recognition.py": {"multiple_integrate.core", "multiple_integrate.analytic_strategies"},
        "region_types.py": {"multiple_integrate.core", "multiple_integrate.region_recognition"},
    }
    for filename, banned in forbidden.items():
        tree = ast.parse((root / filename).read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        assert imports.isdisjoint(banned), (filename, imports & banned)


def test_public_exports_have_docstrings():
    import inspect

    import multiple_integrate as package

    missing = []
    for name in package.__all__:
        obj = getattr(package, name)
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and not inspect.getdoc(obj):
            missing.append(name)
    assert missing == []


def test_repository_text_is_coherent():
    root = Path(__file__).parents[1]
    checked = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
    checked += sorted((root / "docs").rglob("*.rst"))
    checked += sorted((root / "src" / "multiple_integrate").rglob("*.py"))
    checked += sorted((root / "tests").rglob("*.py"))

    violations = []
    current_test = Path(__file__).resolve()
    for path in checked:
        if path.resolve() == current_test:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.rstrip() != line:
                violations.append(f"{path.relative_to(root)}:{number}: trailing whitespace")
    assert violations == []


def test_test_names_are_concise():
    import ast

    root = Path(__file__).parents[1] / "tests"
    long_names = []
    for path in root.rglob("test_*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
                and len(node.name) > 75
            ):
                long_names.append((path.name, node.name))
    assert long_names == []


def test_readme_documentation_links_are_absolute():
    import re

    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text()
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    documentation_links = [link for link in links if "docs" in link.lower()]
    assert documentation_links
    assert all(link.startswith("https://") for link in documentation_links)


def test_production_code_uses_specific_failure_paths():
    import ast

    root = Path(__file__).parents[1] / "src" / "multiple_integrate"
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                violations.append((path.name, node.lineno, "assert"))
            elif isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    violations.append((path.name, node.lineno, "bare except"))
                elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    violations.append((path.name, node.lineno, "except Exception"))
    assert violations == []


def test_production_modules_do_not_override_top_level_definitions():
    import ast
    from collections import Counter

    root = Path(__file__).parents[1] / "src" / "multiple_integrate"
    duplicates = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        names = Counter(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        duplicates.extend((path.name, name) for name, count in names.items() if count > 1)
    assert duplicates == []


def _module_name(root, path):
    rel = path.relative_to(root).with_suffix("")
    parts = rel.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return "multiple_integrate" + ("." + ".".join(parts) if parts else ".__init__")


def _imports_from(tree):
    import ast

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.append(node.module)
    return imports


def test_semialg_imports_are_confined_to_boundary_modules():
    import ast

    root = Path(__file__).parents[1] / "src" / "multiple_integrate"
    allowed = {
        "multiple_integrate.adapters.semialg",
        "multiple_integrate.semialg_geometry",
    }
    violations = []
    for path in root.rglob("*.py"):
        module = _module_name(root, path)
        tree = ast.parse(path.read_text())
        semialg_imports = [
            name for name in _imports_from(tree) if name == "semialg" or name.startswith("semialg.")
        ]
        if semialg_imports and module not in allowed:
            violations.append((module, semialg_imports))
    assert violations == []


def test_semialg_boundary_uses_only_package_public_imports():
    import ast

    root = Path(__file__).parents[1] / "src" / "multiple_integrate"
    violations = []
    for relative in ("adapters/semialg.py", "semialg_geometry.py"):
        path = root / relative
        tree = ast.parse(path.read_text())
        imports = [name for name in _imports_from(tree) if name.startswith("semialg.")]
        unsupported = [
            name
            for name in imports
            if name not in {"semialg.parametric_geometry", "semialg.chart_certificates"}
        ]
        if unsupported:
            violations.append((relative, unsupported))
    assert violations == []


def test_geometry_services_are_not_reimplemented_on_domains():
    import ast

    root = Path(__file__).parents[1] / "src" / "multiple_integrate"
    tree = ast.parse((root / "region_types.py").read_text())
    forbidden_methods = {
        "as_semialg_geometry",
        "as_semialgebraic_region",
        "intrinsic_parametric_cover",
        "measure",
        "centroid",
        "sample_point",
        "sample_points",
        "random_point",
        "random_points",
    }
    violations = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {
            item.name
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        overlap = sorted(methods & forbidden_methods)
        if overlap:
            violations.append((node.name, overlap))
    assert violations == []


def test_package_root_does_not_export_geometry_constructors():
    import multiple_integrate as package

    geometry_names = {
        "Region",
        "BoxRegion",
        "SimplexRegion",
        "AffineSimplexRegion",
        "DiskRegion",
        "AnnulusRegion",
        "BallRegion",
        "EllipsoidRegion",
        "SphericalShellRegion",
    }
    assert geometry_names.isdisjoint(package.__all__)
    assert all(not hasattr(package, name) for name in geometry_names)


def test_regions_facade_exports_operations_not_geometry_models():
    import multiple_integrate.regions as regions

    geometry_names = {
        "Region",
        "BoxRegion",
        "SimplexRegion",
        "AffineSimplexRegion",
        "DiskRegion",
        "AnnulusRegion",
        "BallRegion",
        "EllipsoidRegion",
        "SphericalShellRegion",
    }
    assert geometry_names.isdisjoint(regions.__all__)


def test_geometry_lowering_is_owned_by_semialg_geometry_module():
    import ast

    root = Path(__file__).parents[1] / "src" / "multiple_integrate"
    allowed = {
        "multiple_integrate.semialg_geometry",
        "multiple_integrate.adapters.semialg",
    }
    method_names = {"as_semialgebraic_region", "intrinsic_parametric_cover"}
    violations = []
    for path in root.rglob("*.py"):
        module = _module_name(root, path)
        if module in allowed:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in method_names:
                violations.append((module, node.lineno, node.attr))
    assert violations == []
