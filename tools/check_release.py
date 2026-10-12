"""Validate release identity, dependency floors and source/distribution parity."""

from __future__ import annotations

import argparse
import email
import tarfile
import tomllib
from pathlib import Path
from zipfile import ZipFile


def main(root: Path, distributions: Path, tag: str | None = None) -> None:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if tag is not None and tag != "v" + version:
        raise ValueError(f"tag {tag!r} does not match project version {version}")
    assert f"## {version}" in (root / "CHANGELOG.md").read_text()
    wheel = distributions / f"multiple_integrate-{version}-py3-none-any.whl"
    source = distributions / f"multiple_integrate-{version}.tar.gz"
    with ZipFile(wheel) as archive:
        metadata = email.message_from_bytes(
            archive.read(f"multiple_integrate-{version}.dist-info/METADATA")
        )
        assert metadata["Version"] == version and metadata["Requires-Python"] == ">=3.11"
        dependencies = metadata.get_all("Requires-Dist")
        for requirement in ("semialg>=1.3.2", "funcprops>=0.3.0"):
            assert requirement in dependencies
        assert any(r.startswith("asymptotic>=0.2.1;") for r in dependencies)
        modules = list((root / "src/multiple_integrate").rglob("*.py"))
        for module in modules:
            assert (
                archive.read(module.relative_to(root / "src").as_posix()) == module.read_bytes()
            ), module
    with tarfile.open(source) as archive:
        for relative in (
            "README.md",
            "CHANGELOG.md",
            "LICENSE",
            "pyproject.toml",
            "tools/check_release.py",
            "docs/nonseparable-improper.md",
            "tests/test_nonseparable_improper.py",
        ):
            item = archive.extractfile(f"multiple_integrate-{version}/{relative}")
            assert item is not None and item.read() == (root / relative).read_bytes(), relative
    print(
        f"Release {version}: metadata, dependency floors, {len(modules)} production modules and source payload verified"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--tag")
    args = parser.parse_args()
    main(args.root, args.dist, args.tag)
