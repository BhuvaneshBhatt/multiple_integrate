"""Execute demo code in a fresh Jupyter kernel with cell and whole-run budgets."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_worker(args):
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    notebook = nbformat.read(args.notebook, as_version=4)
    selected = []
    skipped = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            selected.append(cell)
            continue
        tags = cell.metadata.get("tags", [])
        if args.section == "geometry" and "geometry-casebook" not in tags:
            skipped.append({"id": cell.id, "reason": "outside selected geometry section"})
            continue
        if "manual-expensive" in tags:
            reason = cell.metadata.get("multiple_integrate_skip_reason")
            if not reason:
                raise ValueError("Every manual cell needs an explicit skip reason")
            skipped.append({"id": cell.id, "reason": reason})
            continue
        cell.outputs = []
        cell.execution_count = None
        selected.append(cell)
    notebook.cells = selected
    started = time.monotonic()
    deadline = started + args.total_timeout

    def timeout(cell):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Whole-notebook budget expired")
        return max(1, min(args.cell_timeout, int(remaining)))

    with tempfile.TemporaryDirectory(prefix="mi-notebook-") as directory:
        base = Path(directory)
        spec = base / "kernels" / "mi-demo"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "MultipleIntegrate validation",
                    "language": "python",
                }
            )
        )
        manager = KernelManager(
            kernel_name="mi-demo",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(base / "kernels")]),
        )
        client = NotebookClient(
            notebook,
            kernel_name="mi-demo",
            timeout=args.cell_timeout,
            timeout_func=timeout,
            allow_errors=False,
        )

        client.km = manager

        def interrupt(signum, frame):
            raise KeyboardInterrupt("Whole-notebook budget expired")

        signal.signal(signal.SIGTERM, interrupt)
        environment = {
            **os.environ,
            "IPYTHONDIR": str(base / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(base / "runtime"),
        }
        if environment.get("PYTHONPATH"):
            environment["PYTHONPATH"] = os.pathsep.join(
                str(Path(p).resolve()) for p in environment["PYTHONPATH"].split(os.pathsep) if p
            )
        client.execute(cwd=str(ROOT), env=environment)
    elapsed = time.monotonic() - started
    if elapsed > args.total_timeout:
        raise TimeoutError("Whole-notebook budget exceeded")
    nbformat.write(notebook, args.output)
    report = {
        "section": args.section,
        "executed_code_cells": sum(c.cell_type == "code" for c in notebook.cells),
        "skipped": skipped,
        "seconds": elapsed,
        "cell_timeout_seconds": args.cell_timeout,
        "total_timeout_seconds": args.total_timeout,
        "python": sys.executable,
    }
    Path(args.output).with_suffix(".execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook", type=Path, default=ROOT / "notebooks/multiple_integration.ipynb"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--section", choices=("geometry", "supported"), default="geometry")
    parser.add_argument("--cell-timeout", type=int, default=60)
    parser.add_argument("--total-timeout", type=int, default=240)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.cell_timeout, args.total_timeout) <= 0:
        parser.error("Budgets must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        run_worker(args)
        return
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--notebook",
        str(args.notebook.resolve()),
        "--output",
        str(args.output.resolve()),
        "--section",
        args.section,
        "--cell-timeout",
        str(args.cell_timeout),
        "--total-timeout",
        str(args.total_timeout),
    ]
    process = subprocess.Popen(command)
    try:
        result = process.wait(timeout=args.total_timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise SystemExit("Whole-notebook execution timed out") from None
    if result:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
