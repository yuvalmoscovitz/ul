from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def run_demo(
    output: Annotated[
        Path | None,
        typer.Option(help="New private evidence file; defaults to UL's private data directory."),
    ] = None,
) -> None:
    """Run a model-free demonstration against a synthetic local agent."""
    from ul_cli.demo_runner import run_demo as execute_demo

    execute_demo(output=output)
