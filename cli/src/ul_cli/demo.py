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
    """See UL's model-free augment-and-compare workflow."""
    from ul_cli.demo_runner import run_demo as execute_demo

    execute_demo(output=output)
