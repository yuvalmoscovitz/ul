from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ul_cli.dataset import app as dataset_app
from ul_cli.dataset_regression import app as regression_app
from ul_cli.event_stress import app as stress_app

app = typer.Typer(
    name="ul",
    help="Discover consequential failures in high-risk AI agents.",
    no_args_is_help=True,
)
augmentations_app = typer.Typer(help="Inspect UL's built-in augmentation library.")
app.add_typer(augmentations_app, name="augmentations")
app.add_typer(dataset_app, name="dataset")
app.add_typer(regression_app, name="regression")
app.add_typer(stress_app, name="stress")

console = Console()


@augmentations_app.command("list")
def list_augmentations() -> None:
    from ul_core.augmentation import builtin_augmentation_registry

    registry = builtin_augmentation_registry()
    table = Table(title="Built-in plausible-behavior augmentations")
    table.add_column("ID", style="cyan")
    table.add_column("Version")
    table.add_column("Category")
    table.add_column("Description")
    for augmentation in registry.list():
        table.add_row(
            augmentation.metadata.id,
            augmentation.metadata.version,
            str(augmentation.metadata.category),
            augmentation.metadata.summary,
        )
    console.print(table)


if __name__ == "__main__":
    app()
