from __future__ import annotations

import typer

from ul_cli.augmentations import app as augmentations_app
from ul_cli.dataset import app as dataset_app
from ul_cli.dataset_regression import app as regression_app
from ul_cli.event_stress import app as stress_app
from ul_cli.project import initialize_project, report_project, run_project
from ul_cli.sandbox import app as sandbox_app

app = typer.Typer(
    name="ul",
    help="Discover consequential failures in high-risk AI agents.",
    no_args_is_help=True,
)
app.add_typer(augmentations_app, name="augmentations")
app.add_typer(dataset_app, name="dataset")
app.add_typer(regression_app, name="regression")
app.add_typer(sandbox_app, name="sandbox")
app.add_typer(stress_app, name="stress")
app.command("init")(initialize_project)
app.command("run")(run_project)
app.command("report")(report_project)

if __name__ == "__main__":
    app()
