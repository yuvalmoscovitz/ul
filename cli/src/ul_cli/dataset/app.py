from __future__ import annotations

import typer

from ul_cli.dataset_ingest import app as ingest_app
from ul_cli.dataset_review import (
    report_dataset_evidence,
    review_dataset_finding,
    review_dataset_pattern,
)

from .environment.initialize import initialize_dataset_environment
from .evaluation.command import evaluate_dataset
from .evaluation.operators import list_dataset_operators

app = typer.Typer(help="Explore behavioral differences in observed agent interactions.")
app.add_typer(ingest_app, name="ingest")
app.command("report")(report_dataset_evidence)
app.command("review")(review_dataset_finding)
app.command("review-pattern")(review_dataset_pattern)
app.command("init")(initialize_dataset_environment)
app.command("operators")(list_dataset_operators)
app.command("evaluate")(evaluate_dataset)
