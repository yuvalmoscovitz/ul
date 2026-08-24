from __future__ import annotations

import os
from pathlib import Path

from examples.probe_qualification.receipt import append_private_receipt


def invoke(value: object) -> dict[str, object]:
    receipt_path = os.environ.get("UL_QUALIFICATION_RECEIPT")
    if receipt_path:
        append_private_receipt(Path(receipt_path), value)
    return {"status": "open", "ticket": 42}
