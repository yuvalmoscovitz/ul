from __future__ import annotations

import json
import os
from pathlib import Path


def invoke(value: object) -> dict[str, object]:
    receipt_path = os.environ.get("UL_QUALIFICATION_RECEIPT")
    if receipt_path:
        with Path(receipt_path).open("a", encoding="utf-8") as receipt:
            receipt.write(json.dumps({"input": value}, sort_keys=True) + "\n")
    return {"status": "open", "ticket": 42}
