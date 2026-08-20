from __future__ import annotations

from ul_cli.demo_runner import DemoRetryHandler as DefectiveRetryHandler
from ul_cli.demo_runner import create_server

__all__ = ["DefectiveRetryHandler", "create_server"]


if __name__ == "__main__":
    create_server(8766).serve_forever()
