from __future__ import annotations

import unicodedata

from rich.console import Console

console = Console()


def print_dataset_plain(message: str) -> None:
    safe_message = "".join(
        character
        if (ord(character) >= 32 and not 0x7F <= ord(character) <= 0x9F)
        and unicodedata.category(character) not in {"Cf", "Cs"}
        else f"\\u{ord(character):04x}"
        for character in message
    )
    console.print(safe_message, markup=False, highlight=False, soft_wrap=True)
