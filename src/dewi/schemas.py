from __future__ import annotations

import json
from pathlib import Path
from typing import Type

from pydantic import TypeAdapter

from .config import DewiConfig
from .types import Payload


def _schema_for(cls: Type) -> dict:
    """Generate a JSON schema for a dataclass using Pydantic type adapters."""
    return TypeAdapter(cls).json_schema()


def export(output_dir: Path) -> None:
    """Export JSON schemas for public types to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    schemas = {
        "dewi_config.schema.json": _schema_for(DewiConfig),
        "payload.schema.json": _schema_for(Payload),
    }
    for name, schema in schemas.items():
        with open(output_dir / name, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)


if __name__ == "__main__":  # pragma: no cover - manual execution
    export(Path("docs/schemas"))
