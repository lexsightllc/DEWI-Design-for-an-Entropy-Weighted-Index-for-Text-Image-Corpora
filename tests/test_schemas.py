from pathlib import Path

from dewi import schemas


def test_export_schemas(tmp_path: Path) -> None:
    out = tmp_path / "schemas"
    schemas.export(out)
    assert (out / "dewi_config.schema.json").exists()
    assert (out / "payload.schema.json").exists()
