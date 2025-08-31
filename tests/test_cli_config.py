import json
from pathlib import Path

from click.testing import CliRunner

from dewi.cli import cli


def test_config_overwrite_and_presets(tmp_path: Path):
    runner = CliRunner()
    cfg_path = tmp_path / "cfg.yaml"

    r1 = runner.invoke(cli, ["config", "-o", str(cfg_path), "--preset", "web"])
    assert r1.exit_code == 0, r1.output

    r2 = runner.invoke(cli, ["config", "-o", str(cfg_path), "--preset", "product"])
    assert r2.exit_code != 0
    assert "Use --overwrite" in r2.output

    r3 = runner.invoke(cli, ["config", "-o", str(cfg_path), "--preset", "balanced", "--overwrite"])
    assert r3.exit_code == 0
    txt = cfg_path.read_text()
    assert "alpha_t" in txt and "alpha_r" in txt

