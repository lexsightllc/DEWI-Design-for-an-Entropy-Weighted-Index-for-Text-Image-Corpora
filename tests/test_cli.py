"""Tests for DEWI command-line interface."""

import json
import shutil
from pathlib import Path
from typing import List, Optional

import importlib.util
import numpy as np
import pytest
from click.testing import CliRunner

from dewi.cli import cli

if importlib.util.find_spec("torch") is None:  # pragma: no cover - optional dep
    pytest.skip("torch not installed", allow_module_level=True)

# Test data
TEST_IDS = [f"d{i}" for i in range(5)]


def create_test_corpus(tmp_dir: Path, n_docs: int = 5, dim: int = 128) -> None:
    """Create a test corpus with text, images, and embeddings."""
    # Create text data
    texts_dir = tmp_dir / "texts"
    texts_dir.mkdir()
    for i, doc_id in enumerate(TEST_IDS[:n_docs]):
        with open(texts_dir / f"{doc_id}.txt", "w") as f:
            f.write(f"This is a test document {i} with some content." * 10)
    
    # Create image data (dummy numpy arrays)
    images_dir = tmp_dir / "images"
    images_dir.mkdir()
    for doc_id in TEST_IDS[:n_docs]:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        np.save(images_dir / f"{doc_id}.npy", img)
    
    # Create embeddings
    emb_dir = tmp_dir / "embeddings"
    emb_dir.mkdir()
    for doc_id in TEST_IDS[:n_docs]:
        emb = np.random.randn(dim).astype(np.float32)
        np.save(emb_dir / f"{doc_id}.npy", emb)
    
    # Create ID file
    with open(tmp_dir / "ids.txt", "w") as f:
        f.write("\n".join(TEST_IDS[:n_docs]))


def test_config_command(tmp_dir):
    """Test the config command."""
    runner = CliRunner()
    
    # Test default config
    result = runner.invoke(cli, ["config"])
    assert result.exit_code == 0
    assert "text:" in result.output
    
    # Test with output file
    output_file = tmp_dir / "config.yaml"
    result = runner.invoke(cli, ["config", "-o", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()
    
    # Test presets
    for preset in ["default", "web", "product", "balanced"]:
        result = runner.invoke(cli, ["config", "--preset", preset])
        assert result.exit_code == 0


def test_process_command(tmp_dir):
    """Test the process command with test data."""
    # Setup test data
    create_test_corpus(tmp_dir, n_docs=3)
    config_path = tmp_dir / "config.yaml"
    output_dir = tmp_dir / "output"
    
    # Generate config
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "-o", str(config_path)])
    assert result.exit_code == 0
    
    # Run process command
    result = runner.invoke(cli, [
        "process",
        str(config_path),
        str(output_dir),
        "--texts", str(tmp_dir / "texts"),
        "--images", str(tmp_dir / "images"),
        "--embeddings", str(tmp_dir / "embeddings")
    ])
    assert result.exit_code == 0
    
    # Check outputs
    assert (output_dir / "documents.jsonl").exists()
    assert (output_dir / "index").exists()
    
    # Verify documents.jsonl
    with open(output_dir / "documents.jsonl") as f:
        lines = f.readlines()
        assert len(lines) == 3
        doc = json.loads(lines[0])
        assert "doc_id" in doc
        assert "dewi_score" in doc


def test_search_command(tmp_dir):
    """Test the search command with a test index."""
    # Setup test data
    create_test_corpus(tmp_dir, n_docs=3)
    output_dir = tmp_dir / "output"
    output_dir.mkdir()
    
    # Create a simple index
    from dewi.index import DewiIndex, Payload
    
    dim = 128
    index = DewiIndex(dim=dim, space="cosine")
    
    for i, doc_id in enumerate(TEST_IDS[:3]):
        emb = np.random.randn(dim).astype(np.float32)
        payload = Payload(
            dewi=float(np.clip(np.random.beta(2, 2), 0, 1)),
            ht_mean=float(np.random.gamma(2, 0.5)),
            ht_q90=float(np.random.gamma(2, 0.5) * 1.5),
            hi_mean=float(np.random.gamma(2, 0.3)),
            hi_q90=float(np.random.gamma(2, 0.3) * 1.5),
            I_hat=float(np.random.beta(2, 2)),
            redundancy=float(np.random.beta(1, 5)),
            noise=float(np.random.beta(1, 10))
        )
        index.add(doc_id, emb, payload)
    
    index.build()
    index_path = output_dir / "test_index"
    index.save(index_path)
    
    # Test search
    runner = CliRunner()
    query = np.random.randn(dim).astype(np.float32)
    query_path = output_dir / "query.npy"
    np.save(query_path, query)
    
    # Test with default parameters
    result = runner.invoke(cli, [
        "search",
        str(index_path),
        "test query",
        "--k", "2"
    ])
    assert result.exit_code == 0
    
    # Test with custom parameters
    result = runner.invoke(cli, [
        "search",
        str(index_path),
        "test query",
        "--k", "1",
        "--eta", "0.5",
        "--entropy-pref", "0.8",
        "--output", str(output_dir / "results.json")
    ])
    assert result.exit_code == 0
    assert (output_dir / "results.json").exists()


def test_cli_help():
    """Test the CLI help output."""
    runner = CliRunner()
    
    # Test main help
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    
    # Test subcommand help
    for cmd in ["config", "process", "search"]:
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output
