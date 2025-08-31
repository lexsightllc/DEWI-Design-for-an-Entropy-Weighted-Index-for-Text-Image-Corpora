# DEWI Usage Guide

This guide provides comprehensive instructions for using DEWI (Design for an Entropy-Weighted Index) for text+image retrieval with entropy-aware ranking.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.9+
- pip

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dewi.git
   cd dewi
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install DEWI**
   ```bash
   # For basic usage
   pip install -e .
   
   # With optional dependencies for development
   pip install -e '.[dev]'
   ```

## Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:
```
data/
  ├── texts/           # Text documents
  │   ├── doc1.txt
  │   ├── doc2.txt
  │   └── ...
  └── images/          # Corresponding images
      ├── doc1.jpg
      ├── doc2.jpg
      └── ...
```

### 2. Generate a Configuration File

```bash
dewi config --preset balanced -o config/balanced.yaml
```

### 3. Process Your Data

```bash
dewi process \
  config/balanced.yaml \
  output/ \
  --texts data/texts \
  --images data/images
```

### 4. Search the Index

```bash
dewi search output/index/ "your search query" -k 10
```

## Configuration

DEWI uses a YAML configuration file to control various aspects of the system. You can generate a default configuration file using:

```bash
dewi config -o config/default.yaml
```

### Configuration Options

#### Text Processing
- `text.model`: Pretrained language model for text processing
- `text.quantiles`: List of quantiles to compute for text entropy
- `text.batch_size`: Batch size for text processing

#### Image Processing
- `image.model`: Model for image feature extraction
- `image.patch_size`: Size of image patches
- `image.batch_size`: Batch size for image processing

#### Cross-Modal Processing
- `cross_modal.model`: Model for cross-modal embeddings
- `cross_modal.batch_size`: Batch size for cross-modal processing

#### Scoring Weights
- `scoring.weights.alpha_t`: Weight for text entropy
- `scoring.weights.alpha_i`: Weight for image entropy
- `scoring.weights.alpha_m`: Weight for mutual information
- `scoring.weights.alpha_r`: Weight for redundancy penalty
- `scoring.weights.alpha_n`: Weight for noise penalty
- `scoring.delta`: Delta parameter for robust scoring
- `scoring.mode`: Scoring mode ('standard' or 'conditional')

## CLI Reference

### `dewi config`
Generate a configuration file.

**Options:**
- `-o, --output PATH`: Output file path
- `--preset [default|web|product|balanced]`: Configuration preset

**Example:**
```bash
dewi config -o config/my_config.yaml --preset product
```

### `dewi process`
Process documents and build the DEWI index.

**Arguments:**
- `CONFIG`: Path to configuration file
- `OUTPUT_DIR`: Directory to save outputs

**Options:**
- `--texts PATH`: Path to text documents
- `--images PATH`: Path to images
- `--embeddings PATH`: Path to pre-computed embeddings
- `--batch-size INT`: Override batch size
- `--device DEVICE`: Device to use (cpu, cuda, etc.)

**Example:**
```bash
dewi process config/my_config.yaml output/ --texts data/texts --images data/images
```

### `dewi search`
Search the DEWI index.

**Arguments:**
- `INDEX_DIR`: Path to the index directory
- `QUERY`: Search query string

**Options:**
- `-k, --top-k INT`: Number of results to return (default: 10)
- `--eta FLOAT`: Weight of DEWI score in re-ranking (0.0-1.0)
- `--entropy-pref FLOAT`: Entropy preference (-1.0 to 1.0)
- `-o, --output PATH`: Output file for results (JSON)

**Example:**
```bash
dewi search output/index/ "mountain landscape" -k 5 --eta 0.3 --entropy-pref 0.5
```

## Python API

### Basic Usage

```python
from dewi import DewiIndex, DewiPipeline
from dewi.config import get_default_config

# Initialize with default config
config = get_default_config()
pipeline = DewiPipeline(config)

# Process documents
docs = [...]  # List of Document objects
processed_docs = pipeline.compute_signals(docs)
scored_docs = pipeline.compute_dewi_scores(processed_docs)

# Build and search index
index = DewiIndex(dim=768, space="cosine")
for doc in scored_docs:
    index.add(doc.id, doc.embedding, doc.payload)

results = index.search(query_embedding, k=10, entropy_pref=0.5, eta=0.3)
```

## Performance Tuning

### Indexing Performance
- Increase `batch_size` for faster processing
- Use GPU if available by setting `device: cuda`
- For large datasets, process in chunks

### Search Performance
- Adjust `ef_construction` and `M` parameters in the index
- Lower `ef_query` for faster but less accurate search
- Use `entropy_pref=0` for pure similarity search

### Memory Usage
- Process large datasets in batches
- Use `float16` embeddings if precision allows
- Enable memory mapping for large indices

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
- Reduce batch size
- Use smaller models
- Enable gradient checkpointing

#### Slow Processing
- Increase batch size
- Use GPU if available
- Profile with `scripts/profile_index.py`

#### Poor Search Results
- Adjust scoring weights
- Check embedding quality
- Tune `eta` and `entropy_pref` parameters

### Getting Help

For additional help, please [open an issue](https://github.com/yourusername/dewi/issues) with:
- Description of the problem
- Steps to reproduce
- Error messages
- Environment details

## License

MIT
