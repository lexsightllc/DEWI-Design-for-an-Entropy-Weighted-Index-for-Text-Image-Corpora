# DEWI: Design for an Entropy-Weighted Index for Text+Image Corpora

DEWI is a system for building and querying an entropy-weighted index that prioritizes useful surprise in multimodal (text+image) data. It combines signals from text and image modalities to identify and retrieve the most informative content while suppressing redundancy and noise.

## Features

- **Text Entropy Estimation**: Measures the information content of text using language models
- **Image Entropy Estimation**: Estimates image information content using reconstruction errors
- **Cross-Modal Analysis**: Computes mutual information between text and image modalities
- **Redundancy Detection**: Identifies near-duplicate content within and across modalities
- **Noise Estimation**: Detects and scores low-quality content
- **Efficient Indexing**: Fast approximate nearest neighbor search with HNSW
- **Re-ranking**: Combines semantic similarity with DEWI scores for improved retrieval

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dewi.git
   cd dewi
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

3. Install optional dependencies for GPU acceleration and additional features:
   ```bash
   pip install -e ".[dev]"
   ```

## Quick Start

### Building an Index

```python
import numpy as np
from dewi.scorer import DewiScorer, Signals, Weights
from dewi.index import DewiIndex, Payload

# Example data
ids = ["doc1", "doc2", "doc3"]
texts = ["example text 1", "example text 2", "example text 3"]
images = [np.random.rand(3, 224, 224) for _ in range(3)]  # Example image tensors
embeddings = [np.random.rand(768) for _ in range(3)]  # Example embeddings

# Initialize the index
index = DewiIndex(dim=768, space="cosine")

# Create some example signals
rows = []
for i, doc_id in enumerate(ids):
    # In practice, compute these using the appropriate modules
    signals = Signals(
        ht_mean=np.random.uniform(0, 10),  # Text entropy mean
        ht_q90=np.random.uniform(0, 15),   # Text entropy 90th percentile
        hi_mean=np.random.uniform(0, 5),   # Image entropy mean
        hi_q90=np.random.uniform(0, 8),    # Image entropy 90th percentile
        I_hat=np.random.uniform(0, 1),     # Cross-modal mutual information
        redundancy=np.random.uniform(0, 1), # Redundancy score
        noise=np.random.uniform(0, 0.2)     # Noise score
    )
    rows.append(signals)
    
    # Add to index
    payload = Payload(
        dewi=0.0,  # Will be updated after fitting
        **signals.__dict__
    )
    index.add(doc_id, embeddings[i], payload)

# Fit the scorer and update DEWI scores
scorer = DewiScorer(Weights())
scorer.fit_stats(rows)

# Update payloads with computed DEWI scores
for doc_id in ids:
    payload = index.get_payload(doc_id)
    if payload:
        signals = Signals(**{k: getattr(payload, k) for k in Signals.__annotations__})
        payload.dewi = scorer.score(signals)

# Build the index for efficient search
index.build()

# Save the index (example, would need implementation)
# index.save("dewi_index.bin")
```

### Querying the Index

```python
# Example query
query_embedding = np.random.rand(768)  # Your query embedding

# Search with DEWI re-ranking
results = index.search(
    query_embedding, 
    k=5, 
    entropy_pref=0.5,  # Favor higher DEWI scores
    eta=0.3            # Weight of DEWI in re-ranking
)

for doc_id, score, payload in results:
    print(f"Doc ID: {doc_id}, Score: {score:.3f}, DEWI: {payload.dewi:.3f}")
```

## Modules

- `dewi.scorer`: Core scoring logic with robust statistics
- `dewi.index`: Efficient indexing and retrieval with HNSW
- `dewi.metrics`: Evaluation metrics for retrieval performance
- `dewi.local_weights`: Utilities for per-token/per-patch weighting

## Documentation

For detailed documentation, see the [docs](docs/) directory.

## License

MIT

## Citation

If you use DEWI in your research, please cite:

```bibtex
@misc{dewi2023,
  title={DEWI: A Design for an Entropy-Weighted Index for Text+Image Corpora},
  author={Your Name},
  year={2023},
  howpublished={\url{https://github.com/yourusername/dewi}}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
