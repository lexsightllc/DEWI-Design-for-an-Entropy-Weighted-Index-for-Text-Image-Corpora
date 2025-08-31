"""Command-line interface for DEWI."""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, cast, Union

import click

from dewi import __version__

# Check for test mode
try:
    TEST_MODE = os.getenv("DEWI_TEST_MODE", "").lower() in ("1", "true", "yes")
except Exception:
    TEST_MODE = False

# Lazy imports
def _import_pipelines() -> tuple:
    """Lazy import for pipelines module."""
    from dewi.pipelines import DewiPipeline
    return (DewiPipeline, )

def _import_index() -> tuple:
    """Lazy import for index module."""
    from dewi.index import DewiIndex, Payload
    return DewiIndex, Payload

def _import_numpy() -> 'numpy':
    """Lazy import for numpy."""
    import numpy as np
    return np

def _import_yaml() -> 'yaml':
    """Lazy import for yaml."""
    import yaml
    return yaml

CONTEXT_SETTINGS = {
    'help_option_names': ['-h', '--help'],
    'max_content_width': 120
}

@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """DEWI: Design for an Entropy-Weighted Index for Text+Image Corpora."""
    pass

@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output config file path')
@click.option('--overwrite', is_flag=True, help='Overwrite output file if it exists')
@click.option('--preset', type=click.Choice(['default', 'web', 'product', 'balanced']), 
              default='default', help='Configuration preset')
def config(output: Optional[str], overwrite: bool, preset: str):
    """Generate a configuration file with default settings."""
    yaml = _import_yaml()
    from dewi.config import get_default_config
    
    # Get default config
    cfg = get_default_config()
    
    # Apply preset overrides
    if preset != 'default':
        # Preferred path for current config structure
        if hasattr(cfg, 'scoring') and hasattr(cfg.scoring, 'weights'):
            if preset == 'web':
                cfg.scoring.weights.alpha_t = 0.7
                cfg.scoring.weights.alpha_r = 0.3
            elif preset == 'product':
                cfg.scoring.weights.alpha_t = 0.6
                cfg.scoring.weights.alpha_r = 0.4
            elif preset == 'balanced':
                cfg.scoring.weights.alpha_t = 0.5
                cfg.scoring.weights.alpha_r = 0.5
        else:
            # Fallback for older config versions without nested scoring structure
            if preset == 'web':
                cfg.entropy_weight = 0.7
                cfg.redundancy_weight = 0.3
            elif preset == 'product':
                cfg.entropy_weight = 0.6
                cfg.redundancy_weight = 0.4
            elif preset == 'balanced':
                cfg.entropy_weight = 0.5
                cfg.redundancy_weight = 0.5
    
    # Convert config to dictionary for YAML serialization
    if hasattr(cfg, 'dict'):  # Pydantic model
        config_dict = cfg.dict()
    elif hasattr(cfg, 'to_dict'):
        config_dict = cfg.to_dict()
    else:
        # Fallback for generic objects
        config_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}
    
    if output:
        output_path = Path(output)
        if output_path.exists() and not overwrite:
            click.echo(f"Error: File {output} already exists. Use --overwrite to replace it.", err=True)
            sys.exit(1)
            
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            click.echo(f"Configuration saved to {output_path}")
        except Exception as e:
            click.echo(f"Error saving config: {e}", err=True)
            sys.exit(1)
    else:
        # Print to stdout
        click.echo("# DEWI Configuration")
        click.echo(f"# Preset: {preset}")
        click.echo("# Save this to a file and modify as needed\n")
        click.echo(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))

@cli.command()
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--texts', type=click.Path(exists=True), help='File with one text per line')
@click.option('--images', type=click.Path(exists=True), help='Directory containing images')
@click.option('--embeddings', type=click.Path(exists=True), help='Numpy file with embeddings')
@click.option('--batch-size', type=int, help='Override batch size for processing')
@click.option('--device', type=str, help='Device to use (cpu, cuda, etc.)')
def process(
    config_path: str,
    output_dir: str,
    texts: Optional[str],
    images: Optional[str],
    embeddings: Optional[str],
    batch_size: Optional[int],
    device: Optional[str]
) -> None:
    """Process documents and compute DEWI signals.
    
    Args:
        config_path: Path to the configuration file
        output_dir: Directory to save the processed results
        texts: Path to a text file or directory containing text files
        images: Path to a directory containing images
        embeddings: Path to a numpy file with precomputed embeddings
        batch_size: Override batch size for processing
        device: Device to use for processing (cpu, cuda, etc.)
    
    Raises:
        click.ClickException: If no documents are found or processing fails
    """
    try:
        # Lazy imports
        yaml = _import_yaml()
        from dewi.config import DewiConfig  # Lazy import
        from dewi.pipelines import DewiPipeline  # Lazy import
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            cfg = DewiConfig.from_dict(yaml.safe_load(f))
        
        # Override config from command line
        if batch_size:
            cfg.text.batch_size = batch_size
            cfg.image.batch_size = batch_size
            if hasattr(cfg, 'cross_modal') and cfg.cross_modal:
                cfg.cross_modal.batch_size = batch_size
        
        # Set device if provided
        if device:
            import torch  # Lazy import
            cfg.device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
        
        # Initialize pipeline
        pipeline = DewiPipeline(cfg)
        
        # Load documents
        documents = _load_documents(texts, images, embeddings)
        if not documents:
            raise click.ClickException("No documents to process. Provide --texts and/or --images")
        
        # Process documents
        click.echo(f"Processing {len(documents)} documents...")
        processed_docs = pipeline.compute_signals(documents)
        processed_docs = pipeline.compute_dewi_scores(processed_docs)
        
        # Save results
        _save_results(processed_docs, output_path)
        click.echo(f"✓ Processed {len(processed_docs)} documents. Results saved to {output_path}")
        
    except Exception as e:
        if TEST_MODE:
            import traceback
            traceback.print_exc()
        raise click.ClickException(f"Error during processing: {str(e)}")

@cli.command()
@click.argument('index_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('query')
@click.option('--k', type=int, default=10, help='Number of results to return')
@click.option('--eta', type=float, help='Weight for DEWI score (0-1)')
@click.option('--entropy-pref', type=float, help='Entropy preference weight')
@click.option('--output', '-o', type=click.Path(), help='Output file for results (JSON)')
@click.option('--test-mode', is_flag=True, help='Run in test mode with mock data')
def search(
    index_dir: str,
    query: str,
    k: int,
    eta: Optional[float],
    entropy_pref: Optional[float],
    output: Optional[str],
    test_mode: bool
) -> None:
    """Search the DEWI index.
    
    Args:
        index_dir: Path to the directory containing the DEWI index
        query: Search query string
        k: Number of results to return
        eta: Weight for DEWI score (0-1)
        entropy_pref: Entropy preference weight
        output: Output file path for results (JSON)
        test_mode: Run in test mode with mock data
        
    Raises:
        click.ClickException: If the search operation fails
    """
    global TEST_MODE
    TEST_MODE = test_mode or TEST_MODE
    
    try:
        # Lazy imports
        from dewi.index import DewiIndex  # Lazy import
        import json  # Lazy import
        
        # Load index
        index_path = Path(index_dir)
        index = DewiIndex.load(index_path)
        
        # Set default eta if not provided
        if eta is None:
            eta = 0.5  # Default weight for DEWI score
            
        # Set default entropy preference if not provided
        if entropy_pref is None:
            entropy_pref = 1.0  # Default preference for high entropy
        
        click.echo(f"Searching for '{query}'...")
        
        # Search
        results = index.search(
            query=query,
            k=k,
            eta=eta,
            entropy_pref=entropy_pref
        )
        
        if not results:
            click.echo("No results found.")
            return
        
        # Format results
        formatted_results = []
        for doc_id, score, payload in results:
            result = {
                'id': doc_id,
                'score': float(score),
                'text': '',
                'metadata': {},
                'dewi_score': None,
                'entropy': None
            }
            
            # Handle both Payload objects and dictionaries (for test mode)
            if hasattr(payload, 'metadata'):
                result.update({
                    'text': getattr(payload, 'text', ''),
                    'metadata': getattr(payload, 'metadata', {}),
                    'dewi_score': getattr(payload, 'dewi', None),
                    'entropy': (getattr(payload, 'ht_mean', 0) + getattr(payload, 'hi_mean', 0)) / 2 
                              if hasattr(payload, 'ht_mean') and hasattr(payload, 'hi_mean') 
                              else None
                })
            elif isinstance(payload, dict):
                result.update({
                    'text': payload.get('text', ''),
                    'metadata': payload.get('metadata', {}),
                    'dewi_score': payload.get('dewi'),
                    'entropy': payload.get('entropy')
                })
            
            formatted_results.append(result)
        
        # Output results
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(formatted_results, f, indent=2)
            click.echo(f"✓ Results saved to {output_path}")
        else:
            click.echo(json.dumps(formatted_results, indent=2))
            
        if TEST_MODE:
            click.echo("\n[TEST MODE] Results are simulated")    
            
    except Exception as e:
        click.echo(f"Error during search: {e}", err=True)
        if TEST_MODE:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def create_document(text: str = None, metadata: Optional[Dict] = None) -> Any:
    """Create a new document with the given text and metadata."""
    from dewi.pipelines import Document  # Lazy import
    return Document(doc_id=str(uuid.uuid4()), text=text, metadata=metadata or {})

def _load_documents(
    texts_path: Optional[str],
    images_dir: Optional[str],
    embeddings_path: Optional[str],
    max_workers: int = 4
) -> List[Any]:
    """Load documents from text files, image directories, and/or embeddings.
    
    Args:
        texts_path: Path to a text file or directory containing text files
        images_dir: Path to a directory containing images
        embeddings_path: Path to a numpy file with precomputed embeddings
        max_workers: Maximum number of worker processes for parallel loading
        
    Returns:
        List of Document objects with metadata
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import uuid
    
    documents = []
    
    def _load_text_file(file_path: Path) -> Optional[Any]:
        """Helper to load a single text file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                return create_document(
                    text=content,
                    metadata={
                        "source": str(file_path),
                        "type": "text",
                        "file_size": file_path.stat().st_size,
                        "last_modified": file_path.stat().st_mtime
                    }
                )
        except Exception as e:
            click.echo(f"Error reading {file_path}: {e}", err=True)
            return None
    
    # Load text documents
    if texts_path and not TEST_MODE:
        texts_path = Path(texts_path)
        text_files = []
        
        if texts_path.is_file():
            if texts_path.suffix.lower() in ('.txt', '.md', '.json', '.jsonl'):
                text_files.append(texts_path)
        elif texts_path.is_dir():
            # Support multiple text file formats
            for ext in ('*.txt', '*.md', '*.json', '*.jsonl'):
                text_files.extend(texts_path.glob(f'**/{ext}'))
        
        # Process text files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_load_text_file, f) for f in text_files]
            with tqdm(total=len(futures), desc="Loading text files", unit="file") as pbar:
                for future in as_completed(futures):
                    doc = future.result()
                    if doc:
                        documents.append(doc)
                    pbar.update(1)
    
    # Load images
    if images_dir and not TEST_MODE:
        images_path = Path(images_dir)
        if images_path.exists() and images_path.is_dir():
            image_files = list(images_path.glob('**/*.jpg')) + \
                         list(images_path.glob('**/*.jpeg')) + \
                         list(images_path.glob('**/*.png'))
            
            for img_path in tqdm(image_files, desc="Loading images", unit="image"):
                try:
                    doc = create_document(
                        text="",  # Will be filled by image captioning
                        metadata={
                            "source": str(img_path),
                            "type": "image",
                            "file_size": img_path.stat().st_size,
                            "last_modified": img_path.stat().st_mtime,
                            "dimensions": None,  # Will be filled during processing
                            "format": img_path.suffix.lower()
                        }
                    )
                    # Store image path for later processing
                    doc.image_path = str(img_path)
                    documents.append(doc)
                except Exception as e:
                    click.echo(f"Error processing {img_path}: {e}", err=True)
    
    # Load precomputed embeddings
    if embeddings_path and not TEST_MODE:
        try:
            np = _import_numpy()
            data = np.load(embeddings_path, allow_pickle=True)
            
            if 'embeddings' in data and 'doc_ids' in data:
                for i, (emb, doc_id) in enumerate(zip(data['embeddings'], data['doc_ids'])):
                    doc = create_document(
                        text="",
                        metadata={
                            "source": f"embeddings_{i}",
                            "type": "embedding",
                            "embedding_shape": emb.shape
                        }
                    )
                    doc.embedding = emb
                    documents.append(doc)
        except Exception as e:
            click.echo(f"Error loading embeddings from {embeddings_path}: {e}", err=True)
    
    # In test mode, return mock data
    if TEST_MODE:
        return [
            create_document(
                text=f"Test document {i}",
                metadata={"test": True, "id": i}
            )
            for i in range(5)
        ]
    
    return documents

def _save_results(
    documents: List[Any],
    output_dir: Union[str, Path],
    save_embeddings: bool = True,
    save_signals: bool = True,
    save_text: bool = True,
    batch_size: int = 1000
) -> None:
    """Save processing results to disk in a structured format.
    
    Args:
        documents: List of Document objects to save
        output_dir: Directory to save the results
        save_embeddings: Whether to save document embeddings
        save_signals: Whether to save signal information
        save_text: Whether to save document text
        batch_size: Number of documents to process in each batch
    """
    import json
    import shutil
    from pathlib import Path
    from datetime import datetime
    from typing import Dict, Any, List, Optional
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a backup of existing output directory if it exists
    if output_dir.exists() and any(output_dir.iterdir()):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = output_dir.parent / f"{output_dir.name}_backup_{timestamp}"
        shutil.copytree(output_dir, backup_dir)
        click.echo(f"Created backup of existing output at {backup_dir}")
    
    # In test mode, generate and save mock data
    if TEST_MODE:
        test_data = [
            {
                'id': f'doc_{i}',
                'text': f'Test document {i} with some sample content to test the DEWI system.',
                'metadata': {
                    'source': 'test',
                    'id': i,
                    'type': 'test',
                    'created': datetime.now().isoformat()
                },
                'signals': {
                    'ht_mean': round(0.5 + (i * 0.1), 4),
                    'hi_mean': round(0.4 + (i * 0.05), 4),
                    'I_hat': round(0.3 + (i * 0.02), 4),
                    'redundancy': round(0.1 + (i * 0.01), 4),
                    'noise': round(0.05 + (i * 0.005), 4),
                    'entropy_score': round(0.7 - (i * 0.05), 4)
                },
                'embedding': [round(0.1 * (i + 1), 4) for _ in range(10)]  # Simple test embedding
            }
            for i in range(10)  # Create 10 test documents
        ]
        
        # Save test documents in JSONL format
        with open(output_dir / 'documents.jsonl', 'w', encoding='utf-8') as f:
            for doc in test_data:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        # Save test signals summary
        signals_data = []
        for doc in test_data:
            sig = {
                'id': doc['id'],
                'source': doc['metadata'].get('source', ''),
                **doc['signals']
            }
            signals_data.append(sig)
        
        with open(output_dir / 'signals_summary.json', 'w', encoding='utf-8') as f:
            json.dump(signals_data, f, indent=2, ensure_ascii=False)
        
        # Save test embeddings
        np = _import_numpy()
        embeddings = np.array([doc['embedding'] for doc in test_data])
        doc_ids = [doc['id'] for doc in test_data]
        
        np.savez_compressed(
            output_dir / 'embeddings.npz',
            embeddings=embeddings,
            doc_ids=doc_ids
        )
        
        # Save metadata
        metadata = {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'num_documents': len(test_data),
            'dimensions': embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            'test_mode': True
        }
        with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return
    
    # Normal mode - process real documents
    try:
        # Initialize metadata
        metadata = {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'num_documents': len(documents),
            'save_embeddings': save_embeddings,
            'save_signals': save_signals,
            'save_text': save_text,
            'test_mode': False
        }
        
        # Process documents in batches
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(documents), batch_size):
            batch = documents[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            # Prepare batch data
            batch_docs = []
            batch_embeddings = []
            batch_signals = []
            
            for doc in batch:
                # Document data
                doc_data = {
                    'id': getattr(doc, 'id', str(uuid.uuid4())),
                    'metadata': getattr(doc, 'metadata', {})
                }
                
                # Add text if available and requested
                if save_text and hasattr(doc, 'text'):
                    doc_data['text'] = doc.text
                
                # Add signals if available and requested
                if save_signals and hasattr(doc, 'signals') and doc.signals:
                    signals = doc.signals.dict() if hasattr(doc.signals, 'dict') else dict(doc.signals)
                    doc_data['signals'] = signals
                    
                    # Add to signals summary
                    sig_data = {'id': doc_data['id']}
                    if 'source' in doc_data['metadata']:
                        sig_data['source'] = doc_data['metadata']['source']
                    sig_data.update(signals)
                    batch_signals.append(sig_data)
                
                # Add embedding if available and requested
                if save_embeddings and hasattr(doc, 'embedding') and doc.embedding is not None:
                    embedding = doc.embedding
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    doc_data['embedding'] = embedding
                    
                    # Store for batch processing
                    if save_embeddings:
                        batch_embeddings.append((doc_data['id'], embedding))
                
                batch_docs.append(doc_data)
            
            # Save batch documents
            batch_file = output_dir / f"documents_batch_{batch_num:04d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for doc in batch_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            # Save batch embeddings
            if batch_embeddings and save_embeddings:
                np = _import_numpy()
                ids, embs = zip(*batch_embeddings)
                emb_file = output_dir / f"embeddings_batch_{batch_num:04d}.npz"
                np.savez_compressed(
                    emb_file,
                    doc_ids=ids,
                    embeddings=np.array(embs, dtype=np.float32)
                )
            
            click.echo(f"Processed batch {batch_num}/{total_batches} ({len(batch)} documents)")
        
        # Save signals summary
        if batch_signals and save_signals:
            signals_file = output_dir / 'signals_summary.json'
            with open(signals_file, 'w', encoding='utf-8') as f:
                json.dump(batch_signals, f, indent=2, ensure_ascii=False)
        
        # Update metadata with actual counts
        if hasattr(documents[0], 'embedding') and documents[0].embedding is not None:
            metadata['dimensions'] = len(documents[0].embedding)
        
        # Save metadata
        with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        click.echo(f"\n✓ Successfully saved {len(documents)} documents to {output_dir}")
        
    except Exception as e:
        click.echo(f"\n✗ Error saving results: {e}", err=True)
        if TEST_MODE:
            import traceback
            traceback.print_exc()
        raise
if __name__ == '__main__':
    cli()
