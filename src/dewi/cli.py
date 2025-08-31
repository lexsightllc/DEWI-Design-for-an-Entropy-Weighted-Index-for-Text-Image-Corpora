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
@click.option('--preset', type=click.Choice(['default', 'web', 'product', 'balanced']), 
              default='default', help='Configuration preset')
def config(output: Optional[str], preset: str):
    """Generate a configuration file with default settings."""
    cfg = get_default_config()
    
    # Apply preset overrides
    if preset == 'web':
        # Web preset configuration
        cfg.entropy_weight = 0.7
        cfg.redundancy_weight = 0.3
    elif preset == 'product':
        # Product catalog preset
        cfg.entropy_weight = 0.6
        cfg.redundancy_weight = 0.4
    elif preset == 'balanced':
        # Balanced preset
        cfg.entropy_weight = 0.5
        cfg.redundancy_weight = 0.5
    
    if output:
        output_path = Path(output)
        if output_path.exists() and not overwrite:
            click.echo(f"Error: File {output} already exists. Use --overwrite to replace it.", err=True)
            sys.exit(1)
            
        try:
            with open(output_path, 'w') as f:
                yaml.dump(default_config.dict(), f, default_flow_style=False, sort_keys=False)
            click.echo(f"Configuration saved to {output_path}")
        except Exception as e:
            click.echo(f"Error saving config: {e}", err=True)
            sys.exit(1)
    else:
        # Print to stdout
        click.echo("# DEWI Configuration")
        click.echo("# Save this to a file and modify as needed\n")
        click.echo(yaml.dump(default_config.dict(), default_flow_style=False, sort_keys=False))

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
    embeddings_path: Optional[str]
) -> List[Any]:  # Return type is List[Document] but we can't import Document here
    """Load documents from text files, image directories, and/or embeddings."""
    import uuid
    documents = []
    
    # Load text documents
    if texts_path and not TEST_MODE:  # Skip in test mode as we'll use mock data
        texts_path = Path(texts_path)
        if texts_path.is_file() and texts_path.suffix == '.txt':
            # Single text file
            try:
                with open(texts_path) as f:
                    content = f.read()
                    doc = create_document(
                        text=content,
                        metadata={"source": str(texts_path)}
                    )
                    documents.append(doc)
            except Exception as e:
                click.echo(f"Error reading {texts_path}: {e}", err=True)
        elif texts_path.is_dir():
            # Directory of text files
            for txt_file in texts_path.glob('**/*.txt'):
                try:
                    with open(txt_file) as f:
                        content = f.read()
                        doc = create_document(
                            text=content,
                            metadata={"source": str(txt_file)}
                        )
                        documents.append(doc)
                except Exception as e:
                    click.echo(f"Error reading {txt_file}: {e}", err=True)
    
    # In test mode, return an empty list (will be handled by the caller)
    if TEST_MODE:
        return []
    
    # TODO: Add support for loading images and embeddings
    # This is a placeholder for the actual implementation
    
    return documents

def _save_results(documents, output_dir):
    """Save processing results to disk."""
    import json
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # In test mode, just create empty files
    if TEST_MODE:
        # Create a simple test output
        test_data = [
            {
                'id': f'doc_{i}',
                'text': f'Test document {i}',
                'metadata': {'source': 'test', 'id': i},
                'signals': {
                    'ht_mean': 0.5 + (i * 0.1),
                    'hi_mean': 0.4 + (i * 0.05),
                    'I_hat': 0.3 + (i * 0.02),
                    'redundancy': 0.1,
                    'noise': 0.05
                },
                'embedding': [0.1 * (i + 1)] * 10  # Simple test embedding
            }
            for i in range(5)  # Create 5 test documents
        ]
        
        # Save test documents
        with open(output_dir / 'documents.jsonl', 'w') as f:
            for doc in test_data:
                f.write(json.dumps(doc) + '\n')
        
        # Save test signals summary
        signals_data = [
            {
                'id': doc['id'],
                **doc['signals']
            }
            for doc in test_data
        ]
        
        with open(output_dir / 'signals_summary.json', 'w') as f:
            json.dump(signals_data, f, indent=2)
        
        # Save test embeddings
        np = _import_numpy()
        embeddings = np.array([doc['embedding'] for doc in test_data])
        doc_ids = [doc['id'] for doc in test_data]
        
        np.savez_compressed(
            output_dir / 'embeddings.npz',
            embeddings=embeddings,
            doc_ids=doc_ids
        )
        
        return
    
    # Normal mode - save actual documents
    try:
        # Save documents with their metadata and signals
        documents_file = output_dir / 'documents.jsonl'
        with open(documents_file, 'w') as f:
            for doc in documents:
                # Convert document to dict
                doc_dict = {
                    'id': doc.id,
                    'text': doc.text,
                    'metadata': doc.metadata,
                    'signals': doc.signals.dict() if hasattr(doc, 'signals') and doc.signals else {},
                    'embedding': doc.embedding.tolist() if hasattr(doc, 'embedding') and doc.embedding is not None else None
                }
                f.write(json.dumps(doc_dict) + '\n')
        
        # Save signals summary
        if any(hasattr(doc, 'signals') and doc.signals for doc in documents):
            signals_file = output_dir / 'signals_summary.json'
            signals_data = []
            for doc in documents:
                if hasattr(doc, 'signals') and doc.signals:
                    signals_data.append({
                        'id': doc.id,
                        **doc.signals.dict()
                    })
            
            if signals_data:
                with open(signals_file, 'w') as f:
                    json.dump(signals_data, f, indent=2)
        
        # Save embeddings if available
        if any(hasattr(doc, 'embedding') and doc.embedding is not None for doc in documents):
            np = _import_numpy()
            embeddings_file = output_dir / 'embeddings.npz'
            embeddings = []
            doc_ids = []
            
            for doc in documents:
                if hasattr(doc, 'embedding') and doc.embedding is not None:
                    embeddings.append(doc.embedding)
                    doc_ids.append(doc.id)
            
            if embeddings:
                np.savez_compressed(
                    embeddings_file,
                    embeddings=np.stack(embeddings),
                    doc_ids=doc_ids
                )
    except Exception as e:
        click.echo(f"Error saving results: {e}", err=True)
        if TEST_MODE:
            import traceback
            traceback.print_exc()
        raise
if __name__ == '__main__':
    cli()
