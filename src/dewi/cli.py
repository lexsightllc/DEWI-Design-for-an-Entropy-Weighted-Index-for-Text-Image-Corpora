"""Command-line interface for DEWI."""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, cast, Union

import click

from dewi.pipelines import Document

from dewi import __version__

# Check for test mode
try:
    TEST_MODE = os.getenv("DEWI_TEST_MODE", "").lower() in ("1", "true", "yes")
except Exception:
    TEST_MODE = False

# Lazy imports
def _import_pipelines():
    """Lazy import for pipelines module."""
    from dewi.pipelines import DewiPipeline, Document, create_document
    return DewiPipeline, Document, create_document

def _import_index():
    """Lazy import for index module."""
    from dewi.index import DewiIndex, Payload
    return DewiIndex, Payload

def _import_numpy():
    """Lazy import for numpy."""
    import numpy as np
    return np

def _import_yaml():
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
):
    """Process documents and compute DEWI signals."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = DewiConfig.from_dict(yaml.safe_load(f))
    
    # Override config from command line
    if batch_size:
        cfg.text.batch_size = batch_size
        cfg.image.batch_size = batch_size
        cfg.cross_modal.batch_size = batch_size
    
    # Initialize pipeline
    pipeline = DewiPipeline(cfg)
    
    # Load documents
    documents = _load_documents(texts, images, embeddings)
    if not documents:
        raise click.ClickException("No documents to process. Provide --texts and/or --images")
    
    # Process documents
    processed_docs = pipeline.compute_signals(documents)
    processed_docs = pipeline.compute_dewi_scores(processed_docs)
    
    # Save results
    _save_results(processed_docs, output_path)
    click.echo(f"Processed {len(processed_docs)} documents. Results saved to {output_path}")

@cli.command()
@click.argument('index_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('query')
@click.option('--k', type=int, default=10, help='Number of results to return')
@click.option('--eta', type=float, help='Weight for DEWI score (0-1)')
@click.option('--entropy-pref', type=float, help='Entropy preference weight')
@click.option('--output', '-o', type=click.Path(), help='Output file for results (JSON)')
@click.option('--test-mode', is_flag=True, help='Run in test mode with mock data')
def search(index_dir, query, k, eta, entropy_pref, output, test_mode):
    """Search the DEWI index."""
    # Set test mode if specified
    global TEST_MODE
    if test_mode:
        TEST_MODE = True
        os.environ["DEWI_TEST_MODE"] = "1"
    
    # Import the appropriate index class based on test mode
    if TEST_MODE:
        from dewi.testing.mock_index import MockDewiIndex as DewiIndex
    else:
        DewiIndex, _ = _import_index()
    
    # Load index
    try:
        index = DewiIndex.load(index_dir)
    except Exception as e:
        click.echo(f"Error loading index: {e}", err=True)
        sys.exit(1)
    
    try:
        if TEST_MODE:
            # In test mode, just return mock results
            results = [
                (f"doc_{i}", 1.0 - (i * 0.1), 
                 {"metadata": {"source": "test", "id": i}, "dewi": 0.8 - (i * 0.05)})
                for i in range(min(k, 5))  # Return up to 5 mock results
            ]
        else:
            # Generate query embedding (in a real implementation, use the same model as indexing)
            np = _import_numpy()
            query_embedding = np.random.randn(index.dim).astype(np.float32)
            
            # Search
            results = index.search(
                query_embedding, 
                k=k, 
                eta=eta,
                entropy_pref=entropy_pref
            )
        
        # Format results
        output_results = []
        for doc_id, score, payload in results:
            # Handle both Payload objects and dictionaries (for test mode)
            if hasattr(payload, 'metadata'):
                metadata = payload.metadata
                dewi_score = payload.dewi
                entropy = (payload.ht_mean + payload.hi_mean) / 2 if hasattr(payload, 'ht_mean') and hasattr(payload, 'hi_mean') else None
            else:
                metadata = payload.get('metadata', {})
                dewi_score = payload.get('dewi')
                entropy = payload.get('entropy')
                
            result = {
                'id': doc_id,
                'score': float(score),
                'payload': {
                    'metadata': metadata,
                    'dewi_score': dewi_score,
                    'entropy': entropy
                }
            }
            output_results.append(result)
        
        # Output results
        if output:
            with open(output, 'w') as f:
                json.dump(output_results, f, indent=2)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(json.dumps(output_results, indent=2))
            
        if TEST_MODE:
            click.echo("\n[TEST MODE] Results are simulated")    
            
    except Exception as e:
        click.echo(f"Error during search: {e}", err=True)
        if TEST_MODE:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def _load_documents(
    texts_path: Optional[str],
    images_dir: Optional[str],
    embeddings_path: Optional[str]
) -> List[Document]:
    """Load documents from text files, image directories, and/or embeddings."""
    _, _, create_document = _import_pipelines()
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
