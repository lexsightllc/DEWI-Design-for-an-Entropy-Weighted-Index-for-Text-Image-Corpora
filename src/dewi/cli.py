"""Command-line interface for DEWI."""

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
import yaml

from dewi import __version__
from dewi.config import DewiConfig, get_default_config
from dewi.index import DewiIndex, Payload
from dewi.pipelines import DewiPipeline, Document, create_document

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
        cfg.noise.nsfw_filter = 'strict'
        cfg.scoring.weights.alpha_n = 1.5  # Higher noise penalty for web content
        cfg.scoring.weights.alpha_r = 1.2  # Slightly higher redundancy penalty
    elif preset == 'product':
        cfg.scoring.weights.alpha_t = 1.2  # More weight to text (product specs)
        cfg.scoring.weights.alpha_r = 1.5  # Higher redundancy penalty for product dupes
        cfg.scoring.weights.alpha_n = 1.8  # High noise penalty for product images
    elif preset == 'balanced':
        # Default weights are already balanced
        pass
    
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)
        click.echo(f"Configuration saved to {output_path}")
    else:
        yaml.dump(cfg.to_dict(), sys.stdout, default_flow_style=False, sort_keys=False)

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
@click.argument('index_dir', type=click.Path(file_okay=False, exists=True))
@click.argument('query', type=str)
@click.option('--k', type=int, default=10, help='Number of results to return')
@click.option('--eta', type=float, help='DEWI weight in re-ranking')
@click.option('--entropy-pref', type=float, help='Entropy preference (-1.0 to 1.0)')
@click.option('--output', '-o', type=click.Path(), help='Output file for results (JSON)')
def search(
    index_dir: str,
    query: str,
    k: int,
    eta: Optional[float],
    entropy_pref: Optional[float],
    output: Optional[str]
):
    """Search the DEWI index."""
    # Load index and configuration
    index = DewiIndex.load(index_dir)
    
    # Get query embedding (in a real app, this would use the same model as indexing)
    query_embedding = np.random.randn(index.dim).astype('float32')
    
    # Search
    results = index.search(
        query_embedding,
        k=k,
        eta=eta,
        entropy_pref=entropy_pref
    )
    
    # Format results
    formatted = []
    for doc_id, score, payload in results:
        formatted.append({
            'doc_id': doc_id,
            'score': float(score),
            'dewi_score': float(payload.dewi),
            'text_entropy': float(payload.ht_mean),
            'image_entropy': float(payload.hi_mean) if payload.hi_mean is not None else None,
            'cross_modal_mi': float(payload.I_hat) if payload.I_hat is not None else None,
            'redundancy': float(payload.redundancy) if payload.redundancy is not None else None,
            'noise': float(payload.noise) if payload.noise is not None else None
        })
    
    # Output results
    if output:
        with open(output, 'w') as f:
            json.dump(formatted, f, indent=2)
        click.echo(f"Results saved to {output}")
    else:
        click.echo(json.dumps(formatted, indent=2))

def _load_documents(
    texts_path: Optional[str],
    images_dir: Optional[str],
    embeddings_path: Optional[str]
) -> List[Document]:
    """Load documents from text files, image directories, and/or embeddings."""
    documents = []
    doc_id = 0
    
    # Load text documents
    if texts_path:
        with open(texts_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    doc_id += 1
                    documents.append(create_document(
                        doc_id=f"text_{doc_id}",
                        text=line
                    ))
    
    # Load image documents
    if images_dir:
        images_dir = Path(images_dir)
        image_paths = list(images_dir.glob('*.*'))
        
        for img_path in image_paths:
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                doc_id += 1
                documents.append(create_document(
                    doc_id=f"img_{doc_id}",
                    image_path=img_path
                ))
    
    # Load embeddings if provided
    if embeddings_path:
        embeddings = np.load(embeddings_path)
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
    
    return documents

def _save_results(documents: List[Document], output_dir: Path) -> None:
    """Save processing results to disk."""
    # Save documents as JSONL
    with open(output_dir / 'documents.jsonl', 'w') as f:
        for doc in documents:
            doc_dict = {
                'doc_id': doc.doc_id,
                'text': doc.text,
                'image_path': str(doc.image_path) if doc.image_path else None,
                'ht_mean': doc.ht_mean,
                'ht_q90': doc.ht_q90,
                'hi_mean': doc.hi_mean,
                'hi_q90': doc.hi_q90,
                'I_hat': doc.I_hat,
                'redundancy': doc.redundancy,
                'noise': doc.noise,
                'dewi_score': doc.dewi_score
            }
            if doc.embedding is not None:
                doc_dict['embedding'] = doc.embedding.tolist()
            f.write(json.dumps(doc_dict) + '\n')
    
    # Save index if embeddings are available
    if documents and documents[0].embedding is not None:
        index = DewiIndex(
            dim=len(documents[0].embedding),
            space='cosine'
        )
        
        for doc in documents:
            payload = Payload(
                dewi=doc.dewi_score or 0.0,
                ht_mean=doc.ht_mean or 0.0,
                ht_q90=doc.ht_q90 or 0.0,
                hi_mean=doc.hi_mean or 0.0,
                hi_q90=doc.hi_q90 or 0.0,
                I_hat=doc.I_hat or 0.0,
                redundancy=doc.redundancy or 0.0,
                noise=doc.noise or 0.0
            )
            index.add(doc.doc_id, doc.embedding, payload)
        
        index.build()
        index.save(output_dir / 'index')

if __name__ == '__main__':
    cli()
