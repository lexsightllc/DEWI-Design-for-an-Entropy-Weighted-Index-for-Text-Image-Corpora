"""Configuration management for DEWI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml

@dataclass
class TextConfig:
    """Configuration for text signal computation."""
    model: str = "distilroberta-base"
    quantiles: List[float] = field(default_factory=lambda: [0.9])
    batch_size: int = 64

@dataclass
class ImageConfig:
    """Configuration for image signal computation."""
    model: str = "facebook/vit-mae-base"
    patch_size: int = 16
    batch_size: int = 64

@dataclass
class CrossModalConfig:
    """Configuration for cross-modal signal computation."""
    model: str = "openai/clip-vit-base-patch32"
    batch_size: int = 128

@dataclass
class RedundancyConfig:
    """Configuration for redundancy detection."""
    text_sim: str = "minhash"  # or "simhash"
    image_sim: str = "phash+vit"
    cross_modal_density: bool = True

@dataclass
class NoiseConfig:
    """Configuration for noise detection."""
    blur_threshold: float = 0.25
    ocr_min_coverage: float = 0.6
    nsfw_filter: str = "lite"  # or "strict" or "none"

@dataclass
class ScoringWeights:
    """Weights for DEWI scoring components."""
    alpha_t: float = 1.0  # text surprisal
    alpha_i: float = 1.0  # image surprisal
    alpha_m: float = 1.0  # mutual information
    alpha_r: float = 1.0  # redundancy penalty
    alpha_n: float = 1.0  # noise penalty

@dataclass
class ScoringConfig:
    """Configuration for scoring."""
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    delta: float = 3.0  # clamp for robust scoring
    mode: str = "standard"  # or "conditional"

@dataclass
class IndexConfig:
    """Configuration for index construction and querying."""
    ann: str = "hnsw"  # or "bruteforce"
    metric: str = "cosine"  # or "l2"
    ef_construction: int = 200
    M: int = 32
    ef_query: int = 200
    rerank_eta: float = 0.25  # DEWI strength in re-ranking
    entropy_pref: float = 0.0  # [-1, 1]

@dataclass
class TrainingSamplingConfig:
    """Configuration for training data sampling."""
    tau: float = 1.0  # DEWI^tau sampling
    submodular_pick: bool = False

@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    recall_k: List[int] = field(default_factory=lambda: [1, 5, 10])
    ndcg_k: List[int] = field(default_factory=lambda: [10])
    mrr: bool = True
    dewi_bins: List[float] = field(default_factory=lambda: [0.0, 0.33, 0.66, 1.0])

@dataclass
class DewiConfig:
    """Top-level DEWI configuration."""
    text: TextConfig = field(default_factory=TextConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    cross_modal: CrossModalConfig = field(default_factory=CrossModalConfig)
    redundancy: RedundancyConfig = field(default_factory=RedundancyConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    training_sampling: TrainingSamplingConfig = field(default_factory=TrainingSamplingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'DewiConfig':
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DewiConfig':
        """Create a DewiConfig from a dictionary."""
        # Handle nested dataclass construction
        def create_instance(klass, d):
            if d is None:
                return klass()
            fields = {f.name for f in dataclasses.fields(klass) if f.init}
            filtered = {k: v for k, v in d.items() if k in fields}
            return klass(**filtered)
        
        # Create config with proper nesting
        return cls(
            text=create_instance(TextConfig, data.get('text')),
            image=create_instance(ImageConfig, data.get('image')),
            cross_modal=create_instance(CrossModalConfig, data.get('cross_modal')),
            redundancy=create_instance(RedundancyConfig, data.get('redundancy')),
            noise=create_instance(NoiseConfig, data.get('noise')),
            scoring=create_instance(ScoringConfig, data.get('scoring')),
            index=create_instance(IndexConfig, data.get('index')),
            training_sampling=create_instance(TrainingSamplingConfig, data.get('training_sampling')),
            eval=create_instance(EvalConfig, data.get('eval')),
        )
    
    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        def asdict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: asdict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [asdict(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: asdict(v) for k, v in obj.items()}
            else:
                return obj
        
        return asdict(self)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save the configuration to a YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

# Default configuration
default_config = DewiConfig()

def get_default_config() -> DewiConfig:
    """Get a deep copy of the default configuration."""
    import copy
    return copy.deepcopy(default_config)

# Import dataclasses after all classes are defined
import dataclasses
