import torch
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

@dataclass
class TextEntropyResult:
    token_entropies: List[float]  # Per-token negative log-likelihoods
    mean_entropy: float
    q90_entropy: float
    q99_entropy: float

class TextEntropyEstimator:
    """Estimates text entropy using a pre-trained language model."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        domain_model_name: Optional[str] = None,
    ):
        """Initialize the text entropy estimator.
        
        Args:
            model_name: Name of the base language model to use.
            device: Device to run the model on ('cuda' or 'cpu').
            domain_model_name: Optional domain-specific model for relative entropy.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.domain_model = None
        if domain_model_name:
            self.domain_model = AutoModelForCausalLM.from_pretrained(domain_model_name).to(self.device)
            self.domain_model.eval()
    
    def compute_entropy(
        self,
        text: str,
        relative_to_domain: bool = False,
    ) -> TextEntropyResult:
        """Compute token-level and aggregate entropy metrics for the input text.
        
        Args:
            text: Input text to compute entropy for.
            relative_to_domain: If True, compute entropy relative to domain model.
            
        Returns:
            TextEntropyResult containing token and aggregate entropy metrics.
        """
        # Tokenize and prepare inputs
        inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Get token-level logits
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Compute token-level cross-entropy
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_entropies = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # If using domain-relative entropy, compute domain model's entropy
            if relative_to_domain and self.domain_model is not None:
                domain_outputs = self.domain_model(input_ids, attention_mask=attention_mask)
                domain_logits = domain_outputs.logits[..., :-1, :].contiguous()
                domain_entropies = loss_fct(
                    domain_logits.view(-1, domain_logits.size(-1)),
                    shift_labels.view(-1)
                )
                # Subtract domain entropy from base model entropy
                token_entropies = token_entropies - domain_entropies
        
        # Convert to numpy for further processing
        token_entropies = token_entropies.cpu().numpy()
        
        # Compute aggregate statistics
        mean_entropy = float(np.mean(token_entropies))
        q90_entropy = float(np.quantile(token_entropies, 0.9))
        q99_entropy = float(np.quantile(token_entropies, 0.99))
        
        return TextEntropyResult(
            token_entropies=token_entropies.tolist(),
            mean_entropy=mean_entropy,
            q90_entropy=q90_entropy,
            q99_entropy=q99_entropy,
        )
    
    def batch_compute(
        self,
        texts: List[str],
        relative_to_domain: bool = False,
        batch_size: int = 8,
    ) -> List[TextEntropyResult]:
        """Compute entropy for a batch of texts.
        
        Args:
            texts: List of input texts.
            relative_to_domain: If True, compute entropy relative to domain model.
            batch_size: Batch size for processing.
            
        Returns:
            List of TextEntropyResult objects, one per input text.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend([self.compute_entropy(text, relative_to_domain) for text in batch])
        return results

    def __call__(self, text: str, **kwargs) -> TextEntropyResult:
        """Alias for compute_entropy for easier function-like usage."""
        return self.compute_entropy(text, **kwargs)
