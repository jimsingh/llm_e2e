from dataclasses import dataclass, field
import yaml
from typing import Any, ClassVar
import time

@dataclass
class GPT2Config:
    FIELD_SECTIONS: ClassVar[dict[str, list[str]]] = {
        "data": ["dataset_path", "dataset_name", "val_split"],
        "tokenizer": ["encoding_name", "eos_token_id"],
        "model": [
            "vocab_size", "context_length", "emb_dim",
            "n_heads", "n_layers", "dropout_rate", "qkv_bias", "mlp_bias"
        ],
        "training": [
            "num_epochs", "learning_rate", "weight_decay", "beta1", "beta2",
            "grad_clip", "warmup_steps", "max_steps", "batch_size",
            "eval_iters", "eval_interval", "eval_prompt", "save_interval"
        ],
        "system": ["device", "compile_model", "dtype"],
        "logging": ["wandb_log", "wandb_project", "wandb_run_name"],
    }

    # Data
    dataset_path: str                   = 'HuggingFaceFW/fineweb-edu'
    dataset_name: str                   = 'sample-10BT'
    val_split: str                      = None

    # tokenizer
    encoding_name: str                  = 'gpt2'
    eos_token_id: int                   = 50256 # token_id of '<|endoftext|>'
    
    # Model architecture
    vocab_size: int                     = 50257 # gpt2 BPE n_tokens
    context_length: int                 = 1024
    emb_dim: int                        = 768
    n_heads: int                        = 12
    n_layers: int                       = 12
    dropout_rate: float                 = 0.1
    qkv_bias: bool                      = False
    mlp_bias: bool                      = True 
    
    # Training hyperparameters
    num_epochs: int                     = 2
    learning_rate: float                = 7e-4
    weight_decay: float                 = 0.1
    beta1: float                        = 0.9
    beta2: float                        = 0.95
    grad_clip: float                    = 1.0
    warmup_steps: int                   = 10_000    
    max_steps: int                      = 300_000
    
    # Training setup
    batch_size: int                     = 8
    log_interval: int                   = 200
    eval_iters: int                     = 50
    eval_interval: int                  = 1_00
    eval_prompt: str                    = "The main reason why"
    save_interval: int                  = 10_000
    save_filename: str                  = field(init = False)
    
    # System
    device: str                         = 'cuda'
    compile_model: bool                 = True
    dtype: str                          = "bfloat16"

    # Logging
    wandb_log: bool                     = False
    wandb_project: str                  = "llm_e2e"
    wandb_run_name: str                 = ""
     
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'GPT2Config':
        """Load config from nested YAML structure."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # flatten nested structure
        flat_dict = {}
        if config_dict:
            for section, params in config_dict.items():
                if isinstance(params, dict):
                    flat_dict.update(params)
                else:
                    flat_dict[section] = params
            print(f"loaded config from: {yaml_path}")
        else:
            print(f"no yaml config found in {yaml_path}, using defaults")

        # remove block_size if present
        flat_dict.pop('block_size', None)
        return cls(**flat_dict)
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save config to nested YAML structure."""
        config_dict = {}
        for section, fields in self.FIELD_SECTIONS.items():
            config_dict[section] = {field: getattr(self, field) for field in fields}
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update_from_dict(self, updates: dict) -> 'GPT2Config':
        """Create new config with updated values."""
        config_dict = {k: v for k, v in self.__dict__.items()}
        config_dict.update(updates)
        return self.__class__(**config_dict)
    
    @property
    def n_params(self) -> int:
        """Estimate model parameters in millions."""
        # Token embedding: vocab_size * emb_dim
        token_emb = self.vocab_size * self.emb_dim
        
        # Position embedding: context_length * emb_dim
        pos_emb = self.context_length * self.emb_dim
        
        # Transformer blocks
        # Each block: 4 * emb_dim^2 (attention) + 8 * emb_dim^2 (mlp) = 12 * emb_dim^2
        transformer_blocks = self.n_layers * 12 * self.emb_dim**2

        # 2 LayerNorms per block (before attention, before MLP) + 1 final LayerNorm after transformer blocks.
        num_layernorms = (2 * self.n_layers) + 1
        
        # Each LayerNorm has 2 parameters (scale gamma, bias beta) per embedding dimension.
        layer_norms = num_layernorms * (2 * self.emb_dim)
        
        total_params = token_emb + pos_emb + transformer_blocks + layer_norms
        return total_params
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate batch_size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        # Validate context_length
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        # Validate emb_dim
        if self.emb_dim <= 0:
            raise ValueError(f"emb_dim must be positive, got {self.emb_dim}")
        # Validate n_heads
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        # Validate n_layers
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        # Validate dropout_rate
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {self.dropout_rate}")
        # Validate learning_rate (handle scientific notation)
        if isinstance(self.learning_rate, str):
            try:
                self.learning_rate = float(self.learning_rate)
            except ValueError:
                raise ValueError(f"learning_rate must be a valid number, got {self.learning_rate}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        # Validate weight_decay
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        # Validate num_epochs
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        # Now check divisibility
        assert self.emb_dim % self.n_heads == 0, f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
        self.save_filename = f"{self.dataset_path.replace('/','_')}.{self.n_params}.pth"
        # Set wandb_run_name if not provided
        if not self.wandb_run_name:
            self.wandb_run_name = f"{self.save_filename}_train_{time.time()}" 
