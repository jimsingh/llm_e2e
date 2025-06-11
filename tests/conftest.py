import pytest
import torch
from llm_e2e import GPT2Config
import tiktoken

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding('gpt2')

@pytest.fixture
def sample_config():
    """A generic config for testing with reasonable values."""
    cfg = GPT2Config(
        vocab_size=1000,
        emb_dim=64,
        n_heads=4,
        n_layers=2,
        dropout_rate=0.1,
        batch_size=4,
        context_length=32,
        qkv_bias=False,
        device='cpu',
        learning_rate=1e-4,
        weight_decay=0.01,
        num_epochs=2,
        wandb_log=False,
        wandb_project='llm_e2e',
        wandb_run_name=None
    )
    # Set extra attributes not in the constructor
    cfg.dataset_path = "wikimedia/wikipedia"
    cfg.dataset_name = "20231101.simple"
    cfg.shuffle = True
    cfg.streaming = True
    cfg.save_filename = 'test_model.pth'
    return cfg

@pytest.fixture
def small_config():
    """A minimal config for testing with small values."""
    cfg = GPT2Config()
    cfg.batch_size = 2
    cfg.seq_len = 4
    cfg.n_embd = 32
    cfg.n_head = 2
    cfg.n_layer = 1
    cfg.dropout = 0.1
    cfg.learning_rate = 1e-4
    cfg.weight_decay = 0.01
    cfg.num_epochs = 2
    cfg.device = 'cpu'
    cfg.save_filename = 'test_model.pth'
    return cfg 
