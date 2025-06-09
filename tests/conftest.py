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
    cfg.save_filename = 'test_model.pt'
    return cfg 