import pytest
import torch
from llm_e2e.model import GPT2Config, MultiHeadAttention, TransformerBlock

@pytest.fixture
def sample_config():
    return GPT2Config(
        vocab_size=1000,
        n_embd=64,
        n_head=4,
        n_layer=2,
        n_positions=128,
        dropout=0.1,
        batch_size=4,
        context_length=32
    )

def test_multi_head_attention(sample_config):
    attn = MultiHeadAttention(sample_config)
    batch_size = 4
    seq_length = 32
    x = torch.randn(batch_size, seq_length, sample_config.n_embd)
    
    # Test forward pass
    output = attn(x)
    assert output.shape == (batch_size, seq_length, sample_config.n_embd)
    
    # Test attention mask
    # The output should be different for different input sequences
    x2 = torch.randn_like(x)
    output2 = attn(x2)
    assert not torch.allclose(output, output2)

def test_block(sample_config):
    block = Block(sample_config)
    batch_size = 4
    seq_length = 32
    x = torch.randn(batch_size, seq_length, sample_config.n_embd)
    
    # Test forward pass
    output = block(x)
    assert output.shape == (batch_size, seq_length, sample_config.n_embd)
    
    # Test residual connection
    # The output should be different from input due to the transformations
    assert not torch.allclose(output, x)

def test_gpt2_model(sample_config):
    model = GPT2(sample_config)
    batch_size = 4
    seq_length = 32
    
    # Test forward pass without targets
    x = torch.randint(0, sample_config.vocab_size, (batch_size, seq_length))
    logits, loss = model(x)
    assert logits.shape == (batch_size, seq_length, sample_config.vocab_size)
    assert loss is None
    
    # Test forward pass with targets
    targets = torch.randint(0, sample_config.vocab_size, (batch_size, seq_length))
    logits, loss = model(x, targets)
    assert logits.shape == (batch_size, seq_length, sample_config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # scalar loss
    
    # Test sequence length constraint
    with pytest.raises(AssertionError):
        x_long = torch.randint(0, sample_config.vocab_size, (batch_size, sample_config.n_positions + 1))
        model(x_long)

def test_gpt2_weight_initialization(sample_config):
    model = GPT2(sample_config)
    
    # Check embedding weights
    assert torch.allclose(model.wte.weight.mean(), torch.tensor(0.0), atol=0.1)
    assert torch.allclose(model.wpe.weight.mean(), torch.tensor(0.0), atol=0.1)
    
    # Check linear layer weights
    for block in model.blocks:
        assert torch.allclose(block.attn.c_attn.weight.mean(), torch.tensor(0.0), atol=0.1)
        assert torch.allclose(block.attn.c_proj.weight.mean(), torch.tensor(0.0), atol=0.1)
        assert torch.allclose(block.mlp[0].weight.mean(), torch.tensor(0.0), atol=0.1)
        assert torch.allclose(block.mlp[2].weight.mean(), torch.tensor(0.0), atol=0.1)

def test_gpt2_gradient_flow(sample_config):
    model = GPT2(sample_config)
    batch_size = 4
    seq_length = 32
    
    # Create input and targets
    x = torch.randint(0, sample_config.vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, sample_config.vocab_size, (batch_size, seq_length))
    
    # Forward pass
    logits, loss = model(x, targets)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
        assert not torch.isinf(param.grad).any() 
