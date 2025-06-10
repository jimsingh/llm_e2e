import pytest
import torch
from llm_e2e.config import GPT2Config
from llm_e2e.model import GPT2Model, MultiHeadAttention, TransformerBlock

@pytest.fixture
def sample_config():
    return GPT2Config(
        vocab_size=1000,
        emb_dim=64,
        n_heads=4,
        n_layers=2,
        dropout_rate=0.1,
        batch_size=4,
        context_length=32,
        qkv_bias=False
    )

def test_multi_head_attention(sample_config):
    attn = MultiHeadAttention(
        d_in=sample_config.emb_dim,
        d_out=sample_config.emb_dim,
        context_length=sample_config.context_length,
        num_heads=sample_config.n_heads,
        dropout=sample_config.dropout_rate,
        qkv_bias=sample_config.qkv_bias
    )
    batch_size = 4
    seq_length = 32
    x = torch.randn(batch_size, seq_length, sample_config.emb_dim)
    
    # Test forward pass
    output, attn_weights = attn(x)
    assert output.shape == (batch_size, seq_length, sample_config.emb_dim)
    assert attn_weights.shape == (batch_size, sample_config.n_heads, seq_length, seq_length)
    
    # Test attention mask
    # The output should be different for different input sequences
    x2 = torch.randn_like(x)
    output2, _ = attn(x2)
    assert not torch.allclose(output, output2)

def test_transformer_block(sample_config):
    block = TransformerBlock(sample_config)
    batch_size = 4
    seq_length = 32
    x = torch.randn(batch_size, seq_length, sample_config.emb_dim)
    
    # Test forward pass
    output = block(x)
    assert output.shape == (batch_size, seq_length, sample_config.emb_dim)
    
    # Test residual connection
    # The output should be different from input due to the transformations
    assert not torch.allclose(output, x)

def test_gpt_model(sample_config):
    model = GPT2Model(sample_config)
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
    x_long = torch.randint(0, sample_config.vocab_size, (batch_size, sample_config.context_length + 1))
    with pytest.raises(IndexError):
        model(x_long)

def test_gpt_weight_initialization(sample_config):
    model = GPT2Model(sample_config)
    
    # Check embedding weights
    assert torch.allclose(model.token_embedding.weight.mean(), torch.tensor(0.0), atol=0.1)
    assert torch.allclose(model.position_embeddings.weight.mean(), torch.tensor(0.0), atol=0.1)
    
    # Check linear layer weights
    for block in model.transformer_blocks:
        assert torch.allclose(block.att.W_query.weight.mean(), torch.tensor(0.0), atol=0.1)
        assert torch.allclose(block.att.W_key.weight.mean(), torch.tensor(0.0), atol=0.1)
        assert torch.allclose(block.att.W_value.weight.mean(), torch.tensor(0.0), atol=0.1)
        assert torch.allclose(block.att.out_proj.weight.mean(), torch.tensor(0.0), atol=0.1)
        assert torch.allclose(block.ff.expansion.weight.mean(), torch.tensor(0.0), atol=0.1)
        assert torch.allclose(block.ff.projection.weight.mean(), torch.tensor(0.0), atol=0.1)

def test_gpt_gradient_flow(sample_config):
    model = GPT2Model(sample_config)
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
