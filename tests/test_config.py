import pytest
import os
from llm_e2e import GPT2Config

def test_config_initialization():
    cfg = GPT2Config()
    assert hasattr(cfg, 'batch_size')
    assert hasattr(cfg, 'context_length')
    assert hasattr(cfg, 'emb_dim')
    assert hasattr(cfg, 'n_heads')
    assert hasattr(cfg, 'n_layers')
    assert hasattr(cfg, 'dropout_rate')
    assert hasattr(cfg, 'learning_rate')
    assert hasattr(cfg, 'weight_decay')
    assert hasattr(cfg, 'num_epochs')
    assert hasattr(cfg, 'device')
    assert hasattr(cfg, 'save_filename')

def test_config_to_yaml(sample_config, tmp_path):
    # Create a temporary file
    yaml_file = tmp_path / "test_config.yaml"
    
    # Save config to YAML
    sample_config.to_yaml(str(yaml_file))
    
    # Check if file exists
    assert os.path.exists(yaml_file)
    
    # Load config from YAML
    loaded_cfg = GPT2Config().from_yaml(str(yaml_file))
    loaded_cfg.save_filename = sample_config.save_filename  # Override save_filename
    
    # Check if values match
    assert loaded_cfg.batch_size == sample_config.batch_size
    assert loaded_cfg.context_length == sample_config.context_length
    assert loaded_cfg.emb_dim == sample_config.emb_dim
    assert loaded_cfg.n_heads == sample_config.n_heads
    assert loaded_cfg.n_layers == sample_config.n_layers
    assert loaded_cfg.dropout_rate == sample_config.dropout_rate
    assert loaded_cfg.learning_rate == sample_config.learning_rate
    assert loaded_cfg.weight_decay == sample_config.weight_decay
    assert loaded_cfg.num_epochs == sample_config.num_epochs
    assert loaded_cfg.device == sample_config.device
    assert loaded_cfg.save_filename == sample_config.save_filename

def test_config_parameter_validation():
    # Test invalid batch size
    with pytest.raises(ValueError):
        GPT2Config(batch_size=-1)
    
    # Test invalid context length
    with pytest.raises(ValueError):
        GPT2Config(context_length=0)
    
    # Test invalid embedding dimension
    with pytest.raises(ValueError):
        GPT2Config(emb_dim=-64)
    
    # Test invalid number of heads
    with pytest.raises(ValueError):
        GPT2Config(n_heads=0)
    
    # Test invalid number of layers
    with pytest.raises(ValueError):
        GPT2Config(n_layers=-2)
    
    # Test invalid dropout
    with pytest.raises(ValueError):
        GPT2Config(dropout_rate=1.5)
    
    # Test invalid learning rate
    with pytest.raises(ValueError):
        GPT2Config(learning_rate=-1e-4)
    
    # Test invalid weight decay
    with pytest.raises(ValueError):
        GPT2Config(weight_decay=-0.01)
    
    # Test invalid number of epochs
    with pytest.raises(ValueError):
        GPT2Config(num_epochs=0)

def test_config_from_dict():
    config_dict = {
        'vocab_size': 1000,
        'emb_dim': 64,
        'n_heads': 4,
        'n_layers': 2,
        'dropout_rate': 0.1,
        'batch_size': 4,
        'context_length': 32,
        'qkv_bias': False,
        'dataset_path': 'test/path',
        'dataset_name': 'test_dataset',
        'device': 'cpu',
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 2,
        'wandb_log': False,
        'wandb_project': 'llm_e2e',
        'wandb_run_name': None
    }
    config = GPT2Config.from_dict(config_dict)
    assert config.vocab_size == 1000
    assert config.emb_dim == 64
    assert config.n_heads == 4
    assert config.n_layers == 2
    assert config.dropout_rate == 0.1
    assert config.batch_size == 4
    assert config.context_length == 32
    assert config.qkv_bias == False
    assert config.dataset_path == 'test/path'
    assert config.dataset_name == 'test_dataset'
    assert config.device == 'cpu'
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 0.01
    assert config.num_epochs == 2
    assert config.wandb_log == False
    assert config.wandb_project == 'llm_e2e'
    assert config.wandb_run_name is not None  # Should be set to 'train_' + timestamp
