import pytest
import torch
import wandb
from unittest.mock import patch, MagicMock
from llm_e2e.config import GPT2Config
from llm_e2e.logging import WandbLogger

@pytest.fixture
def mock_wandb():
    with patch('wandb.init') as mock_init, patch('wandb.watch') as mock_watch, patch('wandb.finish') as mock_finish, patch('wandb.log') as mock_log:
        yield {
            'init': mock_init,
            'watch': mock_watch,
            'finish': mock_finish,
            'log': mock_log
        }

@pytest.fixture
def sample_config():
    return GPT2Config(
        wandb_log=True,
        wandb_project='test_project',
        wandb_run_name='test_run',
        log_interval=10
    )

@pytest.fixture
def sample_model():
    return torch.nn.Linear(10, 10)

def test_wandb_logger_initialization(sample_config, sample_model, mock_wandb):
    logger = WandbLogger(sample_config, sample_model)
    assert logger.use_wandb == True
    mock_wandb['init'].assert_called_once_with(
        project='test_project',
        name='test_run',
        config={k: getattr(sample_config, k) for k, _ in sample_config.__annotations__.items()}
    )
    mock_wandb['watch'].assert_called_once_with(sample_model, log='all', log_freq=10)

def test_wandb_logger_context_manager(sample_config, sample_model, mock_wandb):
    with WandbLogger(sample_config, sample_model) as logger:
        assert logger.use_wandb == True
    mock_wandb['finish'].assert_called_once()

def test_wandb_logger_logging(sample_config, sample_model, mock_wandb):
    logger = WandbLogger(sample_config, sample_model)
    logger.log(step=1, loss=0.5, accuracy=0.9)
    mock_wandb['log'].assert_called_once_with(step=1, commit=True, loss=0.5, accuracy=0.9) 