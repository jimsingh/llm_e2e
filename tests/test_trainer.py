import pytest
import torch
import os
from unittest.mock import MagicMock, patch

from llm_e2e.config import GPT2Config
from llm_e2e.model import GPT2Model
from llm_e2e.trainer import GPT2Trainer, TrainerState


@pytest.fixture
def mock_loader(sample_config):
    """creates a mock data loader that yields a single batch of random data."""
    def loader():
        x = torch.randint(0, sample_config.vocab_size, (sample_config.batch_size, sample_config.context_length))
        y = torch.randint(0, sample_config.vocab_size, (sample_config.batch_size, sample_config.context_length))
        return [(x, y)]
    return loader


@pytest.fixture
def small_loader(small_config):
    """creates a mock data loader that yields a single batch of random data."""
    def loader():
        for _ in range(small_config.num_epochs):
            x = torch.randint(0, small_config.vocab_size, (small_config.batch_size, small_config.context_length))
            y = torch.randint(0, small_config.vocab_size, (small_config.batch_size, small_config.context_length))
            yield x, y
    return loader


@pytest.fixture
def mock_logger():
    """a mock logger that does nothing."""
    return MagicMock()


@pytest.fixture
def trainer(sample_config, mock_loader, mock_logger):
    """a gpt2trainer instance for testing."""
    model = GPT2Model(sample_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=sample_config.learning_rate)
    scheduler = None  # No scheduler for testing
    
    # patch the loader to avoid StopIteration
    with patch('itertools.islice', return_value=mock_loader()):
        yield GPT2Trainer(
            cfg=sample_config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=mock_loader(),
            val_loader=mock_loader(),
            logger=mock_logger
        )


def test_trainer_initialization(trainer, sample_config):
    """test that the trainer is initialized correctly."""
    assert isinstance(trainer.state, TrainerState)
    assert trainer.state.step == 0
    assert trainer.state.epoch == 0
    assert trainer.state.best_val_loss == float('inf')
    assert trainer.state.config == vars(sample_config)


def test_train_batch(trainer):
    """test that a single training batch updates model parameters."""
    initial_params = {name: p.clone() for name, p in trainer.model.named_parameters()}
    
    x, y = next(iter(trainer.train_loader))
    loss = trainer._train_batch(x, y)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    
    # check that parameters have been updated
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            assert not torch.equal(initial_params[name], param)


def test_estimate_loss(trainer):
    """test that estimate_loss returns a valid loss value."""
    loss = trainer._estimate_loss(trainer.val_loader)
    assert isinstance(loss, float)
    assert loss > 0


def test_save_and_load_checkpoint(trainer, tmp_path):
    """test that saving and loading a checkpoint restores trainer state."""
    # modify the state
    trainer.state.step = 100
    trainer.state.epoch = 1
    trainer.state.best_val_loss = 0.5
    
    checkpoint_path = tmp_path / "checkpoint.pth"
    trainer.save_checkpoint(str(checkpoint_path))
    
    assert os.path.exists(checkpoint_path)
    
    # create a new trainer and load the checkpoint
    new_trainer = GPT2Trainer(
        cfg=trainer.cfg,
        model=GPT2Model(trainer.cfg),
        optimizer=torch.optim.AdamW(trainer.model.parameters(), lr=trainer.cfg.learning_rate),
        scheduler=None,
        train_loader=trainer.train_loader,
        val_loader=trainer.val_loader,
        logger=MagicMock()
    )
    
    new_trainer.load_checkpoint(str(checkpoint_path))
    
    assert new_trainer.state.step == trainer.state.step
    assert new_trainer.state.epoch == trainer.state.epoch
    assert new_trainer.state.best_val_loss == trainer.state.best_val_loss


def test_train_e2e_gradient_updates(small_config, small_loader, mock_logger, tmp_path):
    """
    test end-to-end training for one epoch and verify that model parameters
    are updated.
    """
    small_config.num_epochs = 1
    small_config.eval_interval = 1
    small_config.log_interval = 1
    small_config.save_filename = str(tmp_path / "test_model.pth")

    model = GPT2Model(small_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=small_config.learning_rate)

    trainer = GPT2Trainer(
        cfg=small_config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        train_loader=small_loader(),
        val_loader=small_loader(),
        logger=mock_logger
    )

    initial_params = {name: p.clone() for name, p in trainer.model.named_parameters()}

    trainer.train()

    # check that parameters have been updated
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            assert not torch.equal(initial_params[name], param), f"parameter {name} was not updated."


def test_max_steps_stopping_condition(sample_config, mock_loader, mock_logger, tmp_path):
    """
    Test that training stops when max_steps is reached, regardless of epoch completion.
    """
    # Configure for max_steps testing
    sample_config.max_steps = 3  # Stop after 3 steps
    sample_config.num_epochs = 10  # More epochs than needed
    sample_config.eval_interval = 5  # Won't trigger evaluation
    sample_config.log_interval = 1
    sample_config.save_filename = str(tmp_path / "test_model.pth")
    
    model = GPT2Model(sample_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=sample_config.learning_rate)
    
    trainer = GPT2Trainer(
        cfg=sample_config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        train_loader=mock_loader(),
        val_loader=mock_loader(),
        logger=mock_logger
    )
    
    # Record initial step
    initial_step = trainer.state.step
    assert initial_step == 0
    
    # Run training
    trainer.train()
    
    # Verify that training stopped at max_steps
    assert trainer.state.step == sample_config.max_steps, f"Expected {sample_config.max_steps} steps, got {trainer.state.step}"
    
    # Verify that we didn't complete all epochs (since we stopped early)
    assert trainer.state.epoch < sample_config.num_epochs, "Should have stopped before completing all epochs"


def test_max_steps_with_evaluation_trigger(sample_config, mock_loader, mock_logger, tmp_path):
    """
    Test that max_steps stopping works correctly even when evaluation is triggered.
    """
    # Configure for testing max_steps with evaluation
    sample_config.max_steps = 5  # Stop after 5 steps
    sample_config.num_epochs = 10
    sample_config.eval_interval = 2  # Will trigger evaluation at step 2 and 4
    sample_config.log_interval = 1
    sample_config.save_filename = str(tmp_path / "test_model.pth")
    
    model = GPT2Model(sample_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=sample_config.learning_rate)
    
    trainer = GPT2Trainer(
        cfg=sample_config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        train_loader=mock_loader(),
        val_loader=mock_loader(),
        logger=mock_logger
    )
    
    # Run training
    trainer.train()
    
    # Verify that training stopped at max_steps
    assert trainer.state.step == sample_config.max_steps, f"Expected {sample_config.max_steps} steps, got {trainer.state.step}"


def test_max_steps_exact_boundary(sample_config, mock_loader, mock_logger, tmp_path):
    """
    Test that max_steps works correctly when set to exactly 1 step.
    """
    # Configure for single step training
    sample_config.max_steps = 1  # Stop after exactly 1 step
    sample_config.num_epochs = 5
    sample_config.eval_interval = 10  # Won't trigger evaluation
    sample_config.log_interval = 1
    sample_config.save_filename = str(tmp_path / "test_model.pth")
    
    model = GPT2Model(sample_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=sample_config.learning_rate)
    
    trainer = GPT2Trainer(
        cfg=sample_config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        train_loader=mock_loader(),
        val_loader=mock_loader(),
        logger=mock_logger
    )
    
    # Run training
    trainer.train()
    
    # Verify that training stopped at exactly 1 step
    assert trainer.state.step == 1, f"Expected 1 step, got {trainer.state.step}"
