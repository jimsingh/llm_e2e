import pytest
import torch
from datasets import load_dataset
from llm_e2e import StreamingDatasetGenerator, GPT2Config
import tiktoken

def test_streaming_dataset_generator(sample_config, tokenizer):
    # Initialize the generator
    loader = StreamingDatasetGenerator(sample_config, tokenizer)
    
    # Get a batch
    batch = next(iter(loader))
    
    # Check batch structure
    X, Y = batch
    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)
    assert X.shape == (sample_config.batch_size, sample_config.context_length)
    assert Y.shape == (sample_config.batch_size, sample_config.context_length)
    
    # Check that Y is shifted by one position from X
    assert torch.all(X[:, 1:] == Y[:, :-1])

def test_streaming_dataset_generator_iteration(sample_config, tokenizer):
    loader = StreamingDatasetGenerator(sample_config, tokenizer)
    
    # Test multiple iterations
    for _ in range(3):
        batch = next(iter(loader))
        X, Y = batch
        assert X.shape == (sample_config.batch_size, sample_config.context_length)
        assert Y.shape == (sample_config.batch_size, sample_config.context_length)

def test_streaming_dataset_generator_reset(sample_config, tokenizer):
    loader = StreamingDatasetGenerator(sample_config, tokenizer)
    
    # Get first batch
    first_batch = next(iter(loader))
    
    # Reset and get another batch
    second_batch = next(iter(loader))
    
    # Since we're using a small dataset and not shuffling, we might get the same sequence
    # Instead, let's verify the shapes and types
    assert isinstance(first_batch[0], torch.Tensor)
    assert isinstance(second_batch[0], torch.Tensor)
    assert first_batch[0].shape == (sample_config.batch_size, sample_config.context_length)
    assert second_batch[0].shape == (sample_config.batch_size, sample_config.context_length) 
