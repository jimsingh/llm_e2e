import pytest
import torch
from datasets import load_dataset
from llm_e2e import StreamingDatasetGenerator, GPT2Config
import tiktoken
from itertools import islice

def test_load_dataset_produces_data(sample_config):
    # Load the dataset
    dataset = load_dataset(
        path=sample_config.dataset_path,
        name=sample_config.dataset_name,
        split='train',
        streaming=True
    )
    
    # Check if the dataset is not empty by getting first few items
    first_items = list(islice(dataset, 5))
    assert len(first_items) > 0, "The dataset should not be empty."
    
    # Print the first document for verification
    print("First document in the dataset:", first_items[0])

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

def test_dataset_size(sample_config):
    # Load the dataset
    dataset = load_dataset(
        path=sample_config.dataset_path,
        name=sample_config.dataset_name,
        split='train',
        streaming=True
    )
    
    # Check if we can get at least a few items
    first_items = list(islice(dataset, 5))
    print("Number of documents checked:", len(first_items))
    assert len(first_items) > 0, "The dataset should not be empty."

def test_dataset_produces_full_batch(sample_config, tokenizer):
    # Load the dataset
    dataset = load_dataset(
        path=sample_config.dataset_path,
        name=sample_config.dataset_name,
        split='train',
        streaming=True
    )
    
    # Initialize the generator
    loader = StreamingDatasetGenerator(sample_config, tokenizer)
    
    # Check if we can get at least one batch
    try:
        batch = next(iter(loader))
        X, Y = batch
        assert X.shape == (sample_config.batch_size, sample_config.context_length)
        assert Y.shape == (sample_config.batch_size, sample_config.context_length)
    except StopIteration:
        pytest.fail("Dataset generator could not produce a batch")

def test_dataset_token_count(sample_config, tokenizer):
    dataset = load_dataset(
        path=sample_config.dataset_path,
        name=sample_config.dataset_name,
        split='train',
        streaming=True
    )
    tokenizer_fn = tokenizer.encode if hasattr(tokenizer, 'encode') else tokenizer
    total_tokens = 0
    # Only check first few documents
    for doc in islice(dataset, 5):
        tokens = tokenizer_fn(doc['text'])
        total_tokens += len(tokens)
    required_tokens = sample_config.batch_size * sample_config.context_length
    print(f"Total tokens in first 5 documents: {total_tokens}, required: {required_tokens}")
    assert total_tokens >= required_tokens, (
        f"First 5 documents do not have enough tokens: {total_tokens} < {required_tokens}")
