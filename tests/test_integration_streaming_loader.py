import pytest
import torch
from llm_e2e import StreamingDatasetGenerator, GPT2Config
import tiktoken

def test_streaming_data_loader_integration():
    # Use a small config for fast test
    cfg = GPT2Config()
    cfg.batch_size = 2
    cfg.context_length = 16
    cfg.dataset_path = "wikimedia/wikipedia"
    cfg.dataset_name = "20231101.simple"
    cfg.shuffle = True
    cfg.streaming = True

    tokenizer = tiktoken.get_encoding('gpt2')
    loader = StreamingDatasetGenerator(cfg, tokenizer, split='train', seed=123)

    # Collect several batches
    batches = []
    for i, (X, Y) in enumerate(loader):
        batches.append((X.clone(), Y.clone()))
        if i >= 2:
            break

    # Ensure we got multiple batches
    assert len(batches) >= 2

    # Ensure all batches are correct shape and type
    for X, Y in batches:
        assert isinstance(X, torch.Tensor)
        assert isinstance(Y, torch.Tensor)
        assert X.shape == (cfg.batch_size, cfg.context_length)
        assert Y.shape == (cfg.batch_size, cfg.context_length)

    # Check that Y is X shifted by one token for the first batch and first sequence
    X0, Y0 = batches[0][0][0], batches[0][1][0]  # First batch, first sequence
    X0_tokens = X0.cpu().tolist()
    Y0_tokens = Y0.cpu().tolist()
    # Y should be X shifted left by one, plus one new token at the end
    assert X0_tokens[1:] == Y0_tokens[:-1], "Y is not X shifted by one token."
    # Optionally, decode and print for visual inspection
    x_text = tokenizer.decode(X0_tokens)
    y_text = tokenizer.decode(Y0_tokens)
    print(f"Decoded X: {x_text}\nDecoded Y: {y_text}") 

def test_streaming_data_loader_shuffling_and_determinism():
    # Use a small config for fast test
    cfg = GPT2Config(
        batch_size=2,
        context_length=16,
        dataset_path="karpathy/tiny_shakespeare",
        dataset_name="tiny_shakespeare"
    )

    tokenizer = tiktoken.get_encoding('gpt2')

    # Create a loader with a specific seed and get the first batch
    loader1 = StreamingDatasetGenerator(cfg, tokenizer, split='train', seed=123)
    X1, Y1 = next(iter(loader1))

    # Create another loader with the same seed, it should produce the same batch
    loader2 = StreamingDatasetGenerator(cfg, tokenizer, split='train', seed=123)
    X2, Y2 = next(iter(loader2))

    assert torch.equal(X1, X2), "Batches with the same seed should be identical"
    assert torch.equal(Y1, Y2), "Batches with the same seed should be identical"

    # Create a third loader with a different seed, it should produce a different batch
    loader3 = StreamingDatasetGenerator(cfg, tokenizer, split='train', seed=456)
    X3, Y3 = next(iter(loader3))

    assert not torch.equal(X1, X3), "Batches with different seeds should not be identical"

def test_consecutive_batches_are_different(sample_config, tokenizer):
    """
    Tests that two consecutively drawn batches are not the same,
    which is expected with shuffling enabled.
    """
    loader = StreamingDatasetGenerator(sample_config, tokenizer, split='train')

    # get the first batch
    X1, _ = next(iter(loader))

    # get the second batch
    X2, _ = next(iter(loader))

    # assert that the two X tensors are not identical
    assert not torch.equal(X1, X2), "Two consecutive batches should not be the same."
