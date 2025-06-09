import pytest
import torch
from llm_e2e import StreamingDatasetGenerator, GPT2Config
import tiktoken

def test_streaming_data_loader_integration():
    # Use a small config for fast test
    cfg = GPT2Config()
    cfg.batch_size = 2
    cfg.context_length = 16
    cfg.dataset_path = "karpathy/tiny_shakespeare"
    cfg.dataset_name = "tiny_shakespeare"
    cfg.shuffle = True
    cfg.streaming = True

    tokenizer = tiktoken.get_encoding('gpt2')
    loader = StreamingDatasetGenerator(cfg, tokenizer, split='train', seed=123)

    # Collect several batches
    batches = []
    for i, (X, Y) in enumerate(loader):
        batches.append((X.clone(), Y.clone()))
        if i >= 9:
            break

    # Ensure we got multiple batches
    assert len(batches) == 10

    # Ensure all batches are correct shape and type
    for X, Y in batches:
        assert isinstance(X, torch.Tensor)
        assert isinstance(Y, torch.Tensor)
        assert X.shape == (cfg.batch_size, cfg.context_length)
        assert Y.shape == (cfg.batch_size, cfg.context_length)

    # Ensure at least two batches are not identical (streaming/shuffling)
    unique_batches = set([X.cpu().numpy().tobytes() for X, _ in batches])
    assert len(unique_batches) > 1, "Streaming loader did not yield different batches."

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