"""
Wrapper for hugging face datasets. Stream, shuffle, and generate batches setup for
next token prediction.
"""

import torch
from datasets import load_dataset, Dataset
from typing import Iterator, Any
from .config import GPT2Config



class StreamingDatasetGenerator:    
    """load a huggingface dataset and wrap it as a generator"""
    def __init__(self, cfg: GPT2Config, encoding: Any, split: str = 'train', seed: int = 42):
        self.counter = 0
        self.seed = seed
        self.split = split
        self.cfg = cfg
        self.encoding = encoding
        dataset = load_dataset(
            path=self.cfg.dataset_path,
            name=self.cfg.dataset_name,
            split=split,
            streaming=True
        )
        self.dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Reset the generator and return an iterator that yields batches of data.
        
        Returns:
            Iterator yielding (X, Y) tensor pairs for training
        """
        self.counter += 1
        seed = self.seed + self.counter
        return self.get_dataset_batch(
            batch_size=self.cfg.batch_size,
            seq_len=self.cfg.context_length,
            seed=seed
        ) 

    def get_dataset_batch(self, batch_size: int, seq_len: int, seed: int = 42) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Not iterator interface - access batches from the configured dataset 
        
        Args:
            batch_size: Number of sequences per batch
            seq_len: Length of each sequence
            seed: Random seed for shuffling
            
        Yields:
            Tuples of (X, Y) tensors where:
            - X has shape (batch_size, seq_len)
            - Y has shape (batch_size, seq_len)
            Y is shifted by 1 position from X for next-token prediction
        """
        # Token accumulator
        all_tokens = []

        for doc in self.dataset:
            # text -> token ids
            tokens = self.encoding.encode_ordinary(doc["text"])
            tokens.append(self.encoding.eot_token)
            all_tokens.extend(tokens)

            # only yield full batches
            while len(all_tokens) >= batch_size * (seq_len + 1):
                # extract batch worth of tokens
                batch_tokens = all_tokens[:batch_size * (seq_len + 1)]
                all_tokens = all_tokens[batch_size * (seq_len + 1):]

                # to tensor and appropriate shape 
                batch_tokens = torch.tensor(batch_tokens, dtype=torch.long)
                batch_tokens = batch_tokens.view(batch_size, seq_len + 1)

                # X, Y split for next-token prediction
                X = batch_tokens[:, :-1]  
                Y = batch_tokens[:, 1:] 

                yield X, Y
