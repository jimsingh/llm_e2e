"""
Wrapper for hugging face datasets. Stream, shuffle, and generate batches setup for
next token prediction.
"""

import hashlib
from typing import Any, Iterator
import torch
from datasets import load_dataset, IterableDataset
from .config import GPT2Config


class StreamingDatasetGenerator:
    """Loads a Hugging Face dataset and wraps it as a generator."""
    
    def __init__(self, cfg: GPT2Config, encoding: Any, split: str = 'train', 
                 dataset_split: str = 'train', seed: int = 42, val_frac: float = 0.1,
                 shuffle_buffer_size=100_000):
        self.counter = 0
        self.split = split
        self.cfg = cfg
        self.encoding = encoding
        self.seed = seed
        self.start_step = 0
        self.val_frac = val_frac
        self._get_text = lambda d: d if isinstance(d, str) else d['text']
        self.shuffle_buffer_size = shuffle_buffer_size 

        self.base_dataset: IterableDataset = load_dataset(
            path=self.cfg.dataset_path,
            name=self.cfg.dataset_name,
            split=dataset_split,
            streaming=True
        )

    def __iter__(self):
        """Reset for new epoch and return self as iterator."""
        self.counter += 1
        epoch_seed = self.seed + self.counter + self.start_step
        
        # Create shuffled dataset for this epoch with larger buffer
        self.dataset = self.base_dataset.shuffle(seed=epoch_seed, buffer_size=self.shuffle_buffer_size)
        
        # Reset batch generator
        if hasattr(self, '_batch_generator'):
            del self._batch_generator
            
        return self
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next batch from current epoch."""
        if not hasattr(self, '_batch_generator'):
            self._batch_generator = self._generate_batches()
        
        try:
            return next(self._batch_generator)
        except StopIteration:
            # Clean up for next epoch
            if hasattr(self, '_batch_generator'):
                del self._batch_generator
            raise
    
    def _generate_batches(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Generate batches from the shuffled dataset."""
        all_tokens = []
        max_buffer_size = self.cfg.batch_size * (self.cfg.context_length + 1) * 10
        
        for doc in self.dataset:
            txt = self._get_text(doc)
            if not self._should_include_doc(txt):
                continue

            # text -> token ids
            tokens = self.encoding.encode_ordinary(txt)
            tokens.append(self.encoding.eot_token)
            all_tokens.extend(tokens)
            
            # Yield batches as they become available
            while len(all_tokens) >= self.cfg.batch_size * (self.cfg.context_length + 1):
                yield self._create_batch(all_tokens)
    
    def _create_batch(self, all_tokens: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract and return one batch from token buffer."""
        batch_size = self.cfg.batch_size
        seq_len = self.cfg.context_length
        
        # Extract batch worth of tokens
        batch_tokens = all_tokens[:batch_size * (seq_len + 1)]
        del all_tokens[:batch_size * (seq_len + 1)]
        
        # Convert to tensor and reshape
        batch_tokens = torch.tensor(batch_tokens, dtype=torch.long)
        batch_tokens = batch_tokens.view(batch_size, seq_len + 1)
        
        # Split for next-token prediction
        X = batch_tokens[:, :-1]  # (batch_size, seq_len)
        Y = batch_tokens[:, 1:]   # (batch_size, seq_len)
        
        return X, Y
    
    def _should_include_doc(self, text: str) -> bool:
        """Decide if a document belongs to train or val based on hash of its content."""
        doc_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        hash_int = int(doc_hash, 16)
        hash_fraction = (hash_int % 10000) / 10000.0
        
        if self.split == 'train':
            return hash_fraction >= self.val_frac
        else:  # validation
            return hash_fraction < self.val_frac
