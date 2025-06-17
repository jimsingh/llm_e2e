
import argparse
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import tiktoken
from datasets import load_dataset

from llm_e2e import GPT2Config, GPT2Model
import argparse
import os
import sys
from collections import Counter  # Add this line
from collections.abc import Iterator
from dataclasses import dataclass

@dataclass
class EvalConfig:
    """evaluation configuration"""
    device: str = 'cpu'
    max_seq_len: int = 128
    log_interval: int = 100
    debug: bool = False
    text_chunk_target_size: int = 300 # add this line

class PerplexityEvaluator:
    """evaluates perplexity on text datasets"""

    def __init__(self, model: GPT2Model, tokenizer, config: EvalConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.model.eval()
        self.model.to(config.device)

        # limit punctuation and letters to what we have in training
        self.allowed_punctuation = set("""! . , ? ' " : ; -`.""")
        self._allowed_chars = {chr(c) for c in range(ord('a'), ord('z') + 1)}
        self._allowed_chars.update(chr(c) for c in range(ord('0'), ord('9') + 1))
        self._allowed_chars.update(self.allowed_punctuation)
        self._allowed_chars.add(' ')
        self.is_allowed_char = lambda c: c in self._allowed_chars

    def _preprocess_text(self, text: str) -> str:
        """
        preprocesses text to match training corpus by converting
        to lowercase and removing punctuation.
        """
        text = text.lower()
        text = "".join(char for char in text if self.is_allowed_char(char))
        return text

    def evaluate(self, dataset_name: str, split: str = "test", max_samples: int | None = None, streaming: bool = False) -> dict:
        """evaluate perplexity on a dataset, document by document."""
        print(f"loading dataset: {dataset_name} (split={split}, streaming={streaming})")
        dataset = self._load_dataset(dataset_name, split, streaming)

        total_loss = 0.0
        total_tokens = 0
        samples_processed = 0
        skipped_samples = 0

        print(f"evalating (max_samples={max_samples or 'all'})")

        with torch.no_grad():
            for i, text in enumerate(self._iterate_texts(dataset)):
                if max_samples and samples_processed >= max_samples:
                    break
                
                if len(text.strip()) < 10:
                    skipped_samples += 1
                    continue
                
                samples_processed += 1
                
                document_tokens = self.tokenizer.encode(text, allowed_special="all")
                
                for chunk_tokens in self._chunk_document(document_tokens):
                    
                    if len(chunk_tokens) < 2:
                        continue
                    
                    input_tensor = torch.tensor([chunk_tokens], device=self.cfg.device)
                    loss, n_predictions = self._compute_loss(input_tensor)

                    if n_predictions > 0:
                        total_loss += loss * n_predictions
                        total_tokens += n_predictions

                if self.cfg.debug and samples_processed <= 3:
                    self._print_debug_info(samples_processed, text, torch.tensor([document_tokens]), total_tokens, total_loss / max(1, total_tokens))

                if (samples_processed % self.cfg.log_interval == 0) and (total_tokens > 0):
                    self._print_progress(i, samples_processed, total_tokens, total_loss, skipped_samples)

        print(f"eval complete: {samples_processed} documents, {skipped_samples} skipped")

        return self._calculate_final_metrics(total_loss, total_tokens, samples_processed, skipped_samples)

    def _chunk_document(self, tokens: list[int]) -> Iterator[list[int]]:
        """yields chunks of a tokenized document with overlap."""
        stride = self.cfg.max_seq_len // 2
        for i in range(0, len(tokens), stride):
            chunk = tokens[i : i + self.cfg.max_seq_len]
            if len(chunk) > 1:
                yield chunk

    def _compute_loss(self, input_ids: torch.Tensor) -> tuple[float, int]:
        """ computes the average cross-entropy loss for the sequence. """
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:] # standard next token prediction
        num_predictions = targets.shape[1]

        if num_predictions == 0:
            return 0.0, 0

        # model will calculate loss
        _, loss = self.model(inputs, targets)
        return loss.item(), num_predictions

    def _load_dataset(self, dataset_name: str, split: str, streaming: bool):
        """use hugging face data loaders """
        if ":" in dataset_name:
            path, config_name = dataset_name.split(":", 1)
            dataset = load_dataset(path, config_name, split=split, streaming=streaming, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming, trust_remote_code=True)
        return dataset.shuffle(seed=1234, buffer_size=100_000)

    def _iterate_texts(self, dataset) -> Iterator[str]:
        """
        Pulls data from the dataset in a consistent way so that datasets with long input sequences are split up
        so that they match the typical token lengths we trained on.
        """
        text_buffer = []
        current_chunk_size = 0
        target_size = self.cfg.text_chunk_target_size

        for example in dataset:
            # this chain of gets accounts for different text keys in datasets
            raw_text = example.get('highlights',
                                   example.get('sentence',
                                               example.get('text',
                                                        example.get('content', ''))))
            if not raw_text:
                continue

            # split by paragraph to handle long articles gracefully
            paragraphs = raw_text.split('\n\n')

            for paragraph in paragraphs:
                # remove funny punctuation and lowercase text
                processed_paragraph = self._preprocess_text(paragraph.strip())

                # skip empty or very short paragraphs
                if len(processed_paragraph.split()) < 10:
                    continue

                # add the new paragraph to the buffer
                text_buffer.append(processed_paragraph)
                current_chunk_size += len(processed_paragraph)

                # if the buffer is full, yield the sample
                if current_chunk_size >= target_size:
                    yield " ".join(text_buffer)
                    text_buffer = []
                    current_chunk_size = 0

        # after the loop finishes, yield any remaining text in the buffer
        if text_buffer:
            yield " ".join(text_buffer)

    def _print_debug_info(self, sample_num, text, tokens, n_predictions, loss):
        """output a single sample for debugging"""
        print(f"\n[debug] sample {sample_num}:")
        print(f"  text preview: {text[:1000]}...")
        print(f"  total tokens: {tokens.shape[1]}, predictions: {n_predictions}, loss: {loss:.4f}")

    def _print_progress(self, i, samples, total_tokens, total_loss, skipped):
        current_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        avg_tokens = total_tokens / samples
        print(f"[{i+1:5d}] documents: {samples:4d}, ppl: {current_ppl:7.2f}, "
              f"avg_tokens/doc: {avg_tokens:.1f}, skipped: {skipped}")

    def _calculate_final_metrics(self, total_loss, total_tokens, samples, skipped):
        """calculates and returns the final evaluation metrics"""
        perplexity = avg_loss = float('inf')
        if total_tokens > 0:
            perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
            avg_loss = total_loss / total_tokens

        return {
            "perplexity": perplexity,
            "loss": avg_loss,
            "samples": samples,
            "tokens": total_tokens,
            "skipped": skipped,
            "avg_tokens_per_sample": total_tokens / samples
        }

def run_eval(args, tasks_to_run):
    print(f"model: {args.model}")
    print(f"config: {args.config}")

    cfg = GPT2Config.from_yaml(args.config)
    eval_cfg = EvalConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_seq_len=cfg.context_length,
        debug=args.debug
    )

    print(f"context length: {cfg.context_length}")
    print(f"device: {eval_cfg.device}")

    tokenizer = tiktoken.get_encoding(cfg.encoding_name)
    model = GPT2Model(cfg)
    model.load_parameters(args.model, model_key='model_state_dict', map_location=torch.device(eval_cfg.device))
    evaluator = PerplexityEvaluator(model, tokenizer, eval_cfg)

    results = {}
    for task_name, task_config in tasks_to_run.items():
        print(f"\n{'='*20} {task_name.upper()} {'='*20}")
        results[task_name] = evaluator.evaluate(**task_config)

    print_results(results)

def print_results(results: dict):
    """pretty print evaluation results"""
    print("\neval results:")
    print("\n")
    for task, metrics in results.items():
        print(f"\n[{task.upper()}]")
        print("-" * (len(task) + 2))
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    print("\n")


def main():
    """parses arguments and runs the evaluation"""
    parser = argparse.ArgumentParser(description='evaluate model perplexity.')
    parser.add_argument('--model', default='models/llm_e2e/gpt2_33M_cleancorpus/hahrukhx01_wikipedia-bookscorpus-en-preprocessed.33633024.pth', help='path to model checkpoint')
    parser.add_argument('--config', default='config/gpt2_bert_corpus_gpu.yaml', help='path to config file')
    parser.add_argument('--tasks', nargs='*', help='tasks to run (training-corpus, wikitext-103, c4, simple-wikipedia)', required=True)
    parser.add_argument('--max-samples', type=int, default=100, help='maximum samples for custom dataset or config default')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    args = parser.parse_args()

    all_tasks = {
        'training-corpus': {'dataset_name': 'shahrukhx01/wikipedia-bookscorpus-en-preprocessed', 'split': 'train', 'max_samples': 1000, 'streaming': True},
        'wikitext-103': {'dataset_name': 'wikitext:wikitext-103-v1', 'split': 'test', 'max_samples': 1000},
        'simple-wikipedia': {'dataset_name': 'wikimedia/wikipedia:20231101.simple', 'split': 'train', 'max_samples': 1000, 'streaming': True},
        'c4': {'dataset_name': 'c4:en', 'split': 'train', 'max_samples': 50, 'streaming': True}
    }
    tasks_to_run = {name: all_tasks[name] for name in args.tasks if name in all_tasks}

    run_eval(args, tasks_to_run)

if __name__ == "__main__":
    main()
