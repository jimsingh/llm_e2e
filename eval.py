# uv pip install datasets transformers tiktoken torch

import argparse
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import tiktoken
from datasets import load_dataset

from llm_e2e import GPT2Config, GPT2Model


@dataclass
class EvalConfig:
    """evaluation configuration"""
    device: str = 'cpu'
    max_seq_len: int = 128
    log_interval: int = 100
    debug: bool = False


class PerplexityEvaluator:
    """evaluates perplexity on text datasets"""

    def __init__(self, model: GPT2Model, tokenizer, config: EvalConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.model.eval()
        self.model.to(config.device)
        self.allowed_punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    def _preprocess_text(self, text: str) -> str:
        """
        preprocesses text to match the training corpus format by converting
        to lowercase and removing punctuation not in the allowed set.
        """
        # convert to lowercase
        text = text.lower()
        # filter out punctuation not present in the training corpus
        text = "".join(char for char in text if char.isalnum() or char.isspace() or char in self.allowed_punctuation)
        return text


    def evaluate(self, dataset_name: str, split: str = "test", max_samples: int | None = None, streaming: bool = False) -> dict:
        """evaluate perplexity on a dataset, document by document."""
        print(f"loading dataset: {dataset_name} (split={split}, streaming={streaming})")
        try:
            dataset = self._load_dataset(dataset_name, split, streaming)
            print("dataset loaded successfully")
        except Exception as e:
            print(f"error loading dataset: {e}", file=sys.stderr)
            return {"error": str(e)}

        total_loss = 0.0
        total_tokens = 0
        samples_processed = 0
        skipped_samples = 0

        print(f"starting evaluation (max_samples={max_samples or 'all'})")
        print("-" * 50)

        with torch.no_grad():
            for i, text in enumerate(self._iterate_texts(dataset)):
                if max_samples and samples_processed >= max_samples:
                    print(f"\nreached max samples limit ({max_samples})")
                    break
                
                if len(text.strip()) < 10:
                    skipped_samples += 1
                    continue
                
                samples_processed += 1
                
                # tokenize the entire document
                document_tokens = self.tokenizer.encode(text, allowed_special="all")
                
                # process the document in chunks
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

        print("-" * 50)
        print(f"evaluation complete: {samples_processed} documents, {skipped_samples} skipped")

        return self._calculate_final_metrics(total_loss, total_tokens, samples_processed, skipped_samples)

    def _chunk_document(self, tokens: list[int]) -> Iterator[list[int]]:
        """yields chunks of a tokenized document with overlap."""
        stride = self.cfg.max_seq_len // 2
        for i in range(0, len(tokens), stride):
            chunk = tokens[i : i + self.cfg.max_seq_len]
            if len(chunk) > 1:
                yield chunk

    def _compute_loss(self, input_ids: torch.Tensor) -> tuple[float, int]:
        """
        computes the average cross-entropy loss for a given sequence.
        """
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        num_predictions = targets.shape[1]

        if num_predictions == 0:
            return 0.0, 0

        _, loss = self.model(inputs, targets)
        return loss.item(), num_predictions

    def _load_dataset(self, dataset_name: str, split: str, streaming: bool):
        """load dataset with appropriate configuration"""
        if ":" in dataset_name:
            path, config_name = dataset_name.split(":", 1)
            return load_dataset(path, config_name, split=split, streaming=streaming, trust_remote_code=True)
        return load_dataset(dataset_name, split=split, streaming=streaming, trust_remote_code=True)

    def _iterate_texts(self, dataset) -> Iterator[str]:
        """extract and preprocess text from dataset examples."""
        for example in dataset:
            text = example.get('highlights', example.get('sentence', example.get('text', example.get('content', ''))))
            if text:
                processed_text = self._preprocess_text(text)
                if '<unk>' not in processed_text:
                    yield processed_text

    def _print_debug_info(self, sample_num, text, tokens, n_predictions, loss):
        """prints information for a single sample in debug mode"""
        print(f"\n[debug] sample {sample_num}:")
        print(f"  text preview: {text[:100]}...")
        print(f"  total tokens: {tokens.shape[1]}, predictions: {n_predictions}, loss: {loss:.4f}")

    def _print_progress(self, i, samples, total_tokens, total_loss, skipped):
        """prints the current evaluation progress"""
        current_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        avg_tokens = total_tokens / samples
        print(f"[{i+1:5d}] documents: {samples:4d}, ppl: {current_ppl:7.2f}, "
              f"avg_tokens/doc: {avg_tokens:.1f}, skipped: {skipped}")

    def _calculate_final_metrics(self, total_loss, total_tokens, samples, skipped):
        """calculates and returns the final evaluation metrics"""
        if total_tokens == 0:
            return {"perplexity": float('inf'), "loss": float('inf'), "samples": 0, "tokens": 0, "skipped": skipped}

        perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        avg_loss = total_loss / total_tokens
        return {
            "perplexity": perplexity,
            "loss": avg_loss,
            "samples": samples,
            "tokens": total_tokens,
            "skipped": skipped,
            "avg_tokens_per_sample": total_tokens / max(samples, 1)
        }


def run_evaluation(args, tasks_to_run):
    print("=== model evaluation ===")
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
    if args.debug:
        print("debug mode: enabled")

    # if no tasks are provided, fall back to the dataset specified in the config
    if not tasks_to_run:
        if not hasattr(cfg, 'dataset_path'):
            raise ValueError("no dataset specified in config and no tasks provided via command line.")
        
        dataset_name = cfg.dataset_path
        if hasattr(cfg, 'dataset_name') and cfg.dataset_name and cfg.dataset_name != 'default':
            dataset_name = f"{dataset_name}:{cfg.dataset_name}"
        
        print(f"using dataset from config: {dataset_name}")
        tasks_to_run = {
            'config_dataset': {
                'dataset_name': dataset_name,
                'split': getattr(cfg, 'dataset_split', 'train'),
                'max_samples': args.max_samples,
                'streaming': True
            }
        }

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
    print("\n" + "="*60)
    print("final evaluation results")
    print("="*60)
    for task, metrics in results.items():
        print(f"\n[{task.upper()}]")
        print("-" * (len(task) + 2))
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    print("\n" + "="*60)


def main():
    """parses arguments and runs the evaluation"""
    parser = argparse.ArgumentParser(description='evaluate model perplexity.')
    parser.add_argument('--model', default='model.pth', help='path to model checkpoint')
    parser.add_argument('--config', default='config/gpt2_bert_corpus_gpu.yaml', help='path to config file')
    parser.add_argument('--dataset', help='evaluate a custom dataset (e.g., "openwebtext" or "c4:en")')
    parser.add_argument('--tasks', nargs='*', help='predefined tasks to run (e.g., wikitext-103, ptb, c4, simple-wikipedia)')
    parser.add_argument('--max-samples', type=int, default=100, help='maximum samples for custom dataset or config default')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    args = parser.parse_args()

    tasks_to_run = {}
    if args.dataset:
        tasks_to_run['custom'] = {'dataset_name': args.dataset, 'max_samples': args.max_samples, 'streaming': True}
    elif args.tasks:
        all_tasks = {
            'training-corpus': {'dataset_name': 'shahrukhx01/wikipedia-bookscorpus-en-preprocessed', 'split': 'train', 'max_samples': 1000, 'streaming': True},
            'wikitext-103': {'dataset_name': 'wikitext:wikitext-103-v1', 'split': 'test', 'max_samples': 1000},
            'simple-wikipedia': {'dataset_name': 'wikimedia/wikipedia:20231101.simple', 'split': 'train', 'max_samples': 1000, 'streaming': True},
            #'cnn_dailymail': {'dataset_name': 'cnn_dailymail:3.0.0', 'split': 'test', 'max_samples': 1000},
            'cnn_dailymail': {'dataset_name': 'abisee/cnn_dailymail:3.0.0', 'split': 'test', 'max_samples': 1000},
            'c4': {'dataset_name': 'c4:en', 'split': 'train', 'max_samples': 50, 'streaming': True}
        }
        tasks_to_run = {name: all_tasks[name] for name in args.tasks if name in all_tasks}

    try:
        run_evaluation(args, tasks_to_run)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nerror: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
