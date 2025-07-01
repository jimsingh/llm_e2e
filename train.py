"""
Training script using the integrated GPT2Trainer.
This would replace the training logic in notebooks/03_gpt2_training.ipynb
"""
import argparse
import os
import sys
import math
import torch
import tiktoken

import torch.backends.cuda
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from pathlib import Path

from llm_e2e import GPT2Config, GPT2Model, StreamingDatasetGenerator, WandbLogger, GPT2Trainer
from llm_e2e.trainer import generate_text

def setup_flash_attention():
    if torch.cuda.is_available():
        # Enable Flash Attention v2
        torch.backends.cuda.enable_flash_sdp(True)

        # Enable memory-efficient attention (fallback)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Disable math attention (slower fallback)
        torch.backends.cuda.enable_math_sdp(False)

        print("Flash Attention enabled")
    else:
        print("CUDA not available, Flash Attention disabled")

def setup_cuda(cfg: GPT2Config):
    """setup CUDA following project patterns"""
    if not torch.cuda.is_available():
        cfg.device = 'cpu'
        return

    assert cfg.device == 'cuda', "cfg.device must be 'cuda' if CUDA is available."
    print(f"cuda version: {torch.version.cuda}")
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 7:
        torch.set_float32_matmul_precision("high")
        print("uses tensor cores")
    else:
        print("tensor cores not supported on this gpu.")

    setup_flash_attention()

def create_scheduler(cfg: GPT2Config, optimizer):
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_steps)

    annealing_steps = (cfg.max_steps - cfg.warmup_steps)
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=annealing_steps,
        eta_min=min(cfg.learning_rate * 0.1, 5e-6)
    )

    print(f"scheduler: warmup_steps={cfg.warmup_steps}, anealing_steps={annealing_steps}, total_steps={cfg.max_steps}")
    scheduler = SequentialLR(optimizer, [warmup, main_scheduler], milestones=[cfg.warmup_steps])
    return scheduler

def main(config_yaml: str):
    # configuration
    cfg = GPT2Config.from_yaml(config_yaml)
    encoding = tiktoken.get_encoding(cfg.encoding_name)
    setup_cuda(cfg)

    # adjust save path for colab if needed
    if 'google.colab' in str(os.environ.get('COLAB_GPU', '')):
        cfg.save_filename = "/content/drive/MyDrive/llm_e2e/" + cfg.save_filename
        print(f"save_filename: {cfg.save_filename}")

    # create datasets
    train_dataset = StreamingDatasetGenerator(cfg, encoding=encoding)
    if cfg.val_split:
        val_dataset = StreamingDatasetGenerator(cfg, encoding=encoding, dataset_split=cfg.val_split, seed=1337)
    else:
        val_dataset = StreamingDatasetGenerator(cfg, encoding=encoding, split='val', val_frac=0.05, seed=1337)

    # create model
    model = GPT2Model(cfg)

    # prepare model for training
    if cfg.device == 'cuda':
        model.to(torch.bfloat16)
    model.to(cfg.device)

    if cfg.compile_model:
        model = torch.compile(model)

    # we don't decay params that shift (bias) or normalize (layernorm). These
    # parameters don't learn data, they stabalize the model. Decaying them may
    # do more harm than good.
    param_decays = [
        {
            "params": [p for n, p in model.named_parameters() if p.dim() >= 2],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.dim() < 2], # excludes bias, layer norm
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_decays, 
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2)
    )

    # tone down the LR at first and then switch to our main scheduler 
    scheduler = create_scheduler(cfg, optimizer)

    # create text generator
    gen_f = lambda m: generate_text(m, encoding, cfg.eval_prompt)

    # train with wandb logging
    with WandbLogger(cfg, model) as logger:
        trainer = GPT2Trainer(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_dataset,
            val_loader=val_dataset,
            logger=logger
        )

        # check for existing checkpoint - try multiple patterns
        checkpoint_paths = [
            f"{cfg.save_filename}_latest",
            f"{cfg.save_filename}_checkpoint",
            f"{cfg.save_filename}_best"
        ]

        checkpoint_loaded = False
        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                try:
                    trainer.load_checkpoint(checkpoint_path)
                    print(f"Resuming from checkpoint: {checkpoint_path}")
                    checkpoint_loaded = True
                    break
                except Exception as e:
                    print(f"Could not load checkpoint {checkpoint_path}: {e}")
                    continue

        if not checkpoint_loaded:
            # try loading just model weights if no checkpoint found
            if os.path.exists(cfg.save_filename):
                try:
                    params = torch.load(cfg.save_filename, weights_only=True, map_location=cfg.device)
                    # handle compiled model state dict
                    orig_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                    orig_model.load_state_dict(params)
                    print(f"Loaded model weights: {cfg.save_filename}")
                except Exception as e:
                    print(f"Could not load model weights: {e}")
                    print("Starting fresh training")
            else:
                print("Starting fresh training")

        if checkpoint_loaded:
             for param_group in trainer.optimizer.param_groups:
                 lr = param_group['lr']
                 if not math.isclose(lr, cfg.learning_rate):
                     param_group['lr'] = cfg.learning_rate
                     print("learning rate changed, disabling scheduler")
                     trainer.scheduler = None

        # randomize dataloader using the current step. This ensures
        # we see different data after restart, but isn't ideal for
        # situations were need to process every batch of data
        if trainer.state.step > 0 and trainer.state.epoch == 0:
           trainer.train_loader.start_step = trainer.state.step

        # train
        trainer.train(text_generator=gen_f)

        # save final model
        final_model = trainer._orig_model
        torch.save(final_model.state_dict(), f"{cfg.save_filename}_final.pth")
        print(f"Saved final model to {cfg.save_filename}_final.pth")

    # cleanup for colab
    if 'google.colab' in str(os.environ.get('COLAB_GPU', '')):
        # cleanup GPU memory
        del model, optimizer, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # disconnect runtime
        try:
            from google.colab import runtime
            runtime.unassign()
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gpt2 model training script.")
    parser.add_argument("--config", type=str, required=True, help="path to YAML config file")
    args = parser.parse_args()

    # check if the config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}", file=sys.stderr)
        sys.exit(1)

    main(args.config)
