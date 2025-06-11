import torch
import wandb
from .config import GPT2Config

class WandbLogger:
    """
    A wrapper class for Weights & Biases logging that uses 
    a context manager to handle init / clean up.

    Usage:

        with WandbLogger(cfg, model) as logger:
            training code ...
    """
    def __init__(self, cfg: GPT2Config, model: torch.nn.Module):
        self.use_wandb = cfg.wandb_log
        if self.use_wandb:
            self.run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config={k: getattr(cfg, k) for k, _ in cfg.__annotations__.items()}
            )
            # watch the model to log gradients and parameters
            wandb.watch(model, log='all', log_freq=cfg.log_interval)

    def __enter__(self):
        """Entering 'with' calls this method"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        exiting the 'with' calls this method which calls wandb.finish
        """
        if self.use_wandb:
            wandb.finish()

    def log(self, step, **kwargs):
        """Logs kwargs to wandb (if logging is enabled)."""
        if self.use_wandb:
            wandb.log(step=step, commit=True, **kwargs)
