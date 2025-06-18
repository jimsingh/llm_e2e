"""
refactored trainer from 03_gpt2_trainer.ipynb
"""
import torch
import itertools
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from llm_e2e import GPT2Config, WandbLogger


@dataclass
class TrainerState:
    """encapsulates trainer state into a single member variable""" 
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float('inf')
    running_loss: float = 0.0
    gradient_norms: list = field(default_factory=list) 
    model_state_dict: dict = field(default_factory=dict)
    optimizer_state_dict: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    
    @property
    def avg_gradient_norm(self) -> float:
        if len(self.gradient_norms) == 0: return 0
        return sum(self.gradient_norms) / len(self.gradient_norms)

    def to_dict(self):
        """convert state to dictionary for checkpointing"""
        return {
            'step': self.step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'running_loss': self.running_loss,
            'model_state_dict': self.model_state_dict,
            'optimizer_state_dict': self.optimizer_state_dict,
            'config': self.config
        }
    
    @classmethod
    def from_dict(cls, d):
        """restore state from checkpoint dictionary"""
        return cls(**d)


class GPT2Trainer:
    def __init__(
        self, 
        cfg: GPT2Config,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader,
        val_loader,
        logger
    ):
        self.cfg = cfg 
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.device = torch.device(cfg.device)
        
        # initialize state
        self.state = TrainerState(config=vars(cfg))
        
        # setup cuda if available
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            
        # remove torch.compile wrapper if needed (most of the time)
        self._orig_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    def log(self, message: str, metrics: dict = None, commit: bool = True):
        """unified logging method"""
        print(f"{self.state.step}: {message}")
        
        # GPU memory logging
        if metrics and metrics.get('log_gpu_memory') and self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
            max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            print(f"  GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Peak: {max_allocated:.2f} MB")
            
            # log to wandb
            if self.logger:
                metrics.update({
                    'gpu_memory_allocated_mb': allocated,
                    'gpu_memory_reserved_mb': reserved,
                    'gpu_memory_peak_mb': max_allocated
                })
        
        # log metrics to wandb if available
        if self.logger and metrics:
            # remove internal flags 
            wandb_metrics = {k: v for k, v in metrics.items() if not k.startswith('log_') and k not in ('step')}
            if wandb_metrics:
                self.logger.log(step=self.state.step, commit=commit, **wandb_metrics)
    
    def train(self, text_generator = None):
        """main training loop"""
        # log training start
        self.log(
            f"Starting training on {self.device}\n"
            f"Model parameters: {self.cfg.n_params:_}\n"
            f"Model parameters file: {self.cfg.save_filename}"
        )
        
        try:
            for epoch in range(self.state.epoch, self.cfg.num_epochs):
                self.state.epoch = epoch
                self._train_epoch(epoch, text_generator)
                
                # save end-of-epoch checkpoint
                self.save_checkpoint(f"{self.cfg.save_filename}_epoch{epoch}_final")
                
        except KeyboardInterrupt:
            self.log("\nTraining interrupted by user")
            
        except Exception as e:
            self.log(f"\nTraining failed with error: {str(e)}")
            raise
            
        finally:
            # always save current state
            checkpoint_name = f"{self.cfg.save_filename}_latest"
            self.save_checkpoint(checkpoint_name)
            
            self.log(
                f"\nTraining ended\n"
                f"Best validation loss: {self.state.best_val_loss:.4f}\n"
                f"Total steps: {self.state.step}\n"
                f"Final checkpoint saved: {checkpoint_name}"
            )
    
    def _train_epoch(self, epoch: int, text_generator):
        """train for one epoch"""
        self.model.train()
        self.state.running_loss = 0.0
        
        # log epoch start
        self.log(
            f"\n[Epoch {epoch + 1}/{self.cfg.num_epochs}] "
            f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        for i, (X, Y) in enumerate(self.train_loader):
            self.state.step += 1
            
            # train single batch
            loss = self._train_batch(X, Y)
            self.state.running_loss += loss.item()
            
            # periodic evaluation
            if (i + 1) % self.cfg.eval_interval == 0:
                self._evaluate_and_checkpoint(epoch, i, text_generator)
                self.state.gradient_norms.clear()

            if (i + 1) % self.cfg.log_interval == 0:
                avg_loss = self.state.running_loss / self.cfg.log_interval
                current_lr = self.optimizer.param_groups[0]['lr']
                self.log(
                    f"[{epoch + 1}/{i + 1:5d}] Running loss: {avg_loss:.3f}",
                    {'running_loss': avg_loss, 'step': self.state.step, 'lr': current_lr}
                )
                self.state.running_loss = 0.0
            
    
    def _train_batch(self, X, Y):
        """process single training batch"""
        X, Y = X.to(self.device), Y.to(self.device)
        
        self.optimizer.zero_grad()
        logits, loss = self.model(X, Y)
        loss.backward()

        # clip gradients if they get beyond our limit
        grad_norm = self._gradient_norm()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        self.state.gradient_norms.append(self._gradient_norm())

        self.optimizer.step()

        # optimizer and scheduler .step are called with the same frequency
        if self.scheduler:
            self.scheduler.step()

        return loss
    
    def _evaluate_and_checkpoint(self, epoch: int, batch_idx: int, text_generator):
        """evaluate model and save checkpoint"""
        losses = self.evaluate()
        
        # prepare metrics
        metrics = {
            'train_loss': losses['train'],
            'val_loss': losses['val'],
            'step': self.state.step,
            'gradient_sum': self.state.avg_gradient_norm,
            'epoch': epoch + 1
        }
        
        # generate and log sample text
        if text_generator:
            completion = text_generator(self.model)
            print(f"[{epoch + 1}/{batch_idx + 1:5d}] {completion}")
            metrics['generated_text'] = completion
        
        # log evaluation
        self.log(
            f"[{epoch + 1}/{batch_idx + 1:5d}] "
            f"Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}",
            metrics,
            commit = False
        )
        
        # save checkpoint
        checkpoint_name = f"{self.cfg.save_filename}_{epoch}"
        self.save_checkpoint(checkpoint_name)
        
        # update best model
        if losses['val'] < self.state.best_val_loss:
            self.state.best_val_loss = losses['val']
            self.save_checkpoint(f"{self.cfg.save_filename}_best")
    
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """evaluate model on train and validation sets"""
        train_loss = self._estimate_loss(self.train_loader)
        val_loss = self._estimate_loss(self.val_loader)
        return {'train': train_loss, 'val': val_loss}
    
    def _estimate_loss(self, loader) -> float:
        """estimate average loss over eval_iters batches"""
        self.model.eval()
        losses = torch.zeros(self.cfg.eval_iters)
        
        for i, (X, Y) in enumerate(itertools.islice(loader, self.cfg.eval_iters)):
            X, Y = X.to(self.device), Y.to(self.device)
            _, loss = self.model(X, Y)
            losses[i] = loss.item()
        
        self.model.train()
        return losses.mean().item()
   
    def _gradient_norm(self):
        """return calculate gradient norm"""
        p = self.model.parameters()

        total_norm = sum(
            p.grad.data.norm(2).item() ** 2  # L2 norm squared for each parameter
            for p in self.model.parameters()      # iterate through all parameters
            if p.grad is not None           # only parameters with gradients
        ) ** 0.5                            # square root of the sum = total L2 norm

        return total_norm
         
    def save_checkpoint(self, filepath: str):
        """save training checkpoint with robust error handling"""
        # ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # update state dicts - handle compiled models
        self.state.model_state_dict = self._orig_model.state_dict()
        self.state.optimizer_state_dict = self.optimizer.state_dict()
        
        # save with atomic write
        tmp_path = f"{filepath}.tmp"
        torch.save(self.state.to_dict(), tmp_path)
        os.replace(tmp_path, filepath)
        
        # save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'step': self.state.step,
            'epoch': self.state.epoch,
            'best_val_loss': self.state.best_val_loss,
            'config': self.state.config
        }
        with open(f"{filepath}.meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_checkpoint(self, filepath: str):
        """load training checkpoint"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.state = TrainerState.from_dict(checkpoint)
        
        # restore model and optimizer states
        self._orig_model.load_state_dict(self.state.model_state_dict)
        self.optimizer.load_state_dict(self.state.optimizer_state_dict)
        
        self.log(
            f"Loaded checkpoint from {filepath}\n"
            f"Resuming from epoch {self.state.epoch}, step {self.state.step}"
        )


# utility function
def generate_text(model: torch.nn.Module, tokenizer, prompt: str, max_tokens: int = 20) -> str:
    """generate text from a prompt"""
    model.eval()
    device = next(model.parameters()).device
    
    encoded = tokenizer.encode(prompt)
    encoded_ids = torch.tensor([encoded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output_token_ids = model.generate(encoded_ids, max_tokens)
    
    decoded_ids_list = output_token_ids[0].cpu().tolist()
    decoded_text = tokenizer.decode(decoded_ids_list)
    return decoded_text
