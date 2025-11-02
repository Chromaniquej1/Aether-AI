"""
ModelForge-CV: Training Engine
Orchestrates model training with monitoring, checkpointing, and early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    
    # Learning rate schedule
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau', 'none'
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    
    # Training settings
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 0
    
    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = 'val_loss'  # 'val_loss' or 'val_acc'
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_every_n_epochs: int = 5
    
    # Regularization
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.0
    
    # Gradual unfreezing (if enabled)
    unfreeze_every_n_epochs: int = 5
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingMetrics:
    """Training metrics tracker"""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_f1: float
    learning_rate: float
    epoch_time: float
    gpu_memory_mb: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)


class MetricsLogger:
    """Logs and tracks training metrics"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def log_epoch(self, metrics: TrainingMetrics):
        """Log metrics for an epoch"""
        self.metrics_history.append(metrics)
        
        # Update best metrics
        if metrics.val_loss < self.best_val_loss:
            self.best_val_loss = metrics.val_loss
        
        if metrics.val_acc > self.best_val_acc:
            self.best_val_acc = metrics.val_acc
            self.best_epoch = metrics.epoch
        
        # Save to JSON
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump([m.to_dict() for m in self.metrics_history], f, indent=2)
    
    def get_summary(self) -> Dict:
        """Get training summary"""
        return {
            'total_epochs': len(self.metrics_history),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.metrics_history[-1].train_loss if self.metrics_history else None,
            'final_val_acc': self.metrics_history[-1].val_acc if self.metrics_history else None,
        }


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, metric: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def step(self, metrics: TrainingMetrics) -> bool:
        """Check if should stop training"""
        score = getattr(metrics, self.metric)
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check if improved
        improved = False
        if self.mode == 'min':
            improved = score < self.best_score
        else:
            improved = score > self.best_score
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class TrainingEngine:
    """Main training orchestrator"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        save_dir: str,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        self.scaler = GradScaler() if config.use_amp else None
        
        # Monitoring
        self.metrics_logger = MetricsLogger(self.save_dir)
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            metric=config.early_stopping_metric,
            mode='min' if 'loss' in config.early_stopping_metric else 'max'
        )
        
        # State
        self.current_epoch = 0
        self.global_step = 0
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        if self.config.optimizer.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.config.optimizer.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.scheduler == 'none':
            return None
        
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif self.config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.config.min_lr
            )
        
        return None
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function"""
        return nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                if self.config.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Collect metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            self.global_step += 1
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }
    
    def save_checkpoint(self, metrics: TrainingMetrics, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics.to_dict(),
            'config': self.config.to_dict()
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'checkpoint_latest.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'checkpoint_best.pt')
            print(f"✨ New best model saved! Val Acc: {metrics.val_acc:.4f}")
        
        # Save periodic checkpoints
        if (self.current_epoch + 1) % self.config.checkpoint_every_n_epochs == 0:
            torch.save(
                checkpoint,
                self.save_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pt'
            )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        progress_callback: Optional[Callable] = None
    ):
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            progress_callback: Optional callback for progress updates
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Optimizer: {self.config.optimizer}")
        print(f"Mixed Precision: {self.config.use_amp}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Compute epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Get GPU memory usage
            gpu_memory_mb = None
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            
            # Create metrics object
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_metrics['loss'],
                train_acc=train_metrics['accuracy'],
                val_loss=val_metrics['loss'],
                val_acc=val_metrics['accuracy'],
                val_f1=val_metrics['f1'],
                learning_rate=self.optimizer.param_groups[0]['lr'],
                epoch_time=epoch_time,
                gpu_memory_mb=gpu_memory_mb
            )
            
            # Log metrics
            self.metrics_logger.log_epoch(metrics)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch + 1}/{self.config.epochs} Summary:")
            print(f"   Train Loss: {metrics.train_loss:.4f} | Train Acc: {metrics.train_acc:.4f}")
            print(f"   Val Loss: {metrics.val_loss:.4f} | Val Acc: {metrics.val_acc:.4f} | Val F1: {metrics.val_f1:.4f}")
            print(f"   Time: {epoch_time:.2f}s | LR: {metrics.learning_rate:.6f}")
            if gpu_memory_mb:
                print(f"   GPU Memory: {gpu_memory_mb:.1f} MB")
            
            # Check if best model
            is_best = metrics.val_acc >= self.metrics_logger.best_val_acc
            
            # Save checkpoint
            if self.config.save_best_only:
                if is_best:
                    self.save_checkpoint(metrics, is_best=True)
            else:
                self.save_checkpoint(metrics, is_best=is_best)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics.val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping check
            if self.early_stopping.step(metrics):
                print(f"\n⚠️ Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Progress callback
            if progress_callback:
                progress_callback(metrics)
            
            print()
        
        # Training complete
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"{'='*60}")
        summary = self.metrics_logger.get_summary()
        print(f"Best Val Accuracy: {summary['best_val_acc']:.4f} (Epoch {summary['best_epoch']})")
        print(f"Best Val Loss: {summary['best_val_loss']:.4f}")
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"{'='*60}\n")
        
        return summary


# Example usage
if __name__ == "__main__":
    from torchvision import datasets, transforms
    
    # Mock dataset for demo
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dummy model
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(224*224*3, 10)
    )
    
    # Training configuration
    config = TrainingConfig(
        learning_rate=1e-3,
        epochs=5,
        batch_size=32,
        early_stopping_patience=3
    )
    
    # Create training engine
    engine = TrainingEngine(
        model=model,
        config=config,
        save_dir="./experiments/run_001"
    )
    
    print("✅ Training Engine initialized!")
    print(f"   Save directory: {engine.save_dir}")
    print(f"   Device: {engine.device}")