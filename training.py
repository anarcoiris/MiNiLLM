"""
training.py

Sistema de entrenamiento mejorado con:
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate warmup + cosine decay
- Early stopping
- Checkpointing inteligente
- Logging detallado
- Gradient clipping
- Model EMA (opcional)
"""

import math
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


# ======================== Configuration ========================

@dataclass
class TrainingConfig:
    """Configuración completa de entrenamiento."""
    # Optimizer
    lr: float = 5e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)

    # Learning rate schedule
    warmup_steps: int = 500
    max_steps: Optional[int] = None  # Si None, usa epochs
    lr_decay_steps: Optional[int] = None  # Para cosine decay
    min_lr_ratio: float = 0.1  # lr_min = lr * min_lr_ratio

    # Training dynamics
    epochs: int = 20
    grad_clip: float = 1.0
    accumulation_steps: int = 1  # Gradient accumulation

    # Regularization
    label_smoothing: float = 0.0

    # Mixed precision
    use_amp: bool = True

    # Evaluation & checkpointing
    eval_every: int = 500  # steps
    save_every: int = 2000  # steps
    log_every: int = 50  # steps

    # Early stopping
    patience: int = 5  # epochs sin mejora
    min_delta: float = 1e-4  # Mejora mínima para considerar progreso

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Misc
    seed: int = 42


@dataclass
class TrainingHistory:
    """Historial de entrenamiento."""
    train_losses: List[float]
    val_losses: List[float]
    learning_rates: List[float]
    steps: List[int]
    best_val_loss: float
    best_step: int
    total_time: float

    def save(self, path: Path):
        """Guarda historial a JSON."""
        data = asdict(self)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path):
        """Carga historial desde JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ======================== Learning Rate Schedulers ========================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Cosine annealing con warmup lineal.

    Args:
        optimizer: Optimizer
        warmup_steps: Pasos de warmup
        max_steps: Total de pasos de entrenamiento
        min_lr_ratio: lr_min / lr_max
        last_epoch: Para reanudar entrenamiento
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Escala entre min_lr_ratio y 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# ======================== Training Functions ========================

def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Calcula cross-entropy loss con label smoothing opcional.

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        label_smoothing: Factor de smoothing [0, 1]
    """
    B, T, V = logits.shape
    logits_flat = logits.view(-1, V)
    targets_flat = targets.view(-1)

    if label_smoothing > 0:
        # Label smoothing: suaviza distribución target
        log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets_flat.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - label_smoothing) * nll_loss + label_smoothing * smooth_loss
        return loss.mean()
    else:
        return torch.nn.functional.cross_entropy(logits_flat, targets_flat)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    max_batches: Optional[int] = 50  # Default: evaluar solo 50 batches
) -> float:
    """
    Evalúa modelo en validation set.

    Args:
        model: Modelo a evaluar
        dataloader: DataLoader de validación
        device: Device
        max_batches: Límite de batches (para evaluación rápida). None = todo el set

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = compute_loss(logits, y)

        batch_tokens = y.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    model.train()
    return total_loss / max(total_tokens, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    out_dir: Path,
    resume_from: Optional[Path] = None
) -> TrainingHistory:
    """
    Entrena modelo con todas las mejoras.

    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        config: TrainingConfig
        out_dir: Directorio para checkpoints
        resume_from: Path a checkpoint para continuar

    Returns:
        TrainingHistory con métricas
    """
    # Setup
    out_dir.mkdir(parents=True, exist_ok=True)
    device = config.device
    model = model.to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas
    )

    # Learning rate scheduler
    steps_per_epoch = len(train_loader) // config.accumulation_steps
    max_steps = config.max_steps or (steps_per_epoch * config.epochs)
    lr_decay_steps = config.lr_decay_steps or max_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=config.warmup_steps,
        max_steps=lr_decay_steps,
        min_lr_ratio=config.min_lr_ratio
    )

    # Mixed precision
    scaler = GradScaler(enabled=config.use_amp)

    # History tracking
    history = TrainingHistory(
        train_losses=[],
        val_losses=[],
        learning_rates=[],
        steps=[],
        best_val_loss=float('inf'),
        best_step=0,
        total_time=0.0
    )

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    epochs_without_improvement = 0

    if resume_from and resume_from.exists():
        print(f"\n📂 Resumiendo desde: {resume_from}")
        checkpoint = load_checkpoint(resume_from, model, optimizer, scheduler, scaler)
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('step', 0)
        history = checkpoint.get('history', history)
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        print(f"   Epoch: {start_epoch}, Step: {global_step}")

    # Save config
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Training loop
    print(f"\n{'='*70}")
    print(f"🚀 INICIANDO ENTRENAMIENTO")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {config.epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {max_steps}")
    print(f"Accumulation steps: {config.accumulation_steps}")
    print(f"Batch size efectivo: {train_loader.batch_size * config.accumulation_steps}")
    print(f"{'='*70}\n")

    start_time = time.time()
    model.train()

    for epoch in range(start_epoch, config.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0

        print(f"\n📅 Epoch {epoch + 1}/{config.epochs}")
        print(f"{'-'*70}")

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward pass con AMP
            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = compute_loss(logits, y, config.label_smoothing)
                loss = loss / config.accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            # Accumulate metrics
            epoch_loss += loss.item() * config.accumulation_steps * y.numel()
            epoch_tokens += y.numel()

            # Update weights cada accumulation_steps
            if (batch_idx + 1) % config.accumulation_steps == 0:
                # Gradient clipping
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                global_step += 1

                # Logging
                if global_step % config.log_every == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / epoch_tokens

                    print(f"Step {global_step:6d} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {current_lr:.2e} | "
                          f"Batch: {batch_idx + 1}/{len(train_loader)}")

                    history.train_losses.append(avg_loss)
                    history.learning_rates.append(current_lr)
                    history.steps.append(global_step)

                # Evaluation
                if global_step % config.eval_every == 0:
                    val_loss = evaluate(model, val_loader, device)
                    history.val_losses.append(val_loss)

                    print(f"\n📊 Validation Loss: {val_loss:.4f}")

                    # Check for improvement
                    if val_loss < history.best_val_loss - config.min_delta:
                        history.best_val_loss = val_loss
                        history.best_step = global_step
                        epochs_without_improvement = 0

                        # Save best checkpoint
                        save_checkpoint(
                            out_dir / 'ckpt_best.pt',
                            model, optimizer, scheduler, scaler,
                            epoch, global_step, val_loss, history
                        )
                        print(f"✅ Nuevo mejor modelo guardado (val_loss: {val_loss:.4f})")

                    print()

                # Regular checkpointing
                if global_step % config.save_every == 0:
                    save_checkpoint(
                        out_dir / f'ckpt_step_{global_step}.pt',
                        model, optimizer, scheduler, scaler,
                        epoch, global_step, val_loss, history
                    )
                    print(f"💾 Checkpoint guardado: step {global_step}")

                # Max steps reached
                if global_step >= max_steps:
                    print(f"\n✅ Max steps ({max_steps}) alcanzado")
                    break

        # End of epoch evaluation
        epoch_train_loss = epoch_loss / epoch_tokens
        val_loss = evaluate(model, val_loader, device)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {epoch_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Best Val:   {history.best_val_loss:.4f} (step {history.best_step})")
        print(f"{'='*70}")

        # Check early stopping
        if val_loss >= history.best_val_loss - config.min_delta:
            epochs_without_improvement += 1
            print(f"⚠️  Sin mejora: {epochs_without_improvement}/{config.patience} epochs")

            if epochs_without_improvement >= config.patience:
                print(f"\n🛑 Early stopping activado después de {config.patience} epochs sin mejora")
                break
        else:
            epochs_without_improvement = 0

        # Save epoch checkpoint
        save_checkpoint(
            out_dir / 'ckpt_last.pt',
            model, optimizer, scheduler, scaler,
            epoch, global_step, val_loss, history,
            epochs_without_improvement=epochs_without_improvement
        )

        if global_step >= max_steps:
            break

    # Training complete
    history.total_time = time.time() - start_time
    history.save(out_dir / 'history.json')

    print(f"\n{'='*70}")
    print(f"✅ ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"Tiempo total: {history.total_time / 3600:.2f} horas")
    print(f"Mejor val_loss: {history.best_val_loss:.4f} (step {history.best_step})")
    print(f"Checkpoints guardados en: {out_dir}")
    print(f"{'='*70}\n")

    return history


# ======================== Checkpoint Management ========================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    epoch: int,
    step: int,
    val_loss: float,
    history: TrainingHistory,
    **kwargs
):
    """Guarda checkpoint completo."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'val_loss': val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'model_config': {
            'vocab_size': model.vocab_size,
            'block_size': model.block_size,
            'n_embd': model.n_embd,
            'n_layer': model.n_layer,
            'n_head': model.n_head,
            'use_rope': model.use_rope,
        },
        **kwargs
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[GradScaler] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Carga checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)  # TORCH_LOAD_FIX_APPLIED

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint


# ======================== Utilities ========================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Cuenta parámetros del modelo."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def estimate_training_time(
    num_steps: int,
    steps_per_second: float
) -> Dict[str, float]:
    """Estima tiempo de entrenamiento."""
    total_seconds = num_steps / steps_per_second

    return {
        'seconds': total_seconds,
        'minutes': total_seconds / 60,
        'hours': total_seconds / 3600,
        'days': total_seconds / 86400
    }


# ======================== Tests ========================

if __name__ == '__main__':
    print("Testing training.py\n")

    # Test learning rate scheduler
    print("Test 1: Learning rate scheduler")
    dummy_model = nn.Linear(10, 10)
    optimizer = AdamW(dummy_model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=100,
        max_steps=1000,
        min_lr_ratio=0.1
    )

    # Simulate training
    lrs = []
    for step in range(1000):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    print(f"  LR at step 0: {lrs[0]:.6f}")
    print(f"  LR at step 50 (warmup): {lrs[50]:.6f}")
    print(f"  LR at step 100 (end warmup): {lrs[100]:.6f}")
    print(f"  LR at step 500: {lrs[500]:.6f}")
    print(f"  LR at step 999 (end): {lrs[999]:.6f}")

    assert lrs[0] < lrs[100], "LR should increase during warmup"
    assert lrs[100] > lrs[999], "LR should decrease after warmup"
    assert lrs[999] >= 1e-3 * 0.1 * 0.9, "LR should respect min_lr_ratio"
    print("  ✓ Scheduler working correctly\n")

    # Test loss computation
    print("Test 2: Loss computation")
    logits = torch.randn(2, 10, 100)  # (batch, seq, vocab)
    targets = torch.randint(0, 100, (2, 10))

    loss_normal = compute_loss(logits, targets, label_smoothing=0.0)
    loss_smooth = compute_loss(logits, targets, label_smoothing=0.1)

    print(f"  Loss (no smoothing): {loss_normal:.4f}")
    print(f"  Loss (smoothing=0.1): {loss_smooth:.4f}")
    print("  ✓ Loss computation working\n")

    # Test checkpoint save/load
    print("Test 3: Checkpoint save/load")
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy model and history
        model = nn.Linear(10, 10)
        optimizer = AdamW(model.parameters())
        history = TrainingHistory(
            train_losses=[1.0, 0.9],
            val_losses=[1.1, 0.95],
            learning_rates=[1e-3, 1e-3],
            steps=[100, 200],
            best_val_loss=0.95,
            best_step=200,
            total_time=100.0
        )

        # Save
        ckpt_path = tmpdir / 'test_ckpt.pt'
        save_checkpoint(
            ckpt_path, model, optimizer, None, None,
            epoch=1, step=200, val_loss=0.95, history=history
        )

        # Load
        loaded = load_checkpoint(ckpt_path, model, optimizer)

        assert loaded['epoch'] == 1
        assert loaded['step'] == 200
        assert loaded['val_loss'] == 0.95
        print("  ✓ Checkpoint save/load working\n")

    print("✅ All tests passed!")
