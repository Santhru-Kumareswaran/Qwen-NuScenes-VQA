
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional

def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    val_epochs: List[int],
    out_dir: Path,
    filename: str = "loss_curve.png"
):
    """
    Plot train and validation losses.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    train_epochs = list(range(1, len(train_losses) + 1))
    plt.plot(train_epochs, train_losses, label="Train Loss", linewidth=2, marker="o", markersize=3, alpha=0.8)
    
    if val_losses and val_epochs:
        plt.plot(val_epochs, val_losses, label="Validation Loss", linewidth=2, marker="s", markersize=5, linestyle='--')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120)
    plt.close()

def plot_step_loss(
    step_losses: List[float], 
    out_dir: Path, 
    window: int = 50,
    val_steps: Optional[List[int]] = None,
    val_losses: Optional[List[float]] = None
):
    """
    Plot step-wise loss with moving average and optional overlaid validation points.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    # Train Data
    plt.plot(step_losses, alpha=0.2, color='blue', label="Raw Step Loss", linewidth=0.5)
    
    # Moving average
    if len(step_losses) >= window:
        ma = [sum(step_losses[i:i+window])/window for i in range(len(step_losses)-window+1)]
        plt.plot(range(window-1, len(step_losses)), ma, color='blue', linewidth=1.5, label=f"Train MA ({window})")
        
    # Validation Data
    if val_steps and val_losses:
        plt.plot(val_steps, val_losses, 'o-', color='orange', label="Validation Loss", linewidth=2, markersize=5, markeredgecolor='black')
        
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Step-wise Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "step_loss.png", dpi=120)
    plt.close()
