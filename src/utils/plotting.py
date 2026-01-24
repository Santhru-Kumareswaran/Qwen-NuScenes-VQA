
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import textwrap

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
        # Always connect validation points with a line
        sorted_pairs = sorted(zip(val_epochs, val_losses))
        val_epochs_sorted = [p[0] for p in sorted_pairs]
        val_losses_sorted = [p[1] for p in sorted_pairs]
        plt.plot(val_epochs_sorted, val_losses_sorted, label="Validation Loss", linewidth=2, marker="s", markersize=5, linestyle='-', color='orange')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
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
        
    # Validation Data - connect with line, not just dots
    if val_steps and val_losses and len(val_steps) > 0:
        sorted_pairs = sorted(zip(val_steps, val_losses))
        val_steps_sorted = [p[0] for p in sorted_pairs]
        val_losses_sorted = [p[1] for p in sorted_pairs]
        
        if len(val_steps_sorted) > 1:
            plt.plot(val_steps_sorted, val_losses_sorted, '-o', color='orange', label="Validation Loss", linewidth=2, markersize=5, markeredgecolor='black')
        else:
            # Single point - show as marker
            plt.scatter(val_steps_sorted, val_losses_sorted, color='orange', label="Validation Loss", s=60, edgecolors='black', zorder=5)
        
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Step-wise Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "step_loss.png", dpi=120)
    plt.close()


def save_inference_comparison(
    image: Image.Image,
    question: str,
    ground_truth: str,
    prediction: str,
    bleu_score: float,
    sample_id: str,
    output_dir: Path,
    epoch: int,
    idx: int
):
    """
    Save an inference comparison image with:
    - Left side: Input image (montage)
    - Right side: Question, Ground Truth, and Prediction text
    """
    output_dir = Path(output_dir) / "inference_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image dimensions
    img_w, img_h = image.size
    
    # Right panel width (for text)
    text_panel_w = max(500, img_w // 2)
    total_w = img_w + text_panel_w
    total_h = max(img_h, 600)
    
    # Create canvas
    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 35))
    
    # Paste input image on left (centered vertically)
    y_offset = (total_h - img_h) // 2
    canvas.paste(image, (0, y_offset))
    
    # Create text panel
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a readable font
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_normal = ImageFont.load_default()
    
    # Text formatting
    margin = 20
    x_text = img_w + margin
    max_text_w = text_panel_w - 2 * margin
    wrap_width = max_text_w // 8  # Approximate character width
    
    y_cursor = 30
    
    # Title
    draw.text((x_text, y_cursor), f"Sample: {sample_id}", font=font_large, fill=(255, 255, 255))
    y_cursor += 35
    
    # BLEU Score
    bleu_color = (100, 255, 100) if bleu_score > 0.3 else (255, 200, 100) if bleu_score > 0.1 else (255, 100, 100)
    draw.text((x_text, y_cursor), f"BLEU-4: {bleu_score:.4f}", font=font_large, fill=bleu_color)
    y_cursor += 45
    
    # Question
    draw.text((x_text, y_cursor), "Question:", font=font_large, fill=(150, 200, 255))
    y_cursor += 25
    q_wrapped = textwrap.fill(question[:300], width=wrap_width)
    draw.text((x_text, y_cursor), q_wrapped, font=font_normal, fill=(220, 220, 220))
    y_cursor += len(q_wrapped.split('\n')) * 18 + 30
    
    # Ground Truth
    draw.text((x_text, y_cursor), "Ground Truth:", font=font_large, fill=(100, 255, 150))
    y_cursor += 25
    gt_wrapped = textwrap.fill(ground_truth[:500], width=wrap_width)
    draw.text((x_text, y_cursor), gt_wrapped, font=font_normal, fill=(200, 255, 200))
    y_cursor += len(gt_wrapped.split('\n')) * 18 + 30
    
    # Prediction
    draw.text((x_text, y_cursor), "Prediction:", font=font_large, fill=(255, 200, 100))
    y_cursor += 25
    pred_wrapped = textwrap.fill(prediction[:500], width=wrap_width)
    draw.text((x_text, y_cursor), pred_wrapped, font=font_normal, fill=(255, 230, 180))
    
    # Save
    filename = f"epoch{epoch:02d}_sample{idx:02d}_{sample_id[:20]}.png"
    canvas.save(output_dir / filename)
    
    return output_dir / filename
