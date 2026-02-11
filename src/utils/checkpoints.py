
import torch
from pathlib import Path

def save_checkpoint(
    model, 
    processor, 
    optimizer, 
    scheduler, 
    epoch, 
    step, 
    loss, 
    output_dir: Path,
    keep_last_n=3,
    is_best=False
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create step-specific folder
    if is_best:
        ckpt_dir = output_dir / "checkpoint-best"
    else:
        ckpt_dir = output_dir / f"checkpoint-{step}"
    
    # Remove existing best if overwriting
    if is_best and ckpt_dir.exists():
        import shutil
        shutil.rmtree(ckpt_dir)
        
    ckpt_dir.mkdir(exist_ok=True)
    
    # Save Model (LoRA adapters)
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    
    # Save Optimizer/Scheduler
    torch.save({
        'epoch': epoch,
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss
    }, ckpt_dir / "trainer_state.pt")
    
    # Rotate checkpoints (ONLY for regular checkpoints)
    if not is_best and keep_last_n > 0:
        # Find all checkpoint folders
        # Exclude checkpoint-best from rotation
        all_ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        ckpts = [p for p in all_ckpts if "best" not in p.name]
        
        if len(ckpts) > keep_last_n:
            to_remove = ckpts[:-keep_last_n]
            for folder in to_remove:
                import shutil
                shutil.rmtree(folder)
                print(f"Removed old checkpoint {folder}")
                
    print(f"Saved checkpoint to {ckpt_dir} ({'BEST' if is_best else 'Regular'})")

def load_checkpoint(model, optimizer, scheduler, ckpt_dir: Path):
    # Load adapters
    # model.load_adapter(ckpt_dir, "default") # if using PEFT
    # Usually PEFT model loads adapters automatically if passed to from_pretrained or load_adapter
    # But here we assume we are resuming training.
    
    # For PEFT, we usually load base model then load adapters.
    # If model is already PEFT, we might need `load_peft_weights`.
    from peft import PeftModel, load_peft_weights
    
    print(f"Loading checkpoint from {ckpt_dir}")
    # Ensure model is PeftModel
    if isinstance(model, PeftModel):
        try:
            model.load_adapter(ckpt_dir, "default")
        except Exception:
            try:
                weights = load_peft_weights(ckpt_dir)
                model.load_state_dict(weights, strict=False)
            except Exception as e:
                print(f"Warning: failed to load PEFT adapter weights: {e}")
    else:
        # Fallback: no-op for non-PEFT models
        pass
        
    # Load trainer state
    state_path = ckpt_dir / "trainer_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        if optimizer:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        if scheduler and state['scheduler_state_dict']:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        return state['epoch'], state['step']
    return 0, 0
