
import sys
import os
import torch
import glob
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from configs.constants import MODEL_ID
from data.dataset import QwenNuDataset, load_json_any
from data.splitting import scene_aware_split
from model.setup import setup_model_and_processor
from training.trainer import QwenTrainer
from training.collate import QwenDataCollator

# Optional NuScenes
try:
    from nuscenes import NuScenes
except ImportError:
    print("Warning: nuscenes-devkit not installed. Dataset loading requiring nuScenes will fail.")
    NuScenes = None

def get_training_config() -> dict:
    """
    Get comprehensive training configuration.
    Modified to be standalone config like the reference implementation.
    """
    config = {
        # ──────────────────────────────────────────────────────────────────
        # I/O Paths
        # ──────────────────────────────────────────────────────────────────
        # Hardcoded paths from predecessor codebase (LiDAR-Vision-VQA/src/encoder-decoder/train.py) # Do not change
        "data_root": "/home/j_bindu/fyp-26-grp-38/Dataset_subset/external",
        "nusc_root": "/home/j_bindu/fyp-26-grp-38/Dataset_subset",
        "output_dir": "./checkpoints",
        
        # ──────────────────────────────────────────────────────────────────
        # Data Config
        # ──────────────────────────────────────────────────────────────────
        "max_samples": None,      # Debug: Limit to N samples (None = all)
        "val_split": 0.1,         # 10% for validation
        "num_workers": 8,         # 28 Cores available -> Use 8
        "prefetch_factor": 2,     # Buffer size
        
        # ──────────────────────────────────────────────────────────────────
        # Training Config
        # ──────────────────────────────────────────────────────────────────
        "epochs": 20,
        "batch_size": 1,          # V100 16GB: 1 is best for 3B VL model with large images
        "grad_accum": 16,         # Effective batch size = 1 * 16 = 16
        "lr": 2e-4,
        "weight_decay": 0.01,
        "warmup_steps": 2000,     # ~10% of total steps (48k samples -> ~3k steps/epoch -> 60k steps -> 2k warmup is conservative)
        "save_steps": 1000,
        "keep_last_n": 3,
        "plot_every": 1,
        "gradient_checkpointing": True,
        
        "resume": False,          # Resume from latest checkpoint
        "resume_from_best": False,# Not implemented in simplified trainer yet, default to latest
        
        # ──────────────────────────────────────────────────────────────────
        # Model Config
        # ──────────────────────────────────────────────────────────────────
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "use_4bit": False,        # False = float16, True = QLoRA (nf4)
        "use_flash_attn": False,  # Not supported
        "system_prompt": "You are an autonomous driving assistant. Analyze the camera images to answer questions about the driving environment, traffic rules, and scene details accurately.",
        "max_ans_toks": 256,
        
        # LoRA Config
        "lora_r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.3,
        "llm_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # ──────────────────────────────────────────────────────────────────
        # Inference Config
        # ──────────────────────────────────────────────────────────────────
        "inference_sampling_every": 500, # Steps 
        "inference_samples_n": 4, 
        "inference_max_tokens": 256,
        "inference_temperature": 0.0,
        "inference_do_sample": False,
        "inference_num_beams": 1,
        
        # ──────────────────────────────────────────────────────────────────
        # Metrics
        # ──────────────────────────────────────────────────────────────────
        "eval_caption_bleu4": True,
        "eval_caption_cider": True,
        "eval_caption_spice": False,
        "eval_caption_bertscore": False,
    }
    return config

def find_json_files(root_path):
    # Search for both *_response.json and standard .json files
    files = glob.glob(str(Path(root_path) / "**" / "*_response.json"), recursive=True)
    files += glob.glob(str(Path(root_path) / "**" / "*.json"), recursive=True)
    return sorted(list(set(files))) # Remove duplicates

def setup_output_directory(config: dict) -> Path:
    base_out_dir = Path(config["output_dir"])
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    resume = config.get("resume", False)
    
    # 1. Look for existing run subdirectories
    run_dirs = sorted(base_out_dir.glob("run_*"), reverse=True)
    
    valid_runs = []
    # Identify runs with checkpoints
    for r in run_dirs:
        if list(r.glob("checkpoint-*")):
            valid_runs.append(r)
            
    if resume and valid_runs:
        print("\n" + "=" * 80)
        print("RESUME TRAINING: Select a run to resume from")
        print("=" * 80)
        print(f"  [0] START NEW RUN")
        
        for idx, run_dir in enumerate(valid_runs, start=1):
            # Find latest checkpoint
            ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[-1]))
            latest = ckpts[-1] if ckpts else "?"
            print(f"  [{idx}] {run_dir.name} (Latest: {latest.name if isinstance(latest, Path) else latest})")
            
        print("=" * 80)
        
        while True:
            try:
                choice = input(f"Enter your choice [0-{len(valid_runs)}]: ").strip()
                idx = int(choice)
                if idx == 0:
                    break # New run
                elif 1 <= idx <= len(valid_runs):
                    selected = valid_runs[idx-1]
                    print(f"Resuming from: {selected}")
                    # Find latest checkpoint automatically
                    ckpts = sorted(selected.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[-1]))
                    if ckpts:
                        return selected, ckpts[-1]
                    else:
                        print("Run has no checkpoints!")
                else:
                    print("Invalid.")
            except ValueError:
                print("Invalid input.")
    
    # New run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    return run_dir, None

def main():
    # 1. Load Config
    config = get_training_config()
    
    # 2. Output Dir & Resume Logic
    run_dir, resume_ckpt = setup_output_directory(config)
    print(f"Run Directory: {run_dir}")
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    
    print("="*80)
    print(f"Model: {config['model_id']}")
    print(f"Data Root: {config['data_root']}")
    print(f"Dtype: {'4-bit NF4' if config['use_4bit'] else 'float16'}")
    print("="*80)

    # 3. Setup Data
    print(f"Initializing NuScenes...")
    if NuScenes:
        nusc = NuScenes(version='v1.0-trainval', dataroot=config['nusc_root'], verbose=False)
    else:
        raise ImportError("nuscenes-devkit is required")
        
    print(f"Searching for data...")
    json_paths = find_json_files(config['data_root'])
    print(f"Found {len(json_paths)} JSON annotation files.")
    
    if len(json_paths) == 0:
        print("No data found! Exiting.")
        return

    dataset = QwenNuDataset(
        json_paths=json_paths,
        nusc=nusc,
        max_samples=config['max_samples']
    )
    print(f"Dataset size: {len(dataset)}")
    
    # 4. Setup Model & Processor
    model, processor = setup_model_and_processor(
        config['model_id'],
        use_4bit=config['use_4bit'], 
        use_flash_attn=config['use_flash_attn'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout']
    )
    
    # 4. Data Splitting
    val_size = max(1, int(len(dataset) * config['val_split']))
    train_size = len(dataset) - val_size
    
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    
    # Use Scene-Aware Split to prevent leakage
    print(f"Splitting data: {train_size} train, {val_size} val (Scene-Aware)")
    train_dataset, val_dataset = scene_aware_split(
        dataset, 
        [train_size, val_size], 
        generator=generator, 
        nusc=nusc
    )
    
    # 5. Setup Trainer Components
    collate_fn = QwenDataCollator(processor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        prefetch_factor=config['prefetch_factor'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Initial Scheduler (Linear Warmup)
    total_steps = len(train_loader) // config['grad_accum'] * config['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['lr'], 
        total_steps=total_steps, 
        pct_start=config['warmup_steps']/total_steps if total_steps > 0 else 0.1,
        anneal_strategy='cos'
    )

    trainer = QwenTrainer(
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_dir=run_dir
    )
    
    # 6. Train
    print("Starting Training...")
    trainer.train(epochs=config['epochs'], resume_from=resume_ckpt)

if __name__ == "__main__":
    main()
