
import sys
import os
# Fix OOM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    Optimized for RTX 5080 Laptop GPU (16GB VRAM) with QLoRA and Flash Attention 2.
    
    System specs:
    - GPU: RTX 5080 Laptop (16GB VRAM, Compute Cap 12.0)
    - CPU: Intel Core Ultra 9 275HX (24 cores)
    - RAM: 30GB
    """
    config = {
        # ──────────────────────────────────────────────────────────────────
        # DEBUG MODE - Set to True for verbose logging during testing
        # ──────────────────────────────────────────────────────────────────
        "debug": False,           # Disabled for production run
        
        # ──────────────────────────────────────────────────────────────────
        # I/O Paths
        # ──────────────────────────────────────────────────────────────────
        "data_root": "/home/santhru/FYP38_First Experiment/NuScenesVQA-/scripts",
        "nusc_root": "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train",
        "output_dir": "./checkpoints",
        
        # ──────────────────────────────────────────────────────────────────
        # Data Config (~50k samples)
        # ──────────────────────────────────────────────────────────────────
        "max_samples": None,      # Use ALL samples (~50k) - set to int for testing
        "val_split": 0.1,         # 10% for validation (~5k val, ~45k train)
        "num_workers": 8,
        "prefetch_factor": 1,
        
        # ──────────────────────────────────────────────────────────────────
        # Training Config for Long Run (50 epochs, ~50k samples)
        # Steps per epoch: ~45000 / 16 = ~2812 steps/epoch
        # Total steps: 2812 * 50 = ~140,600 steps
        # ──────────────────────────────────────────────────────────────────
        "epochs": 50,             # 50 epochs (can stop anytime)
        "batch_size": 2,
        "grad_accum": 8,          # Effective batch size: 16
        "lr": 2e-4,
        "weight_decay": 0.01,
        "warmup_steps": 3000,     # ~1 epoch warmup (good for long runs)
        "save_steps": 500,        # Frequent saves for resumability (~6 saves/epoch)
        "keep_last_n": 5,         # Keep more checkpoints for safety
        "plot_every": 500,        # Plot every 500 steps
        "val_every_steps": 1400,  # Validate every half-epoch (~2812/2 = 1406)
        "gradient_checkpointing": True,
        
        "resume": False,          # Set to True to resume from latest checkpoint
        "resume_from_best": False,
        
        # ──────────────────────────────────────────────────────────────────
        # Model Config (Optimized for RTX 5080)
        # ──────────────────────────────────────────────────────────────────
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "use_4bit": True,
        "use_flash_attn": False,
        "use_8bit_optimizer": True,
        "system_prompt": "You are an autonomous driving assistant. Analyze the camera images to answer questions about the driving environment, traffic rules, and scene details accurately.\n\nThinking Process:\n1. Analyze input images (Front-Left, Front, Front-Right, Back-Left, Back, Back-Right).\n2. Identify key objects and road conditions.\n3. Reason about the user question step-by-step.\n4. Formulate the final answer.",
        "max_ans_toks": 256,
        
        # QLoRA Config
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "llm_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # ──────────────────────────────────────────────────────────────────
        # Inference Config
        # ──────────────────────────────────────────────────────────────────
        "inference_sampling_every_epochs": 5,  # Run inference every 5 epochs
        "inference_samples_n": 4, 
        "inference_max_tokens": 256,
        "inference_temperature": 0.7,
        "inference_do_sample": True,
        "inference_num_beams": 1,
        "inference_sampling_every": 14000,     # Every 5 epochs (~2812 * 5)
        
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
    
    # Filter out checkpoints and runs to avoid reading log files as data
    files = [f for f in files if "checkpoints" not in f and "run_" not in f]
    
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
            ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else 0)
            latest = ckpts[-1] if ckpts else "?"
            
            # Check for best
            best_ckpt = run_dir / "checkpoint-best"
            has_best = " (Has Best)" if best_ckpt.exists() else ""
            
            print(f"  [{idx}] {run_dir.name} (Latest: {latest.name if isinstance(latest, Path) else latest}){has_best}")
            
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
                    
                    resume_from_best = config.get("resume_from_best", False)
                    best_ckpt = selected / "checkpoint-best"
                    
                    if resume_from_best:
                        if best_ckpt.exists():
                            print("Resuming from BEST checkpoint.")
                            return selected, best_ckpt
                        else:
                            print("Warning: resume_from_best=True but no 'checkpoint-best' found. Falling back to latest.")
                    
                    # Find latest checkpoint automatically
                    ckpts = sorted(selected.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else 0)
                    # Filter out non-step checkpoints if mixed (though sorted should handle it if careful, best to filter)
                    ckpts = [p for p in ckpts if p.name != "checkpoint-best"]
                    
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
    print(f"Mode: {'QLoRA (4-bit NF4 + bf16)' if config['use_4bit'] else 'LoRA (bf16)'}")
    print(f"Flash Attention 2: {'Enabled' if config['use_flash_attn'] else 'Disabled'}")
    print(f"8-bit Optimizer: {'Enabled' if config.get('use_8bit_optimizer', False) else 'Disabled'}")
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

    system_prompt = config.get("system_prompt", "")
    dataset = QwenNuDataset(
        json_paths=json_paths,
        nusc=nusc,
        max_samples=config['max_samples'],
        system_prompt=system_prompt
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
    
    # Use 8-bit Adam optimizer for memory savings (optional)
    if config.get('use_8bit_optimizer', False):
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(), 
                lr=config['lr'], 
                weight_decay=config['weight_decay']
            )
            print("[Optimizer] Using 8-bit AdamW (bitsandbytes)")
        except ImportError:
            print("[Optimizer] bitsandbytes not available, using standard AdamW")
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Initial Scheduler (Linear Warmup)
    # Use ceil to account for the final step in each epoch if len(train_loader) is not divisible by grad_accum
    import math
    num_update_steps_per_epoch = math.ceil(len(train_loader) / config['grad_accum'])
    total_steps = num_update_steps_per_epoch * config['epochs']
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
