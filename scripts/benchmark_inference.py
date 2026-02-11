
import sys
import os
import time
import torch
from pathlib import Path
from PIL import Image

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from model.setup import setup_model_and_processor
# Using ID directly from train.py to ensure match
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct" 
from utils.checkpoints import load_checkpoint
from utils.plotting import save_inference_comparison

# Checkpoint path (using the one known to have checkpoints)
CHECKPOINT_PATH = PROJECT_ROOT / "scripts/checkpoints/run_20260120_135257/checkpoint-54000"

def main():
    print(f"Benchmarking inference using checkpoint: {CHECKPOINT_PATH}")
    
    # 1. Load Model (Initialize)
    print("Initializing model (4-bit QLoRA)...")
    try:
        model, processor = setup_model_and_processor(
            MODEL_ID,
            use_4bit=True,
            use_flash_attn=False, # Match training
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # 2. Load Weights
    print("Loading adapter weights...")
    try:
        load_checkpoint(model, None, None, CHECKPOINT_PATH)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()
    
    # 3. Prepare Dummy Input
    # Create a dummy image mimicking the montage (roughly 1000x600)
    print("Preparing dummy input...")
    image = Image.new('RGB', (1008, 672), color=(100, 100, 100)) # 2 rows, 3 cols * 336
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the traffic situation."}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 4. Warmup
    print("Running warmup pass...")
    try:
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("OOM during warmup! GPU is too busy.")
            return
        raise e
        
    torch.cuda.synchronize()
    
    # 5. Benchmark
    print("Running timing benchmark (generating up to 128 tokens)...")
    start_time = time.time()
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False, # Use greedy for consistent complexity
            use_cache=True
        )
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    tokens = output.shape[1] - inputs['input_ids'].shape[1]
    
    print("-" * 50)
    print(f"Inference Time: {duration:.4f} seconds")
    print(f"Tokens Generated: {tokens}")
    print(f"Speed: {tokens/duration:.2f} tokens/sec")
    print("-" * 50)
    
    decoded = processor.decode(output[0], skip_special_tokens=True)
    print(f"Output: {decoded}")
    
    # Save visualization
    print("Saving visualization...")
    out_dir = PROJECT_ROOT / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    
    img_path = save_inference_comparison(
        image=image,
        question="Describe the traffic situation.",
        ground_truth="N/A (Benchmark - Dummy Input)",
        prediction=decoded,
        bleu_score=0.0,
        sample_id="benchmark_live",
        output_dir=out_dir,
        epoch=0,
        idx=0
    )
    print(f"Saved visualization to: {img_path}")

if __name__ == "__main__":
    main()
