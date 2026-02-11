
import sys
import os
import torch
import glob
import json
import random
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_score

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data.dataset import QwenNuDataset
from data.splitting import scene_aware_split
from model.setup import setup_model_and_processor
from training.collate import QwenDataCollator

# Optional NuScenes
try:
    from nuscenes import NuScenes
except ImportError:
    print("Warning: nuscenes-devkit not installed. Dataset loading requiring nuScenes will fail.")
    NuScenes = None

def find_json_files(root_path):
    # Search for both *_response.json and standard .json files
    files = glob.glob(str(Path(root_path) / "**" / "*_response.json"), recursive=True)
    files += glob.glob(str(Path(root_path) / "**" / "*.json"), recursive=True)
    
    # Filter out checkpoints and runs to avoid reading log files as data
    files = [f for f in files if "checkpoints" not in f and "run_" not in f]
    
    return sorted(list(set(files))) # Remove duplicates

def calculate_metrics(predictions, references):
    """
    Calculate BLEU-4, CIDEr, SPICE, and BERTScore.
    
    predictions: dict {sample_id: [pred_text]}
    references: dict {sample_id: [ref_text1, ref_text2, ...]}
    """
    metrics = {}
    
    # Sanitize predictions for SPICE/CoreNLP
    # CoreNLP can crash on extremely long or repetitive sentences
    failed_spice = False
    clean_preds = {}
    clean_refs = {}
    
    for k, v in predictions.items():
        # Take first prediction
        p = v[0]
        # remove newlines, truncate to reasonable length (e.g. 500 chars is plenty for this task)
        p = p.replace("\n", " ").strip()
        if len(p) > 800:
            p = p[:800]
        clean_preds[k] = [p]
        
    for k, v in references.items():
        r = v[0]
        r = r.replace("\n", " ").strip()
        clean_refs[k] = [r]

    # BLEU-4, CIDEr
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider(), "CIDEr")
    ]
    
    for scorer, method in scorers:
        print(f"Computing {method}...")
        try:
            score, scores = scorer.compute_score(clean_refs, clean_preds)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    metrics[m] = s
            else:
                metrics[method] = score
        except Exception as e:
            print(f"Error computing {method}: {e}")

    # SPICE (Separate try-block as it is most fragile)
    try:
        print("Computing SPICE...")
        scorer = Spice()
        score, scores = scorer.compute_score(clean_refs, clean_preds)
        metrics["SPICE"] = score
    except Exception as e:
        print(f"Error computing SPICE: {e}")
        metrics["SPICE"] = 0.0
        failed_spice = True
        
    # BERTScore
    print("Computing BERTScore...")
    try:
        preds_list = [clean_preds[k][0] for k in sorted(clean_preds.keys())]
        refs_list = [clean_refs[k][0] for k in sorted(clean_preds.keys())]
        
        P, R, F1 = bert_score(preds_list, refs_list, lang="en", verbose=True)
        metrics["BERTScore"] = F1.mean().item()
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        metrics["BERTScore"] = 0.0
    
    return metrics

def main():
    # Configuration matches training run_20260120_135257
    config = {
        "data_root": "/home/santhru/FYP38_First Experiment/NuScenesVQA-",
        "nusc_root": "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train",
        "checkpoint_path": "/home/santhru/FYP38_First Experiment/NuScenesVQA-/scripts/checkpoints/run_20260120_135257/checkpoint-best",
        "output_dir": "/home/santhru/FYP38_First Experiment/FInal results sematic",
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "val_split": 0.1,
        "seed": 42,
        "max_samples": None,
        "system_prompt": """You are an autonomous driving assistant. Analyze the camera images to answer questions about the driving environment, traffic rules, and scene details accurately.

Thinking Process:
1. Analyze input images (Front-Left, Front, Front-Right, Back-Left, Back, Back-Right).
2. Identify key objects and road conditions.
3. Reason about the user question step-by-step.
4. Formulate the final answer."""
    }

    # Setup Output
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f" evaluating Best Checkpoint: {config['checkpoint_path']}")
    print(f" Output Directory: {output_dir}")
    print("="*80)

    # 1. Setup Data
    print(f"Initializing NuScenes...")
    if NuScenes:
        nusc = NuScenes(version='v1.0-trainval', dataroot=config['nusc_root'], verbose=False)
    else:
        raise ImportError("nuscenes-devkit is required")

    print(f"Loading data...")
    json_paths = find_json_files(config['data_root'])
    
    dataset = QwenNuDataset(
        json_paths=json_paths,
        nusc=nusc,
        max_samples=config['max_samples'],
        system_prompt=config['system_prompt']
    )
    
    # 2. Replicate Splitting Logic
    val_size = max(1, int(len(dataset) * config['val_split']))
    train_size = len(dataset) - val_size
    
    generator = torch.Generator().manual_seed(config["seed"])
    
    print(f"Splitting data: {train_size} train, {val_size} val (Scene-Aware)")
    _, val_dataset = scene_aware_split(
        dataset, 
        [train_size, val_size], 
        generator=generator, 
        nusc=nusc
    )
    
    # 3. Load Model
    print("Loading model...")
    
    # Use the setup function from the repo to ensure consistency (quantization, etc.)
    # skip_lora=True because we will load the specific checkpoint adapter afterwards
    model, processor = setup_model_and_processor(
        config['model_id'],
        use_4bit=True,          # Matches training config
        use_flash_attn=False,   # Matches training config
        skip_lora=True
    )
    
    # Fix right-padding warning for generation
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    # Load Adapter
    print(f"Loading adapter from {config['checkpoint_path']}")
    model = PeftModel.from_pretrained(model, config['checkpoint_path'])
    model.eval()
    
    # 4. Run Inference
    print("Running Inference...")
    predictions = {}
    references = {}
    samples = []
    
    # 4. Run Inference
    print("Running Inference...")
    predictions = {}
    references = {}
    samples = []
    
    # Optimize: Limit to 100 samples for faster verification
    eval_size = min(100, len(val_dataset))
    if len(val_dataset) > eval_size:
        print(f"Subsampling {eval_size} samples from validation set for speed...")
        indices = torch.randperm(len(val_dataset), generator=generator)[:eval_size].tolist()
        eval_subset = torch.utils.data.Subset(val_dataset, indices)
    else:
        eval_subset = val_dataset

    # Optimize: Batch Processing
    BATCH_SIZE = 8
    print(f"Using Batch Size: {BATCH_SIZE}")
    
    def collate_fn(batch):
        return batch

    loader = DataLoader(
        eval_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    with torch.no_grad():
        for batch_items in tqdm(loader):
            batch_prompts = []
            batch_images = []
            batch_infos = [] # Store metadata for saving results
            
            # Prepare Batch
            for item in batch_items:
                messages = item['messages']
                ground_truth = ""
                input_messages = []
                
                for msg in messages:
                    if msg['role'] == 'assistant':
                        ground_truth = msg['content'][0]['text']
                    else:
                        input_messages.append(msg)
                
                # Extract Q (for log)
                q_text = ""
                for content in input_messages[-1]['content']:
                    if content['type'] == 'text':
                        q_text = content['text']
                        break
                        
                text = processor.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
                
                batch_prompts.append(text)
                batch_images.extend(item['images']) # Qwen processor expects flat list of all images in batch order?
                # Actually, check processor docs. Qwen2-VL: images argument matches number of image tokens?
                # Usually processor(text=[t1, t2], images=[[img1], [img2]]) OR flat list?
                # HuggingFace standard for VLM is often list of lists or flattened if handled by processor.
                # Qwen2-VL processor expects 'images' as list of images corresponding to <|image_pad|> tokens.
                # Since we have 1 image per sample (Montage), it's a list.
                
                batch_infos.append({
                    "id": item['id'],
                    "question": q_text,
                    "truth": ground_truth
                })

            # Qwen2-VL Processor Batching:
            # We need to insure images map to texts correctly.
            # processor(text=..., images=...)
            # If we pass text list and images list, it should handle it.
            
            inputs = processor(
                text=batch_prompts,
                images=batch_images,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            for i, out_text in enumerate(output_texts):
                info = batch_infos[i]
                sample_id = info['id']
                
                predictions[sample_id] = [out_text]
                references[sample_id] = [info['truth']]
                
                samples.append({
                    "id": sample_id,
                    "question": info['question'],
                    "prediction": out_text,
                    "truth": info['truth']
                })

    # 5. Calculate Metrics
    # 5. Save Samples (Save BEFORE Metrics to prevent loss)
    samples_path = output_dir / "inference_samples.json"
    print(f"Saving {len(samples)} inference samples to {samples_path}...")
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=4)

    # 6. Calculate Metrics
    print("Calculating Metrics...")
    metrics = {}
    try:
        metrics = calculate_metrics(predictions, references)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Save whatever metrics we might have or empty
    
    print("Metrics:", metrics)
    
    # 7. Save Metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Done. Saved to {output_dir}")

# Helper for Qwen2-VL (simplified version of qwen_vl_utils process_vision_info)
# We might need qwen_vl_utils. 
# Let's check imports or assume standard usage.
# If qwen_vl_utils is not installed, we can rely on processor handling images if filenames passed?
# Qwen2-VL usually requires loading images manually if passed as path strings in some versions,
# OR qwen_vl_utils.
# Let's try to import or implement basic loader.

from qwen_vl_utils import process_vision_info
# If this fails, we need to install it. It is usually part of Qwen demo code.
# Assuming it's installed or we need to add it.
# If not, let's implement a simple version.

if __name__ == "__main__":
    main()
