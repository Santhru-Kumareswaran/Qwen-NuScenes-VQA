
import sys
import os
import random
import torch
import glob
import json
from pathlib import Path
from tqdm import tqdm

try:
    from nuscenes import NuScenes
except ImportError:
    print("Error: nuscenes-devkit is required but not installed.")
    sys.exit(1)

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    sentence_bleu = None
    print("Warning: nltk not installed. BLEU scores will be 0.0")

try:
    from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
except ImportError:
    print("Error: scikit-learn is required but not installed.")
    sys.exit(1)

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from model.setup import setup_model_and_processor
from utils.checkpoints import load_checkpoint
from utils.plotting import save_inference_comparison
from data.dataset import QwenNuDataset
from data.splitting import scene_aware_split
from training.collate import QwenDataCollator

# Configuration 
CONFIG = {
    "model_id": "Qwen/Qwen3-VL-2B-Instruct", 
    "use_4bit": True,
    "use_flash_attn": False,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "max_samples": None,
    # Data paths
    "json_path": "/home/santhru/FYP38_First Experiment/NuScenesVQA-/nuscenes_action_prediction_vqa.json",
    "nusc_root": "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train",
    "output_dir": PROJECT_ROOT / "inference_action_results"
}

SYSTEM_PROMPT = """You are an autonomous driving action planner. Your task is to analyze the provided multi-view camera and the navigation goal to determine the immediate high-level action required.

Input:
1. Visual Context: 6-view camera grid (Front, Back, Front-Left, Front-Right, Back-Left, Back-Right).
2. Navigation Goal: One of the following instructions from the navigation system:
   - GO_STRAIGHT
   - PREPARE_LEFT
   - PREPARE_RIGHT
   - TURN_LEFT
   - TURN_RIGHT

Output:
Predict the specific 'action token' that safely executes the goal in the current scene. Choose strictly from the following allowed tokens:
   - STRAIGHT_FAST
   - STRAIGHT_SLOW
   - LEFT_TURN
   - LEFT_TURN_SHARP
   - RIGHT_TURN
   - RIGHT_TURN_SHARP
   - U_TURN_LEFT_TIGHT
   - U_TURN_LEFT_WIDE
   - LEFT_SLIDE
   - LEFT_SLIDE_GENTLE
   - RIGHT_SLIDE
   - RIGHT_SLIDE_GENTLE
   - STATIONARY
   - CREEPING

You are in @ ACTION_MODE"""

ALLOWED_ACTIONS = [
    "STRAIGHT_FAST", "STRAIGHT_SLOW", "LEFT_TURN", "LEFT_TURN_SHARP",
    "RIGHT_TURN", "RIGHT_TURN_SHARP", "U_TURN_LEFT_TIGHT", "U_TURN_LEFT_WIDE",
    "LEFT_SLIDE", "LEFT_SLIDE_GENTLE", "RIGHT_SLIDE", "RIGHT_SLIDE_GENTLE",
    "STATIONARY", "CREEPING"
]

class InferenceCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        from qwen_vl_utils import process_vision_info
        
        inference_messages = []
        for item in batch:
            msgs = [m for m in item["messages"] if m["role"] != "assistant"]
            inference_messages.append(msgs)
            
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in inference_messages
        ]
        
        image_inputs = []
        for item in batch:
            input_images, _ = process_vision_info(item["messages"])
            image_inputs.append(input_images)
            
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            min_pixels=28*28*4,
            max_pixels=28*28*256
        )
        
        return inputs


def extract_action(text):
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data.get("action", text)
    except:
        pass
    import re
    match = re.search(r'"action":\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    return text

def calculate_weighted_score(gt_raw, pred_raw):
    gt = extract_action(gt_raw).strip().upper()
    pred = extract_action(pred_raw).strip().upper()
    
    if gt == pred:
        return 1.0, "Exact Match"
    
    # Define Rules
    # Speed & Progression
    pair = {gt, pred}
    if pair == {"CREEPING", "STATIONARY"}: return 0.5, "Safe (Near Stop)"
    if pair == {"CREEPING", "STRAIGHT_SLOW"}: return 0.6, "Safe (Slow Progression)"
    if pair == {"CREEPING", "STRAIGHT_FAST"}: return 0.2, "Risky (Speed Mismatch)"
    if pair == {"STRAIGHT_SLOW", "STRAIGHT_FAST"}: return 0.7, "Action OK (Speed Mismatch)"
    
    # Lateral Shifts (Slides) vs Straight
    if ("STRAIGHT" in gt and "SLIDE" in pred) or ("SLIDE" in gt and "STRAIGHT" in pred):
        if "GENTLE" in gt or "GENTLE" in pred:
            return 0.6, "Safe (Maneuver Mismatch)"
        return 0.3, "Risky (Maneuver Mismatch)"
    
    if "SLIDE" in gt and "SLIDE" in pred:
        if ("LEFT" in gt and "LEFT" in pred) or ("RIGHT" in gt and "RIGHT" in pred):
            return 0.7, "Action OK (Slide Logic)"
            
    # Turning Maneuvers
    if "TURN" in gt and "TURN" in pred:
        if ("LEFT" in gt and "LEFT" in pred) or ("RIGHT" in gt and "RIGHT" in pred):
            return 0.7, "Action OK (Turn Intensity)"
            
    # U-Turns
    if "U_TURN" in gt and "U_TURN" in pred:
        if ("LEFT" in gt and "LEFT" in pred) or ("RIGHT" in gt and "RIGHT" in pred):
            return 0.7, "Action OK (U-Turn Radius)"
            
    # Turning Maneuvers vs Slides (Side Match)
    if (("TURN" in gt and "SLIDE" in pred) or ("SLIDE" in gt and "TURN" in pred)):
        if ("LEFT" in gt and "LEFT" in pred) or ("RIGHT" in gt and "RIGHT" in pred):
            return 0.4, "Side Match (Maneuver Mismatch)"
            
    return 0.0, "Critical Failure"

def main():
    print("=" * 60)
    print("   NuScenesVQA Action Planner Inference Script   ")
    print("=" * 60)
    
    # 1. Setup Model
    print("\n[1/5] Loading Model...")
    model, processor = setup_model_and_processor(
        CONFIG["model_id"],
        use_4bit=CONFIG["use_4bit"],
        use_flash_attn=CONFIG["use_flash_attn"],
        lora_r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"]
    )
    
    # 2. Load Checkpoint
    checkpoint_path = Path("/home/santhru/FYP38_First Experiment/NuScenesVQA-/scripts/checkpoints/run_20260126_022550/checkpoint-best")
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"\n[2/5] Loading weights from: {checkpoint_path}")
    try:
        load_checkpoint(model, None, None, checkpoint_path)
    except Exception as e:
        print(f"Warning loading checkpoint: {e}")
        
    model.eval()

    # 3. Load Data
    print("\n[3/5] Initializing NuScenes & Dataset...")
    if not os.path.exists(CONFIG['nusc_root']):
        print(f"Error: NuScenes root not found at {CONFIG['nusc_root']}")
        return

    nusc = NuScenes(version='v1.0-trainval', dataroot=CONFIG['nusc_root'], verbose=False)
    
    json_paths = [CONFIG['json_path']]
    print(f"Using Action Prediction JSON: {json_paths[0]}")
    
    full_dataset = QwenNuDataset(
        json_paths=json_paths,
        nusc=nusc,
        max_samples=CONFIG["max_samples"],
        system_prompt=SYSTEM_PROMPT
    )
    
    total = len(full_dataset)
    val_len = int(total * 0.1)
    train_len = total - val_len
    lengths = [train_len, val_len]
    generator = torch.Generator().manual_seed(42)
    
    print(f"Splitting dataset (Total: {total})...")
    subsets = scene_aware_split(full_dataset, lengths, generator=generator, nusc=nusc)
    val_dataset = subsets[1]
    
    print(f"Validation set size: {len(val_dataset)}")
    
    # 4. Select Samples
    print("\n[4/5] Selecting 100 Random Samples...")
    subset_indices = random.sample(range(len(val_dataset)), min(100, len(val_dataset)))
    
    collate_fn = InferenceCollator(processor)
    final_subset = torch.utils.data.Subset(val_dataset, subset_indices)
    dataloader = torch.utils.data.DataLoader(final_subset, batch_size=1, collate_fn=collate_fn)
    
    # 5. Run Inference
    print("\n[5/5] Running Inference...")
    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(exist_ok=True, parents=True)
    
    total_safety_score = 0.0
    total_bleu_score = 0.0
    exact_match_count = 0
    total_processed = 0
    
    y_true = []
    y_pred = []
    
    for i, batch in enumerate(tqdm(dataloader)):
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'metadata' and k != 'labels'}
        raw_item = final_subset[i]
        
        msgs = raw_item["messages"]
        user_msg = next((m for m in msgs if m["role"] == "user"), None)
        asst_msg = next((m for m in msgs if m["role"] == "assistant"), None)
        
        question = user_msg["content"][1]["text"] if user_msg else "N/A"
        gt_answer = asst_msg["content"][0]["text"] if asst_msg else "N/A"
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=128,
                do_sample=False
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.get('input_ids'), generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Extract actions for classification metrics
        gt_action = extract_action(gt_answer).strip().upper()
        pred_action = extract_action(output_text).strip().upper()
        
        # Handle cases where model might output something weird
        if pred_action not in ALLOWED_ACTIONS:
            # Try to find if the predicted text contains any allowed action as a substring
            found = False
            for action in ALLOWED_ACTIONS:
                if action in pred_action:
                    pred_action = action
                    found = True
                    break
            if not found:
                pred_action = "STATIONARY" # Default or keep as is? Let's keep as is but it might break report
        
        y_true.append(gt_action)
        y_pred.append(pred_action)
        
        # 1. Weighted Safety Scoring
        safety_score, safety_status = calculate_weighted_score(gt_answer, output_text)
        total_safety_score += safety_score
        if safety_score == 1.0:
            exact_match_count += 1
        
        # 2. BLEU-4 Scoring
        sample_bleu = 0.0
        if sentence_bleu:
            try:
                ref = [gt_answer.lower().split()]
                hyp = output_text.lower().split()
                sample_bleu = sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)
                total_bleu_score += sample_bleu
            except Exception as e:
                print(f"BLEU calc error: {e}")
        
        total_processed += 1
        
        # Save visualization with BOTH scores
        save_inference_comparison(
            image=raw_item['images'][0],
            question=question,
            ground_truth=gt_answer,
            prediction=output_text,
            bleu_score=sample_bleu,
            safety_score=safety_score,
            safety_status=safety_status,
            sample_id=raw_item['id'],
            output_dir=output_dir,
            epoch=0,
            idx=i
        )
        
    avg_exact = (exact_match_count / total_processed) * 100 if total_processed > 0 else 0
    avg_weighted = (total_safety_score / total_processed) * 100 if total_processed > 0 else 0
    avg_bleu = (total_bleu_score / total_processed) * 100 if total_processed > 0 else 0
    
    print("\n" + "=" * 60)
    print("   Final Weighted Inference Summary   ")
    print("=" * 60)
    print(f"Total Samples: {total_processed}")
    print(f"Exact Match Accuracy:  {avg_exact:.2f}%")
    print(f"Weighted Safety Score: {avg_weighted:.2f}%")
    print(f"Average BLEU-4:        {avg_bleu:.4f}")
    
    # ---------------------------------------------------------
    # NEW: Classification Metrics
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("   Detailed Classification Report   ")
    print("=" * 60)
    
    # Use only labels present in the dataset + allowed actions to avoid clutter but maintain structure
    present_labels = sorted(list(set(y_true + y_pred)))
    # For a clean confusion matrix, we might want to restrict to ALLOWED_ACTIONS
    report_labels = [a for a in ALLOWED_ACTIONS if a in present_labels]
    
    print(classification_report(y_true, y_pred, labels=report_labels, zero_division=0))
    
    print("\n" + "=" * 60)
    print("   Confusion Matrix   ")
    print("=" * 60)
    cm = confusion_matrix(y_true, y_pred, labels=report_labels)
    
    # Print header
    header = " " * 20 + " | ".join([f"{l[:6]:>6}" for l in report_labels])
    print(header)
    print("-" * len(header))
    
    for i, row in enumerate(cm):
        row_str = f"{report_labels[i][:18]:<18} | " + " | ".join([f"{val:>6}" for val in row])
        print(row_str)

    # Save metrics to JSON
    metrics_summary = {
        "total_samples": total_processed,
        "exact_match_accuracy": avg_exact,
        "weighted_safety_score": avg_weighted,
        "average_bleu4": avg_bleu,
        "per_class_metrics": classification_report(y_true, y_pred, labels=report_labels, output_dict=True, zero_division=0)
    }
    
    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"\nResults and metrics summary saved to {output_dir}")

if __name__ == "__main__":
    main()
