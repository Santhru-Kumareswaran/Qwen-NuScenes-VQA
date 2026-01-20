
import sys
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import QwenNuDataset
from model.setup import setup_model_and_processor
from nuscenes.nuscenes import NuScenes
from configs.constants import VIEW_ORDER

def parse_args():
    parser = argparse.ArgumentParser(description="Run Inference with Visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory (adapter)")
    parser.add_argument("--data_root", type=str, default="/data/nuscenes", help="Path to NuScenes root")
    parser.add_argument("--val_json", type=str, default="/home/santhru/FYP38_First Experiment/NuScenesVQA-/data/processed/v1.0-mini/val.json", help="Path to validation JSON")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="Base model ID")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to run")
    parser.add_argument("--system_prompt", type=str, default="You are an autonomous driving assistant. Analyze the camera images to answer questions about the driving environment, traffic rules, and scene details accurately.\n\nThinking Process:\n1. Analyze input images (Front-Left, Front, Front-Right, Back-Left, Back, Back-Right).\n2. Identify key objects and road conditions.\n3. Reason about the user question step-by-step.\n4. Formulate the final answer.", help="System Prompt")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max generation tokens")
    return parser.parse_args()

def create_visualization(image, question, gt, pred, metrics, output_path):
    """
    Create a composite image with the input image on the left/top 
    and text (Q, GT, Pred) on the right/bottom.
    """
    # Create Figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Image
    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title("Input Montage", fontsize=14)
    
    # 2. Text
    ax_text = fig.add_subplot(1, 2, 2)
    ax_text.axis("off")
    
    # Text Content
    text_content = (
        f"QUESTION:\n{question}\n\n"
        f"GROUND TRUTH:\n{gt}\n\n"
        f"PREDICTION:\n{pred}\n\n"
        f"METRICS:\n"
        f"BLEU-4: {metrics['bleu4']:.4f}\n"
        f"ROUGE-L: {metrics['rouge_l']:.4f}\n"
        f"METEOR: {metrics['meteor']:.4f}"
    )
    
    # Wrap text roughly
    import textwrap
    lines = []
    for para in text_content.split('\n'):
        lines.extend(textwrap.wrap(para, width=60))
        
    final_text = "\n".join(lines)
    
    ax_text.text(0.05, 0.95, final_text, fontsize=12, family='monospace', va='top', wrap=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading NuScenes from {args.data_root}...")
    try:
        # Try trainval first as it's the main path
        nusc = NuScenes(version='v1.0-trainval', dataroot=args.data_root, verbose=False)
    except:
        try:
             # Fallback to mini
             nusc = NuScenes(version='v1.0-mini', dataroot=args.data_root, verbose=False)
        except:
             print("Warning: Could not load NuScenes. 'v1.0-trainval' or 'v1.0-mini' not found.")
             nusc = None
        
    print(f"Loading Model from {args.checkpoint}...")
    # Load model with Adapter
    # Passing 'adapter_path' to setup function if supported, or load matching base then load_adapter
    # Our setup.py usually takes model_id. 
    # We load base, then load adapter.
    
    model, processor = setup_model_and_processor(
        args.base_model,
        use_4bit=True,
        use_flash_attn=False, # Use SDPA
        lora_r=16 # Default
    )
    
    from peft import PeftModel
    print(f"Loading Adapter from {args.checkpoint}...")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()
    
    print("Loading Dataset...")
    dataset = QwenNuDataset(
        json_paths=[args.val_json],
        nusc=nusc,
        view_order=VIEW_ORDER,
        max_samples=args.num_samples if args.num_samples > 0 else None,
        system_prompt=args.system_prompt
    )
    
    print(f"Running inference on {len(dataset)} samples...")
    
    metrics_summary = {"bleu4": [], "rouge_l": [], "meteor": []}
    inference_results = []
    
    # Store for Corpus Metrics (CIDEr, SPICE)
    gts_dict = {}
    res_dict = {}
    
    chencherry = SmoothingFunction()
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        sample_id = item['id']
        messages = item['messages']
        image = item['images'][0]
        
        # Extract Prompt and GT
        # Messages: [System, User, Assistant]
        conversation = [msg for msg in messages if msg['role'] != 'assistant']
        gt_msg = messages[-1]
        gt_answer = gt_msg['content'][0]['text']
        question = conversation[-1]['content'][1]['text'] # User text
        
        # Prepare Input
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False, # Deterministic for evaluation
                num_beams=1
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        pred_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        
        # Metrics
        ref_tokens = [gt_answer.split()]
        hyp_tokens = pred_text.split()
        
        # BLEU-4
        bleu4 = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=chencherry.method1)
        
        # METEOR (NLTK expects list of references, and hypothesis string tokens?)
        # nltk.translate.meteor_score.meteor_score(references, hypothesis)
        # References: list of list of str (or list of str?)
        # Docs: references (list(str)), hypothesis (str) - Wait, it usually expects tokenized list?
        # NLTK version dependent. Usually references=[ref_str], hypothesis=hyp_str (pre-tokenized or not?)
        # Actually meteor_score expects tokenized lists usually.
        # Let's check NLTK version. 3.9+. 
        # meteor_score([['this', 'is', ...]], ['this', 'is', ...])
        meteor = meteor_score([gt_answer.split()], pred_text.split())
        
        # ROUGE-L
        scores = rouge.score(gt_answer, pred_text)
        rouge_l = scores['rougeL'].fmeasure
        
        metrics_summary["bleu4"].append(bleu4)
        metrics_summary["rouge_l"].append(rouge_l)
        metrics_summary["meteor"].append(meteor)
        
        current_metrics = {"bleu4": bleu4, "rouge_l": rouge_l, "meteor": meteor}
        
        # Collect for Corpus Metrics (Keys must be str)
        gts_dict[str(sample_id)] = [gt_answer]
        res_dict[str(sample_id)] = [pred_text]
        
        # Save Result
        res = {
            "id": sample_id,
            "question": question,
            "ground_truth": gt_answer,
            "prediction": pred_text,
            "metrics": current_metrics
        }
        inference_results.append(res)
        
        # Visualize
        vis_path = output_dir / f"vis_{sample_id}.png"
        create_visualization(image, question, gt_answer, pred_text, current_metrics, vis_path)
        
    # Calculate Corpus Metrics (CIDEr, SPICE)
    print("Calculating CIDEr...")
    try:
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts_dict, res_dict)
        avg_metrics["cider"] = cider_score
    except Exception as e:
        print(f"Error calculating CIDEr: {e}")
        avg_metrics["cider"] = 0.0

    print("Calculating SPICE (Requires Java)...")
    try:
        spice_scorer = Spice()
        spice_score, _ = spice_scorer.compute_score(gts_dict, res_dict)
        avg_metrics["spice"] = spice_score
    except Exception as e:
        print(f"Error calculating SPICE (Check Java): {e}")
        avg_metrics["spice"] = 0.0
        
    summary = {
        "overall_metrics": avg_metrics,
        "samples": inference_results
    }
    
    with open(output_dir / "inference_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nInference Complete!")
    print(f"Average BLEU-4: {avg_metrics['bleu4']:.4f}")
    print(f"Average ROUGE-L: {avg_metrics['rouge_l']:.4f}")
    print(f"Average METEOR: {avg_metrics['meteor']:.4f}")
    print(f"Average CIDEr: {avg_metrics['cider']:.4f}")
    print(f"Average SPICE: {avg_metrics['spice']:.4f}")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
