
import json
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Sequence, Any
from nuscenes.nuscenes import NuScenes

from configs.constants import VIEW_ORDER
from data.montage import MontageBuilder

def load_json_any(path: str):
    """Load JSON from file or JSONL format."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

class QwenNuDataset(Dataset):
    """
    Dataset for Qwen2.5-VL VQA on nuScenes.
    Stitches 6 camera views into a montage with visual prompts.
    """
    
    def __init__(
        self,
        json_paths: List[str],
        nusc: Optional[NuScenes] = None,
        view_order: Sequence[str] = VIEW_ORDER,
        max_samples: Optional[int] = None,
        montage_builder: Optional[MontageBuilder] = None
    ):
        self.rows = []
        for jp in json_paths:
            data = load_json_any(jp)
            self.rows.extend(data)
            
        if max_samples:
            import random
            random.seed(42)
            random.shuffle(self.rows)
            self.rows = self.rows[:max_samples]
            
        self.nusc = nusc
        self.view_order = view_order
        self.montage_builder = montage_builder or MontageBuilder()
        
    def _resolve_paths(self, sample_token: str) -> Dict[str, np.ndarray]:
        """
        Get images for a sample token.
        """
        if self.nusc is None:
            # Fallback for when we don't have nusc but maybe have paths in json?
            # NOT IMPLEMENTED based on user constraints (old code relied on nusc)
            return {}
            
        images = {}
        try:
            sample = self.nusc.get("sample", sample_token)
            for cam in self.view_order:
                sd_tok = sample["data"].get(cam)
                if sd_tok:
                    sd = self.nusc.get("sample_data", sd_tok)
                    # Path resolution
                    # Assuming nusc.dataroot is correct
                    p = (Path(self.nusc.dataroot) / sd["filename"]).resolve()
                    if p.exists():
                        # Read with OpenCV
                        img = cv2.imread(str(p))
                        if img is not None:
                            # Convert BGR to RGB
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images[cam] = img
        except Exception as e:
            print(f"Error loading images for {sample_token}: {e}")
            
        return images

    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        sample_token = row.get("sample_token")
        
        # Load Images and Create Montage
        images = self._resolve_paths(sample_token)
        montage = self.montage_builder.create_montage(images)
        
        question = row.get("question", "")
        answer = row.get("answer", "")
        
        # Format for Qwen2.5-VL
        # We perform template construction here
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": montage},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        
        return {
            "id": sample_token,
            "messages": messages,
            "image": montage # Keep raw image accessible if needed
        }

def make_collate(processor):
    """
    Collate function to process inputs using Qwen processor.
    Processor handles image resizing and tokenization.
    """
    def collate_fn(batch):
        # Separate inputs
        texts = [b["messages"] for b in batch]
        # Qwen Processor takes list of messages.
        # But we need to handle training labels.
        # Qwen2.5-VL processor usually handles text tokenization.
        
        # For training, we usually format as:
        # User: <image> <text>
        # Assistant: <answer>
        
        # The processor from 'transformers' or 'qwen_vl_utils' usually handles this.
        # usage: processor(text=prompt, images=images, return_tensors="pt", padding=True)
        # But here we have full conversation.
        
        # We will use the processor's apply_chat_template if available? 
        # Or just manually extract.
        
        # Let's assume we use the standard Qwen2.5-VL processor flow
        # inputs = processor(text=[...], images=[...], padding=True, return_tensors="pt")
        
        start_texts = [b["messages"][0]["content"][1]["text"] for b in batch] # Question
        answers = [b["messages"][1]["content"][0]["text"] for b in batch] # Answer
        images = [b["image"] for b in batch] # PIL Images
        
        # Construct full prompt text?
        # Qwen2.5-VL uses specific tokens for vision.
        # <|vision_start|>...<|vision_end|> ? 
        # Actually Qwen2.5-VL uses <|image_pad|> usually handled by processor.
        
        # We will return the raw batch to be processed inside the training loop 
        # or process it here if processor is passed.
        
        if processor:
            # Prepare conversations
            # We need to construct conversation text
            
            # This part depends on the exact `transformers` Qwen2.5-VL implementation details.
            # Assuming `Qwen2_5_VLProcessor` supports `apply_chat_template` or we pass `text` + `images`.
            
            # The standard way for Qwen2-VL:
            # text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            # image_inputs = [img for img in images]
            # inputs = processor(text=text_inputs, images=image_inputs, padding=True, return_tensors="pt")
            
            # However, for training we need labels.
            # We usually use DataCollatorForSeq2Seq or similar.
            
            pass 
        
        # Return Raw list for flexibility in Main Loop
        return batch
        
    return collate_fn
