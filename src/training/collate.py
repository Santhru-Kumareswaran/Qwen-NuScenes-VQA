
import torch
from typing import List, Dict, Any
from qwen_vl_utils import process_vision_info

class QwenDataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Batch: list of {id, messages, images}
        
        # 1. Prepare texts using apply_chat_template
        texts = [
            self.processor.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            for item in batch
        ]
        
        # 2. Extract images via process_vision_info
        # This handles the extraction of images relative to the messages
        image_inputs = []
        for item in batch:
            input_images, _ = process_vision_info(item["messages"])
            image_inputs.append(input_images)
            
        # image_inputs is now a list of lists of images (e.g. [[img1..6], [img1..6], ...])
        
        # 3. Process with Qwen VL Processor
        # Limit image resolution to reduce VRAM (default max is ~1.8M pixels = OOM)
        # 256*28*28 = 200,704 pixels max (~448x448) - fits in 16GB VRAM with 6 images
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            min_pixels=28*28*4,      # ~3136 pixels min (56x56)
            max_pixels=28*28*256     # ~200K pixels max (~448x448)
        )
        
        # 4. Create Labels (Masking)
        # Simple instruction masking: Mask everything except Assisant response.
        # However, for simplicity and robustness in this custom loop, we just return inputs.
        # Standard SFT often trains on full sequence or masks user.
        # We'll stick to a simple strategy: clone input_ids as labels and mask padding.
        
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        
        # Mask padding
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
        inputs["labels"] = labels
        
        return inputs
