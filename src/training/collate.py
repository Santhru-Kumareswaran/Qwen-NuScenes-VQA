
import torch
from typing import List, Dict, Any

class QwenDataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Batch: list of {id, messages, image}
        
        # 1. Prepare texts and images
        texts = [item["messages"] for item in batch]
        images = [item["image"] for item in batch]
        
        # 2. Apply chat template
        # We need to construct the prompt string for training.
        # User: <image> Question
        # Assistant: Answer
        
        # Processor.apply_chat_template handles this if we format correctly.
        # But for training, we need to mask user tokens in loss.
        # This is non-trivial with standard processor calls.
        
        # Simplified approach:
        # Use processor to process (text + image) -> input_ids
        # Then create labels by masking.
        
        text_inputs = [
            self.processor.apply_chat_template(
                t, tokenize=False, add_generation_prompt=False
            )
            for t in texts
        ]
        
        inputs = self.processor(
            text=text_inputs,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        
        # Create labels
        # By default inputs['input_ids'] includes everything.
        # We need to identify where "Assistant" response starts.
        # Qwen2.5-VL uses <|im_start|>assistant ...
        
        # We can implement a naive masking strategy or just train on everything (User+Assistant) 
        # but that's suboptimal. 
        # Ideally we use the standard masking.
        
        # For now, let's just return input_ids as labels (calculating loss on prompt too),
        # or -100 masking is better.
        
        # Let's try to do simple masking if possible, else default to all-text training 
        # (common in simple SFT scripts, though not ideal).
        
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        
        # Masking padded tokens
        # processor padding uses pad_token_id
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
        # TODO: Advanced masking (User Turn masking)
        # This requires finding the boundaries of turns in the tokenized sequence.
        
        inputs["labels"] = labels
        
        return inputs
