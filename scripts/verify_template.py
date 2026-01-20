from transformers import AutoProcessor
import torch

try:
    model_id = "Qwen/Qwen3-VL-2B-THinking"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "http://placeholder.com/1.jpg"},
                {"type": "image", "image": "http://placeholder.com/2.jpg"},
                {"type": "text", "text": "What is happening?"}
            ]
        }
    ]

    # Test apply_chat_template
    # We want to see if <think> tags appear or if we need to add them
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("="*40)
    print("GENERATED PROMPT:")
    print("="*40)
    print(prompt)
    print("="*40)

except Exception as e:
    print(f"Error: {e}")
