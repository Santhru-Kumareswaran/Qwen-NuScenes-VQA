
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def setup_model_and_processor(
    model_id: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_4bit: bool = False,
    use_flash_attn: bool = True,
    device_map = "auto"
):
    """
    Load Qwen2.5-VL model with LoRA support.
    """
    
    # 1. Quantization Config
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
    # 2. Load Processor
    # Qwen2.5-VL processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # 3. Load Model
    # Note: attn_implementation="flash_attention_2" requires appropriate hardware/deps
    attn_impl = "flash_attention_2" if use_flash_attn else "eager"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation=attn_impl
    )
    
    # Enable Gradient Checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # 4. Prepare for LoRA
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
        
    # 5. Apply LoRA
    # Target modules for Qwen2-VL: usually ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # We apply to vision encoder too? Usually yes for best VQA performance.
    # Qwen2-VL has 'visual' module.
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # Add vision targets if needed, e.g. "patch_embed", "c_fc", etc. 
    # For now, let's stick to LLM backbone or auto-detect.
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # Qwen2-VL is treated as Causal LM with images
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
        # modules_to_save... e.g. "embed_tokens" if we had special tokens, but we use existing visual tokens
    )
    
    model = get_peft_model(model, peft_config)
    
    # 6. Gradient Checkpointing (Ensure enabled if passed in config, but typically we handle outside)
    # The caller usually handles this via model.gradient_checkpointing_enable()
    # But just in case
    
    model.print_trainable_parameters()
    
    return model, processor

