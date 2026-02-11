
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def setup_model_and_processor(
    model_id: str,
    lora_r: int = 16,           # Increased for QLoRA
    lora_alpha: int = 32,       # Standard 2x ratio
    lora_dropout: float = 0.1,  # Reduced dropout for QLoRA
    use_4bit: bool = True,      # QLoRA enabled by default
    use_flash_attn: bool = True, # Flash Attention 2 enabled by default
    device_map = "auto",
    skip_lora: bool = False     # Skip LoRA creation (for loading from checkpoint)
):
    """
    Load Qwen VL model with QLoRA support optimized for RTX 5080 (16GB VRAM).
    
    Key optimizations:
    - Uses AutoModelForVision2Seq for Qwen3-VL compatibility
    - bf16 compute dtype for better numerical stability
    - SDPA (Scaled Dot Product Attention) when flash_attn disabled
    - Double quantization for memory efficiency
    - 8-bit Adam optimizer compatible
    """
    
    # 1. QLoRA Quantization Config with bf16
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16  # bf16 instead of fp16
        )
        
    # 2. Load Processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # 3. Attention Implementation
    # Options: "flash_attention_2", "sdpa", "eager"
    if use_flash_attn:
        attn_impl = "flash_attention_2"
        print(f"[Model] Using Flash Attention 2")
    else:
        attn_impl = "sdpa"  # Scaled Dot Product Attention (PyTorch native, efficient)
        print(f"[Model] Using SDPA (Scaled Dot Product Attention)")
    
    # 4. Load Model using AutoModelForVision2Seq (works with Qwen2-VL and Qwen3-VL)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # bf16 instead of fp16
        attn_implementation=attn_impl
    )
    
    # Enable Gradient Checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # 4. Prepare for QLoRA
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        
    # 5. Apply LoRA (skip if loading from checkpoint)
    if skip_lora:
        print(f"[Model] Skipping LoRA creation (will load from checkpoint)")
        print(f"[Model] Loaded base model with {'4-bit quantization' if use_4bit else 'full precision'}")
        print(f"[Model] Attention: {attn_impl}")
        print(f"[Model] Dtype: bf16")
        return model, processor
    
    # Target modules for Qwen2-VL: LLM backbone layers
    # For QLoRA, higher rank (r=16+) works better than standard LoRA
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Qwen2-VL is treated as Causal LM with images
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",  # No bias for efficiency
    )
    
    model = get_peft_model(model, peft_config)
    
    # 6. Log configuration
    print(f"[Model] Loaded with {'QLoRA (4-bit)' if use_4bit else 'LoRA'}")
    print(f"[Model] Attention: {attn_impl}")
    print(f"[Model] Dtype: bf16")
    print(f"[Model] LoRA r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    model.print_trainable_parameters()
    
    return model, processor

