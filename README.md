# Qwen-NuScenes-VQA

Fine-tuning Qwen3-VL-2B for Visual Question Answering on NuScenes autonomous driving dataset.

## Features

- **Multi-camera Montage**: Stitches 6 camera views (Front-Left, Front, Front-Right, Back-Left, Back, Back-Right) into a single annotated image
- **QLoRA Training**: 4-bit quantized LoRA for memory-efficient fine-tuning on consumer GPUs (tested on RTX 5080 16GB)
- **Comprehensive Metrics**: BLEU-4, ROUGE-L, METEOR, CIDEr, SPICE evaluation
- **Resumable Training**: Checkpoint saving with automatic resume support
- **Visualization**: Inference visualization with side-by-side comparison

## Requirements

- Python 3.11+
- CUDA 12.0+ (for RTX 40/50 series)
- 16GB+ VRAM recommended

## Installation

```bash
# Create conda environment
conda create -n FPY38 python=3.11 -y
conda activate FPY38

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
NuScenesVQA-/
├── scripts/
│   ├── train.py              # Main training script
│   └── inference_visualizer.py  # Inference with visualization
├── src/
│   ├── configs/
│   │   └── constants.py      # View order, prompts
│   ├── data/
│   │   ├── dataset.py        # QwenNuDataset
│   │   └── montage.py        # MontageBuilder (6-cam stitch)
│   ├── model/
│   │   └── setup.py          # Model loading with QLoRA
│   ├── training/
│   │   ├── trainer.py        # QwenTrainer
│   │   └── collate.py        # Collate function
│   └── utils/
│       ├── checkpoints.py    # Save/load checkpoints
│       └── plotting.py       # Loss plots
└── requirements.txt
```

## Training

1. **Prepare Data**: Generate `vision_finetuning_dataset.json` with your VQA samples

2. **Configure Training**: Edit `scripts/train.py`:
   - `data_root`: Path to JSON files
   - `nusc_root`: Path to NuScenes dataset
   - `max_samples`: Set to `None` for full dataset

3. **Run Training**:
```bash
cd scripts
python train.py
```

4. **Resume Training**: Set `"resume": True` in config and run again

## Inference

```bash
python inference_visualizer.py \
  --checkpoint "checkpoints/run_XXXXXX/checkpoint-best" \
  --output_dir "results" \
  --num_samples 10
```

## Configuration

Key parameters in `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 50 | Number of training epochs |
| `batch_size` | 2 | Batch size per GPU |
| `grad_accum` | 8 | Gradient accumulation (effective batch=16) |
| `warmup_steps` | 3000 | LR warmup steps |
| `save_steps` | 500 | Checkpoint frequency |
| `val_every_steps` | 1400 | Validation frequency (~2x per epoch) |
| `debug` | False | Verbose logging mode |

## Hardware Tested

- **GPU**: NVIDIA RTX 5080 Laptop (16GB VRAM)
- **CPU**: Intel Core Ultra 9 275HX
- **RAM**: 32GB

## License

MIT License
