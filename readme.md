# Qwen2.5-VL Autonomous Driving VQA

This package implements a "Stitched-Montage" approach for multi-view autonomous driving VQA using **Qwen2.5-VL-3B-Instruct**. It is designed to process 6 surround-view cameras (NuScenes format) by stitching them into a unified 2x3 grid with visual prompting.

## Features
- **Stitched-Montage**: Combines 6 surround-view cameras (Front, Front-Left, Front-Right, Back, Back-Left, Back-Right) into a single 2x3 image grid.
- **Visual Prompting**: Embeds camera labels and color-coded borders directly into the pixel space to assist the model in spatial reasoning.
- **Parameter Efficient**: Uses **QLoRA** (4-bit quantization) to fine-tune on consumer or workstation hardware.
- **NuScenes Integration**: Automatically resolves camera view paths using NuScenes sample tokens.

## Project Structure
```text
.
├── scripts/
│   └── train.py           # Main training entry point (Configuration inside)
├── src/
│   ├── configs/
│   │   └── constants.py   # View colors, order, and visual prompt settings
│   ├── data/
│   │   ├── dataset.py     # QwenNuDataset implementation
│   │   ├── montage.py     # Image stitching and border drawing logic
│   │   └── splitting.py   # Scene-aware train/val splitting
│   ├── model/
│   │   └── setup.py       # Model initialization with LoRA & 4-bit quantization
│   ├── training/
│   │   ├── collate.py     # Data collator for batching
│   │   └── trainer.py     # Custom training loop
│   └── utils/
│       ├── checkpoints.py # Save/Load logic
│       ├── metrics.py     # Evaluation metrics
│       └── plotting.py    # Visualization tools
├── readme.md
└── requirements.txt
```

## Setup & Configuration

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Paths**:
   The project uses a configuration dictionary inside `scripts/train.py`. You **must** edit this file to point to your data.
   
   Open `scripts/train.py` and find `get_training_config()`:
   ```python
   config = {
       # Path to folder containing your VQA annotation JSON files
       "data_root": "/path/to/your/annotations", 
       
       # Path to the root of your NuScenes dataset (e.g., contains maps, samples, sweeps)
       "nusc_root": "/path/to/nuscenes/data",
       
       "output_dir": "./checkpoints",
       
       # Hyperparameters
       "batch_size": 1, 
       "epochs": 5,
       # ...
   }
   ```

## Training

Once configured, start the training process:

```bash
python scripts/train.py
```

The script will:
1. Scan `data_root` for JSON files.
2. Initialize the Qwen2.5-VL model with 4-bit quantization.
3. Perform a scene-aware split (ensuring samples from the same drive/scene don't leak into validation).
4. Save checkpoints to `output_dir`.

## Architecture Details

- **Input Resolution**: The model receives a single stitched image of resolution **1008x672** (composed of 2x3 grid of 336x336 sub-images).
- **Visual Cues**: 
  - Each camera view is wrapped in a distinct colored border (defined in `src/configs/constants.py`).
  - Text labels (e.g., "FRONT LEFT") are burned into the image.
- **Model**: Qwen2.5-VL-3B-Instruct with LoRA adapters targeting effective parameter efficient fine-tuning.
