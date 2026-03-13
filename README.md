# Qwen-NuScenes-VQA

**From Scene Understanding to Safe Motion: A Modular Framework for Language-Driven Autonomous Driving**

A modular autonomous driving framework integrating distilled vision-language reasoning, action-token planning, and multi-modal trajectory execution. Fine-tunes Qwen3-VL-2B on nuScenes for near real-time inference on a single consumer GPU.

📄 **Paper**: *From Scene Understanding to Safe Motion: A Modular Framework for Language-Driven Autonomous Driving*
📦 **Dataset**: [Scene2Motion-50K on HuggingFace](https://huggingface.co/datasets/mahexh/Scene2Motion-50K)

---

## Architecture

The framework separates **high-level semantic reasoning** from **low-level trajectory control**:

1. **Vision-Language Planner** — Qwen3-VL-2B distilled from Gemini Robotics ER 1.5, produces discrete action tokens from multi-view camera inputs
2. **Action-Token Interface** — Discrete maneuver vocabulary (straight, left, right, U-turn, speed variants) bridging semantic planning and numerical control
3. **Trajectory Refinement Network** — LSTM + cross-attention module fusing LiDAR BEV features, ego waypoint history, and reference paths

---

## Project Structure

```
Qwen-NuScenes-VQA/
├── src/                            # VLM planner core modules
│   ├── configs/constants.py        # View order, prompts, constants
│   ├── data/
│   │   ├── dataset.py              # QwenNuDataset
│   │   ├── montage.py              # MontageBuilder (6-cam surround-view stitch)
│   │   └── splitting.py            # Train/val split utilities
│   ├── model/setup.py              # Model loading with QLoRA (4-bit)
│   ├── training/
│   │   ├── trainer.py              # QwenTrainer
│   │   └── collate.py              # Collate function
│   └── utils/
│       ├── checkpoints.py          # Save/load checkpoints
│       ├── metrics.py              # BLEU, ROUGE-L, METEOR, CIDEr, SPICE
│       └── plotting.py             # Loss plots & inference visualization
│
├── scripts/                        # Training, inference & visualization
│   ├── train.py                    # Main VLM training script (Stage 1 & 2)
│   ├── inference_action.py         # Action-token inference
│   ├── evaluate_best_ckpt.py       # Evaluate best checkpoint
│   ├── benchmark_inference.py      # Inference speed benchmarking
│   ├── generate_vqa_dataset.py     # Generate VQA dataset JSON
│   ├── generate_bev_frames.py      # Generate BEV frames from LiDAR
│   ├── generate_goal_tokens_refined.py  # Goal-token generation
│   ├── generate_organized_bev.py   # Organized BEV generation
│   └── visualize_*.py              # Various visualization scripts
│
├── action_token_clustering/        # Action-token vocabulary construction
│   ├── main.py                     # Pipeline: classify → cluster → label
│   └── src/
│       ├── config.py               # Paths, cluster counts
│       ├── preprocessing.py        # Feature extraction from ego waypoints
│       ├── classification.py       # Rule-based maneuver classification
│       ├── clustering.py           # K-Medoids sub-clustering
│       ├── labeling.py             # Action-token naming from centroids
│       ├── visualization.py        # Cluster visualization
│       ├── visualize_global_scatter.py
│       └── visualize_templates.py  # Reference trajectory templates
│
├── trajectory_refinement/          # Trajectory execution module
│   ├── single_frame/               # Single LiDAR frame baseline
│   │   ├── config.json
│   │   ├── trajectory_model.py     # LSTM + cross-attention model
│   │   ├── trajectory_dataset.py   # Dataset with LiDAR BEV features
│   │   ├── train_trajectory_refinement.py
│   │   ├── inference_trajectory_refinement.py
│   │   └── visualization.py
│   └── temporal_bev/               # Temporal BEV (5 past LiDAR keyframes)
│       ├── config.json
│       ├── trajectory_model.py     # + Temporal BEV Fusion module
│       ├── trajectory_dataset.py   # + Past sweep loading
│       ├── train_trajectory_refinement.py
│       ├── inference_trajectory_refinement.py
│       └── visualization.py
│
├── action_token_templates.json     # Canonical reference trajectories
├── requirements.txt
└── README.md
```

---

## Key Results (nuScenes)

| Model | Action Score | ADE ↓ | FDE ↓ | nADE ↓ | R² ↑ | Temporal Frames |
|-------|-------------|-------|-------|--------|------|-----------------|
| Single Frame | 75.3% | 2.3116 | 4.9857 | 0.2185 | 0.5875 | 1 |
| Temporal BEV | 75.3% | 2.1021 | 4.4296 | 0.1521 | 0.6469 | 6 |

- **ADE reduction**: 9.1%
- **FDE reduction**: 11.2%
- **nADE reduction**: 30.4%

---

## Requirements

- Python 3.11+
- CUDA 12.0+ (for RTX 40/50 series)
- 16GB+ VRAM recommended

```bash
conda create -n FPY38 python=3.11 -y
conda activate FPY38
pip install -r requirements.txt
```

---

## Training

### Stage 1: Scene-Level Semantic Pretraining
Fine-tune on driving scene VQA (scene summaries + entity lists):
```bash
cd scripts
python train.py  # Configure data_root, nusc_root in script
```

### Stage 2: Action-Token Planning
Further fine-tune to predict discrete action tokens from multi-view observations + goal tokens.

### Trajectory Refinement
```bash
cd trajectory_refinement/single_frame   # or temporal_bev
python train_trajectory_refinement.py
```

### Action-Token Vocabulary Construction
```bash
cd action_token_clustering
python main.py
```

---

## Hardware Tested

- **GPU**: NVIDIA RTX 5080 Laptop (16GB VRAM)
- **CPU**: Intel Core Ultra 9 275HX
- **RAM**: 32GB

---

## Citation

If you use this code, please cite:
```bibtex
@inproceedings{sajeev2025scene2motion,
  title={From Scene Understanding to Safe Motion: A Modular Framework for Language-Driven Autonomous Driving},
  author={Sajeev, Advaith and Raj, B. Mahesh and Kuttalam R., Kishore and K., Santhru and K.R., Bindu},
  year={2025}
}
```

## License

MIT License
