#!/usr/bin/env python3
"""
Visualize one sample from the NuScenes VQA dataset showing all 6 camera views.
"""
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
from PIL import Image
from nuscenes import NuScenes

from configs.constants import VIEW_ORDER, VIEW_COLORS, VIEW_LABELS, GRID_LAYOUT
from data.dataset import QwenNuDataset, load_json_any
import glob

def create_montage_with_labels(images_dict, save_path):
    """Create a 2x3 montage with labeled camera views."""
    target_size = (400, 300)  # w, h per image
    rows, cols = GRID_LAYOUT
    
    canvas = np.zeros((target_size[1] * rows, target_size[0] * cols, 3), dtype=np.uint8)
    
    for idx, view_name in enumerate(VIEW_ORDER):
        r = idx // cols
        c = idx % cols
        
        y_start = r * target_size[1]
        x_start = c * target_size[0]
        
        if view_name in images_dict and images_dict[view_name] is not None:
            img = images_dict[view_name]
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            # Resize
            img_resized = cv2.resize(img, target_size)
            
            # Add border and label
            color = VIEW_COLORS.get(view_name, (255, 255, 255))
            label = VIEW_LABELS.get(view_name, view_name)
            
            # Draw border
            cv2.rectangle(img_resized, (0, 0), (target_size[0]-1, target_size[1]-1), color, 5)
            
            # Draw label with background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img_resized, (5, 5), (15 + text_w, 15 + text_h), (0, 0, 0), -1)
            cv2.putText(img_resized, label, (10, 10 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            canvas[y_start:y_start+target_size[1], x_start:x_start+target_size[0]] = img_resized
        else:
            # Fill with black and label as "MISSING"
            cv2.putText(canvas, f"{view_name}\nMISSING", 
                       (x_start + 50, y_start + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Save
    cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"Saved montage to: {save_path}")
    return canvas

def main():
    # Config - using user's local paths
    nusc_root = "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train"
    data_root = "/home/santhru/FYP38_First Experiment/NuScenesVQA-/scripts"
    
    print("Initializing NuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_root, verbose=False)
    
    # Find JSON files
    json_paths = glob.glob(str(Path(data_root) / "**" / "*.json"), recursive=True)
    print(f"Found {len(json_paths)} JSON files")
    
    if not json_paths:
        print("No JSON files found!")
        return
    
    # Load dataset (just first 5 samples for testing)
    dataset = QwenNuDataset(
        json_paths=json_paths,
        nusc=nusc,
        max_samples=5
    )
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No samples in dataset!")
        return
    
    # Get first sample
    sample = dataset[0]
    print(f"\nSample ID: {sample['id']}")
    print(f"Number of images: {len(sample['images'])}")
    
    # Get question and answer
    messages = sample['messages']
    question = ""
    answer = ""
    for item in messages[0]['content']:
        if item['type'] == 'text':
            question = item['text']
    for item in messages[1]['content']:
        if item['type'] == 'text':
            answer = item['text']
    
    print(f"Question: {question[:100]}...")
    print(f"Answer: {answer[:100]}...")
    
    # Create montage from the images
    # The dataset returns images in VIEW_ORDER
    images_dict = {}
    for i, view_name in enumerate(VIEW_ORDER):
        if i < len(sample['images']):
            images_dict[view_name] = sample['images'][i]
    
    save_path = Path(data_root) / "sample_montage.png"
    create_montage_with_labels(images_dict, save_path)
    
    print(f"\n✅ Montage saved to: {save_path}")
    print("Open this file to verify the 6 camera images are correct.")

if __name__ == "__main__":
    main()
