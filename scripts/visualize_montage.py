
import csv
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from nuscenes.nuscenes import NuScenes

# Import MontageBuilder and Constants
import sys
sys.path.append(os.getcwd()) # Ensure src is in path
from src.data.montage import MontageBuilder
from src.configs.constants import VIEW_ORDER

# ================= CONFIG =================
NUSCENES_ROOT = "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train"
VERSION = "v1.0-trainval"
CSV_PATH = "nuscenes_goal_tokens_refined_subset_2.csv"
OUTPUT_DIR = "camera_montages_refined_subset_2"
# ==========================================

def add_header(montage_img, row):
    """
    Add a header strip with Goal and Action info.
    """
    w, h = montage_img.size
    header_h = 60
    
    # Create new image with header
    new_img = Image.new('RGB', (w, h + header_h), (0, 0, 0))
    new_img.paste(montage_img, (0, header_h))
    
    draw = ImageDraw.Draw(new_img)
    
    # Text
    goal = row.get('goal_name', 'UNKNOWN')
    action = row.get('action_token', 'UNKNOWN')
    maneuver = row.get('maneuver_type', '-')
    
    text = f"GOAL: {goal}  |  ACTION: {action}  |  MANEUVER: {maneuver}"
    
    # Font (default if specific font not found)
    try:
        font = ImageFont.truetype("FreeMonoBold.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    x = (w - text_w) // 2
    y = (header_h - text_h) // 2
    
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return new_img

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading NuScenes...")
    nusc = NuScenes(version=VERSION, dataroot=NUSCENES_ROOT, verbose=False)
    
    # Initialize Montage Builder
    builder = MontageBuilder()
    
    print(f"Reading {CSV_PATH}...")
    scene_rows = {}
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        return

    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            st = row['scene_token']
            if st not in scene_rows:
                scene_rows[st] = []
            scene_rows[st].append(row)
            
    print(f"Generating Montages for {len(scene_rows)} scenes...")
    
    for st in tqdm(scene_rows.keys(), desc="Scenes"):
        scene_dir = os.path.join(OUTPUT_DIR, st)
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
            
        for i, row in enumerate(scene_rows[st]):
            save_path = os.path.join(scene_dir, f"frame_{i:03d}.jpg")
            
            # 1. Fetch Images
            sample_token = row['sample_token']
            try:
                sample = nusc.get('sample', sample_token)
            except:
                continue
                
            images_dict = {}
            for view in VIEW_ORDER:
                cam_token = sample['data'][view]
                cam_path = nusc.get_sample_data_path(cam_token)
                
                # Load with CV2 (BGR) then convert to RGB
                img_bgr = cv2.imread(cam_path)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    images_dict[view] = img_rgb
                else:
                    images_dict[view] = None
            
            # 2. Create Montage
            montage_pil = builder.create_montage(images_dict)
            
            # 3. Add Info Header
            final_img = add_header(montage_pil, row)
            
            # 4. Save
            final_img.save(save_path)

    print(f"Done. Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
