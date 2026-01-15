
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple
from configs.constants import (
    VIEW_ORDER, GRID_LAYOUT, VIEW_COLORS, VIEW_LABELS, 
    SUB_IMAGE_SIZE, BORDER_THICKNESS, FONT_SCALE, FONT_THICKNESS
)

class MontageBuilder:
    """
    Handles the creation of the Stitched-Montage input with Visual Prompting.
    """
    
    def __init__(self, sub_image_size=SUB_IMAGE_SIZE):
        self.w, self.h = sub_image_size
        
        # Calculate total grid size
        self.rows, self.cols = GRID_LAYOUT
        self.total_w = self.w * self.cols
        self.total_h = self.h * self.rows
        
    def _add_visual_prompts(self, img_np: np.ndarray, view_name: str) -> np.ndarray:
        """
        Add color border and text label to an image.
        """
        color = VIEW_COLORS.get(view_name, (255, 255, 255))
        label = VIEW_LABELS.get(view_name, view_name)
        
        # 1. Add Border
        # CV2 uses BGR, but we defined RGB in constants? Let's assume RGB for PIL consistency if we used PIL.
        # But here img_np is likely coming from cv2.imread which is BGR. 
        # Let's verify input format. Usually we assume we get RGB or we ensure it.
        # We will work with RGB numpy arrays for consistency.
        
        # Add border (inset)
        h, w = img_np.shape[:2]
        t = BORDER_THICKNESS
        
        # Draw border
        # cv2.rectangle(img_np, (0,0), (w-1, h-1), color, thickness=t) 
        # But we want 'embedded directly into pixel space'. 
        
        # Ensure consistent color handling (draw in BGR then convert back to RGB)
        img_with_border = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        color_bgr = (color[2], color[1], color[0])
        cv2.rectangle(img_with_border, (0, 0), (w, h), color_bgr, t*2) # *2 because it strokes center
        
        # 2. Add Text Label
        # Use a contrasting background for text or just the text?
        # "Explicit text markers... embedded"
        
        # Create a text box background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        
        # Position: Top-Left corner usually
        text_x, text_y = 10 + t, 10 + t + text_h
        
        # Draw background rect for text for readability
        cv2.rectangle(img_with_border, (text_x - 5, text_y - text_h - 5), (text_x + text_w + 5, text_y + 5), (0,0,0), -1)
        
        # Draw text
        cv2.putText(img_with_border, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color_bgr, FONT_THICKNESS)
        
        return cv2.cvtColor(img_with_border, cv2.COLOR_BGR2RGB)

    def create_montage(self, images_dict: Dict[str, np.ndarray]) -> Image.Image:
        """
        Args:
            images_dict: Dictionary mapping view names (e.g. CAM_FRONT) to numpy arrays (H, W, 3) RGB.
        Returns:
            PIL Image of the stitched montage
        """
        # Canvas
        canvas = np.zeros((self.total_h, self.total_w, 3), dtype=np.uint8)
        
        for idx, view_name in enumerate(VIEW_ORDER):
            if view_name not in images_dict or images_dict[view_name] is None:
                # Fill with black or noise if missing
                img_chunk = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            else:
                img_org = images_dict[view_name]
                # Resize if necessary
                if img_org.shape[0] != self.h or img_org.shape[1] != self.w:
                    img_chunk = cv2.resize(img_org, (self.w, self.h))
                else:
                    img_chunk = img_org.copy()
                
                # Apply Visual Prompting
                img_chunk = self._add_visual_prompts(img_chunk, view_name)
            
            # Place in grid
            # Row-major order
            r = idx // self.cols
            c = idx % self.cols
            
            y_start = r * self.h
            x_start = c * self.w
            
            canvas[y_start:y_start+self.h, x_start:x_start+self.w] = img_chunk
            
        return Image.fromarray(canvas)

