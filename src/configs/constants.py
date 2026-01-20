
# Qwen2.5-VL-3B Constants

# Camera definitions
VIEW_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"
]

# Grid Layout (Rows, Cols)
GRID_LAYOUT = (2, 3)

# View specific visual prompting configuration
# Use distinct colors for each view for clear disambiguation
VIEW_COLORS = {
    "CAM_FRONT_LEFT": (255, 0, 0),     # Red
    "CAM_FRONT": (0, 255, 0),          # Green
    "CAM_FRONT_RIGHT": (0, 0, 255),    # Blue
    "CAM_BACK_LEFT": (255, 255, 0),    # Yellow
    "CAM_BACK": (255, 0, 255),         # Magenta
    "CAM_BACK_RIGHT": (0, 255, 255)    # Cyan
}

VIEW_LABELS = {
    "CAM_FRONT_LEFT": "FRONT LEFT",
    "CAM_FRONT": "FRONT",
    "CAM_FRONT_RIGHT": "FRONT RIGHT",
    "CAM_BACK_LEFT": "BACK LEFT",
    "CAM_BACK": "BACK",
    "CAM_BACK_RIGHT": "BACK RIGHT"
}

# Image parameters
# Qwen2.5-VL handles dynamic resolutions, but for batching we might want some consistency
# or we let the processor handle it. 
# Stitched resolution target (per sub-image)
SUB_IMAGE_SIZE = (336, 336) # Approx standard patch size multiple
BORDER_THICKNESS = 5
FONT_SCALE = 1.0
FONT_THICKNESS = 2

# Model
MODEL_ID = "Qwen/Qwen3-VL-2B-THinking"
MAX_LENGTH = 2048
