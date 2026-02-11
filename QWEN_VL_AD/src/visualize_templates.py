
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as cfg

TEMPLATE_FILE = os.path.join(cfg.output_dir, 'action_token_templates.json')
GLOBAL_PLOT = os.path.join(cfg.plots_dir, 'action_token_templates_viz.png')

def rotate(pts):
    """Rotate points by +90 degrees (x->y, y->x) so Forward is Up."""
    return np.stack([-pts[..., 1], pts[..., 0]], axis=-1)

def plot_set(templates, title, output_path, colors_map=None):
    """Helper to plot a set of templates with arrows and labels."""
    plt.figure(figsize=(10, 10))
    
    # Collect names to plot
    names = list(templates.keys())
    
    # If no color map provided, use a cycle
    import matplotlib.cm as cm
    
    for i, name in enumerate(names):
        points = templates[name]
        pts = np.array(points)
        pts_rot = rotate(pts)
        
        # Color logic
        color = 'black'
        if colors_map: # Use fixed prefix-based colors
             for key, c in colors_map.items():
                if name.startswith(key):
                    color = c
                    break
        else: # Use colormap for distinction within group
             color = cm.get_cmap('tab10')(i % 10)

        # Plot line
        plt.plot(pts_rot[:, 0], pts_rot[:, 1], color=color, linewidth=2, label=name)
        
        # Plot arrow
        plt.arrow(pts_rot[-2, 0], pts_rot[-2, 1], 
                  pts_rot[-1, 0] - pts_rot[-2, 0], 
                  pts_rot[-1, 1] - pts_rot[-2, 1], 
                  head_width=1.5, color=color)
        
        # Add label
        plt.text(pts_rot[-1, 0], pts_rot[-1, 1], name, fontsize=8, color=color)

    plt.title(title)
    plt.xlabel("Lateral (m)")
    plt.ylabel("Longitudinal (m) [Up]")
    plt.grid(True)
    plt.axis('equal')
    
    if len(names) < 15:
        plt.legend(loc='best')
    else:
        # Custom legend for global
        if colors_map:
             from matplotlib.lines import Line2D
             custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors_map.values()]
             plt.legend(custom_lines, colors_map.keys(), loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def main():
    print(f"Loading templates from {TEMPLATE_FILE}...")
    with open(TEMPLATE_FILE, 'r') as f:
        templates = json.load(f)
        
    # 1. Global Plot
    print("Generating Global Plot...")
    global_colors = {
        'LEFT': 'red',
        'RIGHT': 'blue',
        'STRAIGHT': 'green',
        'U_TURN_LEFT': 'purple',
        'U_TURN_RIGHT': 'magenta',
        'STATIONARY': 'gray',
        'CREEPING': 'orange'
    }
    plot_set(templates, "Action Token Templates (Global)", GLOBAL_PLOT, global_colors)
    
    # 2. Group Plots
    print("Generating Group Plots...")
    groups = ['LEFT', 'RIGHT', 'STRAIGHT', 'U_TURN_LEFT', 'U_TURN_RIGHT', 'STATIONARY', 'CREEPING']
    
    for group in groups:
        # Filter templates
        subset = {k: v for k, v in templates.items() if k.startswith(group)}
        if not subset: continue
        
        out_path = os.path.join(cfg.plots_dir, 'templates', f'templates_{group}.png')
        plot_set(subset, f"Action Templates: {group}", out_path, None) # Use distinct colors for subset
        
    # 3. Individual Plots - Disabled per user request
    # print("Generating Individual Plots...")
    # ... code removed ...

if __name__ == "__main__":
    main()
