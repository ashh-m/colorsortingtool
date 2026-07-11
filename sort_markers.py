import json
import numpy as np
from PIL import Image
from scipy.spatial import distance

# ------------------------------------------------------------------------
# STEP 1: Extract RGB values from an image
# ------------------------------------------------------------------------
def pick_color_from_image(image_path, x, y):
    """
    Simulates color picking a specific coordinate (x, y) from your swatch image.
    In a full UI app, x and y would come from your mouse click.
    """
    with Image.open(image_path) as img:
        rgb_img = img.convert('RGB')
        return rgb_img.getpixel((x, y))

# Example dictionary simulating your 120 picked marker colors
marker_swatches = {
    "Marker_001": (255, 0, 50),
    "Marker_002": (0, 240, 255),
    "Marker_003": (120, 255, 0),
    "Marker_004": (230, 210, 0),
    # ... up to 120 colors
}

# ------------------------------------------------------------------------
# STEP 2: Convert RGB to CIELAB (Best for AI and sorting)
# ------------------------------------------------------------------------
def rgb_to_lab(rgb):
    """Converts an RGB tuple (0-255) to CIELAB coordinates."""
    # Normalize RGB to 0-1
    r, g, b = [v / 255.0 for v in rgb]
    
    # Convert to XYZ space
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92
    
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    
    # Normalize for D65 illuminant
    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883
    
    fx = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16/116)
    fy = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16/116)
    fz = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16/116)
    
    l = (116 * fy) - 16
    a = 500 * (fx - fy)
    b_lab = 200 * (fy - fz)
    return (l, a, b_lab)

# ------------------------------------------------------------------------
# STEP 3: Sort colors into a seamless chromatic rainbow (Nearest Neighbor)
# ------------------------------------------------------------------------
def sort_rainbow(swatches):
    """Sorts swatches seamlessly by finding the closest perceptual color."""
    names = list(swatches.keys())
    rgbs = list(swatches.values())
    labs = [rgb_to_lab(rgb) for rgb in rgbs]
    
    # Start with the absolute brightest/most vivid red-like color in your dataset
    current_index = np.argmax([lab[1] for lab in labs]) # Max 'a' value leans red/magenta
    
    unvisited = set(range(len(labs)))
    sorted_indices = [current_index]
    unvisited.remove(current_index)
    
    while unvisited:
        last_index = sorted_indices[-1]
        # Find the unvisited color with the smallest perceptual distance (Delta E)
        closest_index = min(unvisited, key=lambda i: distance.euclidean(labs[last_index], labs[i]))
        sorted_indices.append(closest_index)
        unvisited.remove(closest_index)
    
    # Rebuild the dictionary in the seamless sorted order
    return {names[i]: rgbs[i] for i in sorted_indices}

# ------------------------------------------------------------------------
# STEP 4: Run and Save to AI-friendly JSON format
# ------------------------------------------------------------------------
sorted_markers = sort_rainbow(marker_swatches)

with open("sorted_markers.json", "w") as f:
    json.dump(sorted_markers, f, indent=4)

print("Colors successfully sorted into a rainbow and saved to JSON!")
