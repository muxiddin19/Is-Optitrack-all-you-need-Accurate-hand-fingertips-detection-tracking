import os
import cv2
import numpy as np
from PIL import Image

def process_dataset(input_dir, output_dir):
    # Find all RGB and depth files
    pairs = []
    
    for root, dirs, files in os.walk(input_dir):
        if 'rgb' in root:
            for file in files:
                if file.endswith('.png'):
                    rgb_path = os.path.join(root, file)
                    depth_path = rgb_path.replace('/rgb/', '/depth_converted/')
                    
                    if os.path.exists(depth_path):
                        pairs.append((rgb_path, depth_path))
    
    print(f"Found {len(pairs)} RGB-depth pairs")
    
    # Process each pair
    for i, (rgb_path, depth_path) in enumerate(pairs):
        # Determine split and create filename
        if '/train/' in rgb_path:
            split = 'train'
        else:
            split = 'test'
        
        filename = f"{i:06d}.png"
        
        # Create output directories
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/depth", exist_ok=True)
        
        # Fix RGB: resize to (480,640,3)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.resize(rgb, (640, 480))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(f"{output_dir}/{split}/images/{filename}")
        
        # Fix depth: resize to (480,640), convert to meters
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth.max() > 100:
            depth = depth / 1000.0  # mm to meters
        depth = np.clip(depth, 0.1, 10.0)
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
        depth_png = (depth * 1000).astype(np.uint16)
        cv2.imwrite(f"{output_dir}/{split}/depth/{filename}", depth_png)
        
        if i % 100 == 0:
            print(f"Processed {i} pairs...")
    
    print(f"Done! Processed {len(pairs)} pairs")

# Run it
process_dataset("datasets/custom_data/official_split", "datasets/custom_data_fixed")
EOF
