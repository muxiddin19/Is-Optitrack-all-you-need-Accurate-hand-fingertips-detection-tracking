import os
import shutil
import random
import re

# Configuration
DATASET_PATH = "./datasets/custom_data/rgb_depth/"
OUTPUT_PATH = "./datasets/custom_data/official_split/"
TRAIN_RATIO = 0.8  # 80% Train, 20% Test

# Create output directories
os.makedirs(OUTPUT_PATH, exist_ok=True)

def extract_matching_key(filename):
    """Extract matching key from filename for pairing files"""
    # Method 1: Direct filename match (for simple numbered files like 000001.png)
    base_name = os.path.splitext(filename)[0]  # Remove .png extension
    
    # Check if it's a simple number (6 digits)
    if re.match(r'^\d{6}$', base_name):
        return ('direct_match', base_name)
    
    # Method 2: Extract timestamp pattern (for complex names)
    # Pattern: 6digits_anything_8digits_6digits (e.g., 000000_ir_20250526_160126)
    match1 = re.search(r'(\d{6})_.*_(\d{8}_\d{6})', filename)
    if match1:
        return ('timestamp_pattern', f"{match1.group(1)}_{match1.group(2)}")
    
    # Method 3: Extract just the 6-digit number at the beginning
    match2 = re.search(r'^(\d{6})', filename)
    if match2:
        return ('number_prefix', match2.group(1))
    
    # Method 4: Find any 6-digit sequence
    match3 = re.search(r'(\d{6})', filename)
    if match3:
        return ('any_number', match3.group(1))
    
    # Method 5: Use the full base filename as key
    return ('filename_match', base_name)

def is_rgb_file(filename, folder_path):
    """Check if file is an RGB file based on naming and location"""
    filename_lower = filename.lower()
    
    # Must be PNG
    if not filename_lower.endswith('.png'):
        return False
    
    # If in RGB folder, assume it's RGB
    if 'rgb' in folder_path.lower():
        return True
    
    # Check for RGB/IR indicators in filename
    rgb_indicators = ['_rgb_', '_ir_', 'rgb_', 'ir_']
    for indicator in rgb_indicators:
        if indicator in filename_lower:
            return True
    
    return True  # Default to True if in rgb folder

def is_depth_file(filename, folder_path):
    """Check if file is a depth file based on naming and location"""
    filename_lower = filename.lower()
    
    # Must be PNG
    if not filename_lower.endswith('.png'):
        return False
    
    # If in depth folder, assume it's depth
    if 'depth' in folder_path.lower():
        return True
    
    # Check for depth indicators in filename
    depth_indicators = ['_depth_', 'depth_', '_depth.', 'sync_depth']
    for indicator in depth_indicators:
        if indicator in filename_lower:
            return True
    
    return True  # Default to True if in depth folder

def analyze_folder_pairing(folder, rgb_folder, depth_folder):
    """Analyze and pair files in a specific folder"""
    folder_info = {
        'rgb_files': {},
        'depth_files': {},
        'camera_type': 'unknown',
        'pairing_method': 'unknown',
        'matched_pairs': []
    }
    
    print(f"\nAnalyzing folder: {folder}")
    
    # Get RGB files
    rgb_files = []
    if os.path.exists(rgb_folder):
        rgb_files = [f for f in os.listdir(rgb_folder) if f.endswith('.png')]
        print(f"  RGB folder: {len(rgb_files)} files")
        
        if rgb_files:
            sample_rgb = rgb_files[0]
            print(f"  Sample RGB: {sample_rgb}")
            
            # Determine camera type and pairing method
            if re.match(r'^\d{6}\.png$', sample_rgb):
                folder_info['camera_type'] = 'Simple numbered'
                folder_info['pairing_method'] = 'direct_filename'
                print(f"  → Simple numbered files (e.g., 000001.png)")
            elif '_ir_' in sample_rgb:
                folder_info['camera_type'] = 'D405'
                folder_info['pairing_method'] = 'timestamp_pattern'
                print(f"  → D405 camera (IR naming)")
            elif '_rgb_' in sample_rgb:
                folder_info['camera_type'] = 'D415'
                folder_info['pairing_method'] = 'timestamp_pattern'
                print(f"  → D415 camera (RGB naming)")
            else:
                folder_info['pairing_method'] = 'flexible'
                print(f"  → Unknown type, using flexible matching")
    
    # Get depth files
    depth_files = []
    if os.path.exists(depth_folder):
        depth_files = [f for f in os.listdir(depth_folder) if f.endswith('.png')]
        print(f"  Depth folder: {len(depth_files)} files")
        
        if depth_files:
            sample_depth = depth_files[0]
            print(f"  Sample Depth: {sample_depth}")
    
    # Try different pairing strategies
    matched_pairs = []
    
    # Strategy 1: Direct filename matching (for simple numbered files)
    if folder_info['pairing_method'] in ['direct_filename', 'flexible']:
        direct_matches = find_direct_matches(rgb_files, depth_files)
        if direct_matches:
            matched_pairs.extend(direct_matches)
            print(f"  ✓ Direct filename matches: {len(direct_matches)}")
    
    # Strategy 2: Pattern-based matching (for timestamped files)
    if folder_info['pairing_method'] in ['timestamp_pattern', 'flexible'] and not matched_pairs:
        pattern_matches = find_pattern_matches(rgb_files, depth_files)
        if pattern_matches:
            matched_pairs.extend(pattern_matches)
            print(f"  ✓ Pattern-based matches: {len(pattern_matches)}")
    
    # Strategy 3: Flexible matching (try all methods)
    if not matched_pairs:
        flexible_matches = find_flexible_matches(rgb_files, depth_files)
        if flexible_matches:
            matched_pairs.extend(flexible_matches)
            print(f"  ✓ Flexible matches: {len(flexible_matches)}")
    
    # Convert to full paths
    final_pairs = []
    for rgb_file, depth_file in matched_pairs:
        rgb_path = f"{folder}/rgb/{rgb_file}"
        depth_path = f"{folder}/depth_converted/{depth_file}"
        final_pairs.append((rgb_path, depth_path))
    
    folder_info['matched_pairs'] = final_pairs
    print(f"  ✓ Total matched pairs: {len(final_pairs)}")
    
    return folder_info

def find_direct_matches(rgb_files, depth_files):
    """Find matches by direct filename comparison"""
    matches = []
    rgb_basenames = {os.path.splitext(f)[0]: f for f in rgb_files}
    depth_basenames = {os.path.splitext(f)[0]: f for f in depth_files}
    
    for basename in rgb_basenames:
        if basename in depth_basenames:
            matches.append((rgb_basenames[basename], depth_basenames[basename]))
    
    return matches

def find_pattern_matches(rgb_files, depth_files):
    """Find matches by extracting patterns from filenames"""
    matches = []
    
    # Group RGB files by pattern
    rgb_patterns = {}
    for rgb_file in rgb_files:
        method, key = extract_matching_key(rgb_file)
        if key not in rgb_patterns:
            rgb_patterns[key] = []
        rgb_patterns[key].append(rgb_file)
    
    # Group depth files by pattern
    depth_patterns = {}
    for depth_file in depth_files:
        method, key = extract_matching_key(depth_file)
        if key not in depth_patterns:
            depth_patterns[key] = []
        depth_patterns[key].append(depth_file)
    
    # Find matches
    for key in rgb_patterns:
        if key in depth_patterns:
            # Take the first match for each pattern
            rgb_file = rgb_patterns[key][0]
            depth_file = depth_patterns[key][0]
            matches.append((rgb_file, depth_file))
    
    return matches

def find_flexible_matches(rgb_files, depth_files):
    """Try multiple matching strategies"""
    # Try direct match first
    matches = find_direct_matches(rgb_files, depth_files)
    if matches:
        return matches
    
    # Try pattern matching
    matches = find_pattern_matches(rgb_files, depth_files)
    if matches:
        return matches
    
    # Try number-based matching (extract any numbers)
    rgb_numbers = {}
    for rgb_file in rgb_files:
        numbers = re.findall(r'\d+', rgb_file)
        if numbers:
            key = numbers[0]  # Use first number as key
            rgb_numbers[key] = rgb_file
    
    depth_numbers = {}
    for depth_file in depth_files:
        numbers = re.findall(r'\d+', depth_file)
        if numbers:
            key = numbers[0]  # Use first number as key
            depth_numbers[key] = depth_file
    
    matches = []
    for key in rgb_numbers:
        if key in depth_numbers:
            matches.append((rgb_numbers[key], depth_numbers[key]))
    
    return matches

def collect_all_pairs():
    """Collect all matched pairs from all folders"""
    folders = os.listdir(DATASET_PATH)
    all_pairs = []
    
    print(f"Found {len(folders)} folders in dataset")
    print("=" * 70)
    
    for folder in folders:
        rgb_folder = os.path.join(DATASET_PATH, folder, "rgb")
        depth_folder = os.path.join(DATASET_PATH, folder, "depth_converted")
        
        if os.path.exists(rgb_folder) or os.path.exists(depth_folder):
            folder_info = analyze_folder_pairing(folder, rgb_folder, depth_folder)
            all_pairs.extend(folder_info['matched_pairs'])
        else:
            print(f"\nSkipping {folder}: no rgb or depth_converted folders")
    
    return all_pairs

def copy_with_structure(src_path, dst_path):
    """Copy file maintaining directory structure"""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)

def create_train_test_split(all_pairs, train_ratio=0.8):
    """Split data into train and test sets"""
    random.shuffle(all_pairs)
    
    train_split = int(len(all_pairs) * train_ratio)
    
    train_data = all_pairs[:train_split]
    test_data = all_pairs[train_split:]
    
    print(f"\n" + "=" * 70)
    print("DATA SPLIT SUMMARY")
    print("=" * 70)
    print(f"Total pairs: {len(all_pairs)}")
    print(f"Train samples: {len(train_data)} ({len(train_data)/len(all_pairs)*100:.1f}%)")
    print(f"Test samples: {len(test_data)} ({len(test_data)/len(all_pairs)*100:.1f}%)")
    
    return train_data, test_data

def process_split(data, split_name, output_path):
    """Process a data split - copy files and create txt file"""
    split_dir = os.path.join(output_path, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    txt_file = f"data_splits/{split_name}.txt"
    os.makedirs("data_splits", exist_ok=True)
    
    print(f"\nProcessing {split_name.upper()} split...")
    
    successful_copies = 0
    with open(txt_file, 'w') as f:
        for i, (rgb_path, depth_path) in enumerate(data):
            # Copy files maintaining structure
            src_rgb = os.path.join(DATASET_PATH, rgb_path)
            src_depth = os.path.join(DATASET_PATH, depth_path)
            
            dst_rgb = os.path.join(split_dir, rgb_path)
            dst_depth = os.path.join(split_dir, depth_path)
            
            # Verify source files exist
            if not os.path.exists(src_rgb):
                print(f"  ✗ Missing source RGB: {src_rgb}")
                continue
            if not os.path.exists(src_depth):
                print(f"  ✗ Missing source depth: {src_depth}")
                continue
            
            copy_with_structure(src_rgb, dst_rgb)
            copy_with_structure(src_depth, dst_depth)
            
            # Write to txt file (relative paths)
            f.write(f"{rgb_path} {depth_path} 518.8579\n")
            successful_copies += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(data)} files...")
    
    print(f"  ✓ Successfully processed: {successful_copies}/{len(data)} pairs")
    print(f"  ✓ Created {txt_file}")
    return txt_file

def main():
    print("ENHANCED UNIVERSAL CAMERA DATA SPLITTER")
    print("=" * 70)
    print("Handles: D415 (_rgb_), D405 (_ir_), Simple numbered (000001.png)")
    print("=" * 70)
    
    # Collect all matched pairs
    all_pairs = collect_all_pairs()
    
    if not all_pairs:
        print("\n❌ ERROR: No valid RGB-Depth pairs found!")
        print("Please check your data structure and naming conventions.")
        return
    
    # Create train/test split
    train_data, test_data = create_train_test_split(all_pairs, TRAIN_RATIO)
    
    # Process splits
    train_txt = process_split(train_data, "train", OUTPUT_PATH)
    test_txt = process_split(test_data, "test", OUTPUT_PATH)
    
    print("\n" + "=" * 70)
    print("✅ DATA SPLITTING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"📁 Output directory: {OUTPUT_PATH}")
    print(f"📄 Train file: {train_txt}")
    print(f"📄 Test file: {test_txt}")
    print(f"🎯 Ready for training!")
    
    # Show sample results
    print(f"\nSample from train.txt:")
    if os.path.exists(train_txt):
        with open(train_txt, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:
                    print(f"  {line.strip()}")

if __name__ == "__main__":
    # Set random seed for reproducible splits
    random.seed(42)
    main()