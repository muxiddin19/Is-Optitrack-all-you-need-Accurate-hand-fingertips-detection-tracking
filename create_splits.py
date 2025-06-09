import os

def create_split_files(data_dir):
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Process train split
    train_images_dir = f"{data_dir}/train/rgb"
    train_entries = []
    
    if os.path.exists(train_images_dir):
        for filename in sorted(os.listdir(train_images_dir)):
            if filename.endswith('.png'):
                base_name = filename.replace('.png', '')
                parts = base_name.split('_')
                aligned_part = parts[0]
                file_part = '_'.join(parts[1:])
                
                rgb_path = f"rgb/{aligned_part}{file_part}.png"
                depth_path = f"depth/{aligned_part}{file_part}.png"
                focal_length = "518.8579"
                
                entry = f"{rgb_path} {depth_path} {focal_length}"
                train_entries.append(entry)
    else:
        print(f"Train images directory not found: {train_images_dir}")
    
    # Process test split
    test_images_dir = f"{data_dir}/test/rgb"
    test_entries = []
    
    if os.path.exists(test_images_dir):
        for filename in sorted(os.listdir(test_images_dir)):
            if filename.endswith('.png'):
                base_name = filename.replace('.png', '')
                parts = base_name.split('_')
                aligned_part = parts[0]
                file_part = '_'.join(parts[1:])
                
                rgb_path = f"rgb/{aligned_part}{file_part}.png"
                depth_path = f"depth/{aligned_part}{file_part}.png"
                focal_length = "518.8579"
                
                entry = f"{rgb_path} {depth_path} {focal_length}"
                test_entries.append(entry)
    else:
        print(f"Test images directory not found: {test_images_dir}")
    
    # Write train.txt
    train_file = f"{data_dir}/train_fixed.txt"
    with open(train_file, 'w') as f:
        for entry in train_entries:
            f.write(f"{entry}\n")
    print(f"Created {train_file} with {len(train_entries)} entries")
    
    # Write test.txt
    test_file = f"{data_dir}/test_fixed.txt"
    with open(test_file, 'w') as f:
        for entry in test_entries:
            f.write(f"{entry}\n")
    print(f"Created {test_file} with {len(test_entries)} entries")
    
    # Show sample entries
    if train_entries:
        print("\nSample train.txt entries:")
        for entry in train_entries[:3]:
            print(f"  {entry}")
    
    if test_entries:
        print("\nSample test.txt entries:")
        for entry in test_entries[:3]:
            print(f"  {entry}")

# Run it
create_split_files("datasets/custom_data_fixed")