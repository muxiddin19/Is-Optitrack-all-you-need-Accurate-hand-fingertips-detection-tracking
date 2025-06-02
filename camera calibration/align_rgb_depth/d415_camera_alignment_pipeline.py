import os
import cv2
import numpy as np
import glob
from pathlib import Path
import shutil

class SimpleD415ToNeWCRFs:
    def __init__(self):
        """Simple converter for D415 data to NeWCRFs training format"""
        print("🚀 Simple D415 to NeWCRFs Converter")
        print("=" * 40)
        
        # Your exact data paths
        self.base_path = r"E:\vscode\dpt_d\extracted_frames_d\415\user01\action00"
        self.color_path = os.path.join(self.base_path, "color")
        self.depth_path = os.path.join(self.base_path, "depth")
        
        # Output path for NeWCRFs
        self.output_path = r"E:\vscode\newcrfs_training_data"
        
        print(f"📁 Color folder: {self.color_path}")
        print(f"📁 Depth folder: {self.depth_path}")
        print(f"📁 Output folder: {self.output_path}")
        
    def check_data(self):
        """Check what data we have"""
        print("\n🔍 Checking your data...")
        
        # Check if folders exist
        if not os.path.exists(self.color_path):
            print(f"❌ Color folder not found: {self.color_path}")
            return False
            
        if not os.path.exists(self.depth_path):
            print(f"❌ Depth folder not found: {self.depth_path}")
            return False
        
        # Find all color files
        color_files = glob.glob(os.path.join(self.color_path, "frame_*"))
        depth_files = glob.glob(os.path.join(self.depth_path, "frame_*"))
        
        print(f"✅ Found {len(color_files)} color frames")
        print(f"✅ Found {len(depth_files)} depth frames")
        
        if len(color_files) == 0 or len(depth_files) == 0:
            print("❌ No frames found!")
            return False
        
        # Show first few examples
        print("\n📋 Sample files:")
        for i, color_file in enumerate(sorted(color_files)[:3]):
            print(f"   Color: {Path(color_file).name}")
            
        for i, depth_file in enumerate(sorted(depth_files)[:3]):
            print(f"   Depth: {Path(depth_file).name}")
            
        return True
    
    def find_matching_pairs(self):
        """Find matching color-depth pairs"""
        print("\n🔗 Finding matching color-depth pairs...")
        
        color_files = glob.glob(os.path.join(self.color_path, "frame_*"))
        depth_files = glob.glob(os.path.join(self.depth_path, "frame_*"))
        
        color_dict = {}
        depth_dict = {}
        
        # Extract frame numbers and timestamps
        for color_file in color_files:
            filename = Path(color_file).name
            # Extract frame number from "frame_000002_time_1747879995.114"
            parts = filename.split('_')
            if len(parts) >= 4:
                frame_num = parts[1]  # "000002"
                timestamp = parts[3]  # "1747879995.114"
                color_dict[frame_num] = color_file
        
        for depth_file in depth_files:
            filename = Path(depth_file).name
            parts = filename.split('_')
            if len(parts) >= 4:
                frame_num = parts[1]  # "000002"
                timestamp = parts[3]  # "1747879995.114"
                depth_dict[frame_num] = depth_file
        
        # Find matching pairs
        matched_pairs = []
        for frame_num in color_dict:
            if frame_num in depth_dict:
                matched_pairs.append((color_dict[frame_num], depth_dict[frame_num]))
        
        print(f"✅ Found {len(matched_pairs)} matching pairs")
        return sorted(matched_pairs)
    
    def process_frame_pair(self, color_file, depth_file, output_index):
        """Process a single color-depth pair for NeWCRFs"""
        try:
            # Load images
            color_img = cv2.imread(color_file)
            depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            
            if color_img is None:
                print(f"❌ Could not load color: {color_file}")
                return False
                
            if depth_img is None:
                print(f"❌ Could not load depth: {depth_file}")
                return False
            
            print(f"📷 Processing pair {output_index:06d}")
            print(f"   Color shape: {color_img.shape}")
            print(f"   Depth shape: {depth_img.shape}, dtype: {depth_img.dtype}")
            
            # Resize to NeWCRFs standard size (480x640)
            target_size = (640, 480)  # (width, height)
            
            color_resized = cv2.resize(color_img, target_size)
            depth_resized = cv2.resize(depth_img, target_size)
            
            # Handle depth values
            if depth_img.dtype == np.uint16:
                # Already in millimeters, keep as is
                depth_final = depth_resized
            else:
                # Convert to uint16 millimeters
                depth_final = depth_resized.astype(np.uint16)
            
            # Clip depth to 1-2 meter range (1000-2000mm)
            depth_final = np.clip(depth_final, 0, 2000)
            
            # Create output paths
            rgb_output = os.path.join(self.output_path, "rgb", f"{output_index:06d}.png")
            depth_output = os.path.join(self.output_path, "depth", f"{output_index:06d}.png")
            
            # Save images
            cv2.imwrite(rgb_output, color_resized)
            cv2.imwrite(depth_output, depth_final)
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {color_file}: {e}")
            return False
    
    def create_training_structure(self, matched_pairs):
        """Create NeWCRFs training data structure"""
        print(f"\n📦 Creating NeWCRFs training structure...")
        
        # Create output directories
        rgb_dir = os.path.join(self.output_path, "rgb")
        depth_dir = os.path.join(self.output_path, "depth")
        
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        
        print(f"📁 Created: {rgb_dir}")
        print(f"📁 Created: {depth_dir}")
        
        # Process each pair
        success_count = 0
        filenames_list = []
        
        for i, (color_file, depth_file) in enumerate(matched_pairs):
            if self.process_frame_pair(color_file, depth_file, i):
                success_count += 1
                filenames_list.append(f"{i:06d}")
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"✅ Processed {i + 1}/{len(matched_pairs)} pairs")
        
        # Create filenames list for NeWCRFs
        filenames_file = os.path.join(self.output_path, "train_filenames.txt")
        with open(filenames_file, 'w') as f:
            for filename in filenames_list:
                f.write(f"{filename}\n")
        
        print(f"\n🎯 Training data created!")
        print(f"✅ Successfully processed: {success_count}/{len(matched_pairs)} pairs")
        print(f"📄 Filenames list: {filenames_file}")
        
        return success_count
    
    def create_sample_visualization(self):
        """Create sample overlay to verify alignment"""
        print("\n🖼️ Creating sample visualization...")
        
        rgb_dir = os.path.join(self.output_path, "rgb")
        depth_dir = os.path.join(self.output_path, "depth")
        
        # Load first few samples
        for i in range(min(3, len(os.listdir(rgb_dir)))):
            rgb_file = os.path.join(rgb_dir, f"{i:06d}.png")
            depth_file = os.path.join(depth_dir, f"{i:06d}.png")
            
            if os.path.exists(rgb_file) and os.path.exists(depth_file):
                rgb_img = cv2.imread(rgb_file)
                depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                
                # Create depth visualization
                depth_norm = cv2.convertScaleAbs(depth_img, alpha=255.0/2000.0)  # Normalize to 2m max
                depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                
                # Create overlay
                overlay = cv2.addWeighted(rgb_img, 0.6, depth_colored, 0.4, 0)
                
                # Save visualization
                viz_file = os.path.join(self.output_path, f"sample_overlay_{i:03d}.png")
                cv2.imwrite(viz_file, overlay)
                
                print(f"📷 Sample visualization saved: {viz_file}")
    
    def generate_newcrfs_command(self):
        """Generate the NeWCRFs training command"""
        print(f"\n🚀 NeWCRFs Training Command:")
        print("=" * 50)
        
        command = f"""cd /home/spacetop/SpaceTop/NeWCRFs

python newcrfs/train.py \\
    --mode train \\
    --model_name newcrfs \\
    --encoder large07 \\
    --dataset custom \\
    --data_path {self.output_path}/rgb \\
    --gt_path {self.output_path}/depth \\
    --filenames_file {self.output_path}/train_filenames.txt \\
    --input_height 480 \\
    --input_width 640 \\
    --max_depth 2.0 \\
    --batch_size 4 \\
    --num_epochs 50 \\
    --learning_rate 0.0001 \\
    --save_freq 500 \\
    --log_freq 100 \\
    --num_threads 2"""
        
        print(command)
        
        # Save command to file
        command_file = os.path.join(self.output_path, "run_training.sh")
        with open(command_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(command)
        
        print(f"\n💾 Command saved to: {command_file}")
        
    def run_complete_conversion(self):
        """Run the complete conversion process"""
        print("🎯 Starting D415 to NeWCRFs conversion...")
        
        # Step 1: Check data
        if not self.check_data():
            return False
        
        # Step 2: Find matching pairs
        matched_pairs = self.find_matching_pairs()
        if not matched_pairs:
            print("❌ No matching pairs found!")
            return False
        
        # Step 3: Create training structure
        success_count = self.create_training_structure(matched_pairs)
        if success_count == 0:
            print("❌ No frames processed successfully!")
            return False
        
        # Step 4: Create visualizations
        self.create_sample_visualization()
        
        # Step 5: Generate training command
        self.generate_newcrfs_command()
        
        print(f"\n✅ Conversion Complete!")
        print(f"📊 Processed {success_count} frame pairs")
        print(f"📁 Training data ready at: {self.output_path}")
        
        return True

def main():
    """Main function"""
    converter = SimpleD415ToNeWCRFs()
    
    # Run conversion
    success = converter.run_complete_conversion()
    
    if success:
        print(f"\n🎓 Next Steps:")
        print(f"1. Copy training data to Linux server")
        print(f"2. Run the generated training command")
        print(f"3. Monitor training progress")
    else:
        print(f"\n❌ Conversion failed. Check the errors above.")

if __name__ == "__main__":
    main()