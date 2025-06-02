import os
import cv2
import numpy as np
import glob
import json
from pathlib import Path
import shutil

class MultiCameraDepthAligner:
    def __init__(self):
        """Align D415 depth with platform camera RGB data"""
        print("🎯 Multi-Camera Depth Alignment System")
        print("=" * 50)
        print("📷 D415 Depth → Platform Camera RGB")
        
        # Base paths for your calibration data
        self.base_data_path = r"E:\vscode\calibration\data\user01\2"
        
        # D415 paths (source depth)
        self.d415_rgb_path = os.path.join(self.base_data_path, "d415", "rgb")
        self.d415_depth_path = os.path.join(self.base_data_path, "d415", "depth")
        
        # Platform camera paths (target RGB)
        self.platform_cameras = {
            'd405': os.path.join(self.base_data_path, "d405", "ir"),
            'cam1': os.path.join(self.base_data_path, "platform_cameras", "cam_1"), 
            'cam2': os.path.join(self.base_data_path, "platform_cameras", "cam_2")
        }
        
        # Output base path
        self.output_base = r"E:\vscode\newcrfs_training_data\aligned12"
        
        # Calibration data (you'll need to provide these)
        self.calibration_data = self.load_or_create_calibration()
        
        print(f"📁 D415 depth source: {self.d415_depth_path}")
        for cam_name, cam_path in self.platform_cameras.items():
            print(f"📁 {cam_name} RGB target: {cam_path}")
        print(f"📁 Output base: {self.output_base}")
    
    def load_or_create_calibration(self):
        """Load calibration data or create template"""
        calib_file = os.path.join(self.base_data_path, "calibration.json")
        
        if os.path.exists(calib_file):
            print(f"📋 Loading calibration from: {calib_file}")
            with open(calib_file, 'r') as f:
                return json.load(f)
        else:
            print("⚠️ No calibration file found, creating template...")
            return self.create_calibration_template(calib_file)
    
    def create_calibration_template(self, calib_file):
        """Create calibration template with default values"""
        calibration = {
            "d415": {
                "rgb_intrinsics": [[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]],
                "depth_intrinsics": [[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]],
                "rgb_distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "depth_distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "depth_scale": 0.001,
                "baseline": 0.055
            },
            "cam1": {
                "intrinsics": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "cam2": {
                "intrinsics": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "cam3": {
                "intrinsics": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "transformations": {
                "d415_to_cam1": {
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation": [0.0, 0.0, 0.0]
                },
                "d415_to_cam2": {
                    "rotation": [[0.998, -0.052, 0.035], [0.053, 0.998, -0.017], [-0.034, 0.019, 0.999]],
                    "translation": [0.1, 0.02, 0.05]
                },
                "d415_to_cam3": {
                    "rotation": [[0.998, 0.052, -0.035], [-0.053, 0.998, 0.017], [0.034, -0.019, 0.999]],
                    "translation": [-0.1, 0.02, 0.05]
                }
            }
        }
        
        # Save template
        with open(calib_file, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"📄 Calibration template created: {calib_file}")
        print("⚠️ Please update with your actual calibration parameters!")
        
        return calibration
    
    def check_data_availability(self):
        """Check which cameras have data available"""
        print("\n🔍 Checking data availability...")
        
        available_data = {}
        
        # Check D415 depth
        d415_depth_files = glob.glob(os.path.join(self.d415_depth_path, "*.png"))
        available_data['d415_depth'] = len(d415_depth_files)
        print(f"📊 D415 depth files: {len(d415_depth_files)}")
        
        # Check platform cameras
        for cam_name, cam_path in self.platform_cameras.items():
            if os.path.exists(cam_path):
                rgb_files = glob.glob(os.path.join(cam_path, "*.png"))
                available_data[cam_name] = len(rgb_files)
                print(f"📊 {cam_name} RGB files: {len(rgb_files)}")
            else:
                available_data[cam_name] = 0
                print(f"❌ {cam_name} path not found: {cam_path}")
        
        return available_data
    
    def find_synchronized_frames(self, cam_name, cam_path):
        """Find frames that exist in both D415 depth and target camera RGB"""
        print(f"\n🔗 Finding synchronized frames for {cam_name}...")
        
        # Get D415 depth files
        d415_depth_files = glob.glob(os.path.join(self.d415_depth_path, "*.png"))
        
        # Get platform camera RGB files
        cam_rgb_files = glob.glob(os.path.join(cam_path, "*.png"))
        
        # Extract frame numbers
        d415_frames = set()
        for depth_file in d415_depth_files:
            filename = Path(depth_file).name
            frame_num = filename.split('_')[0]  # Extract "000000" from "000000_depth_..."
            d415_frames.add(frame_num)
        
        cam_frames = {}
        for rgb_file in cam_rgb_files:
            filename = Path(rgb_file).name
            frame_num = filename.split('_')[0]  # Extract "000000" from "000000_rgb_..."
            cam_frames[frame_num] = rgb_file
        
        # Find synchronized frames
        synchronized_pairs = []
        for frame_num in d415_frames:
            if frame_num in cam_frames:
                d415_depth_file = os.path.join(self.d415_depth_path, f"{frame_num}_depth_*.png")
                d415_depth_matches = glob.glob(d415_depth_file)
                if d415_depth_matches:
                    synchronized_pairs.append((d415_depth_matches[0], cam_frames[frame_num]))
        
        print(f"✅ Found {len(synchronized_pairs)} synchronized pairs for {cam_name}")
        return synchronized_pairs
    
    def transform_depth_to_camera(self, depth_img, source_cam, target_cam):
        """Transform depth from D415 to target camera coordinate system"""
        
        # Get calibration parameters
        d415_calib = self.calibration_data['d415']
        target_calib = self.calibration_data[target_cam]
        
        # Get transformation
        transform_key = f"d415_to_{target_cam}"
        if transform_key not in self.calibration_data['transformations']:
            print(f"⚠️ No transformation found for {transform_key}, using identity")
            return depth_img  # Return original if no transformation
        
        transform = self.calibration_data['transformations'][transform_key]
        R = np.array(transform['rotation'])
        t = np.array(transform['translation'])
        
        # D415 depth intrinsics
        depth_intrinsics = np.array(d415_calib['depth_intrinsics'])
        fx_d, fy_d = depth_intrinsics[0, 0], depth_intrinsics[1, 1]
        cx_d, cy_d = depth_intrinsics[0, 2], depth_intrinsics[1, 2]
        
        # Target camera intrinsics
        target_intrinsics = np.array(target_calib['intrinsics'])
        fx_t, fy_t = target_intrinsics[0, 0], target_intrinsics[1, 1]
        cx_t, cy_t = target_intrinsics[0, 2], target_intrinsics[1, 2]
        
        # Convert depth to meters
        depth_scale = d415_calib.get('depth_scale', 0.001)
        depth_meters = depth_img.astype(np.float32) * depth_scale
        
        h, w = depth_meters.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Back-project to 3D points in D415 depth frame
        x = (u - cx_d) * depth_meters / fx_d
        y = (v - cy_d) * depth_meters / fy_d
        z = depth_meters
        
        # Stack into points (N, 3)
        points_3d = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        valid_mask = z.flatten() > 0
        
        # Transform points to target camera frame
        points_valid = points_3d[valid_mask]
        if len(points_valid) == 0:
            return np.zeros_like(depth_img)
        
        points_transformed = (R @ points_valid.T).T + t
        
        # Project to target camera image plane
        u_target = (points_transformed[:, 0] * fx_t / points_transformed[:, 2] + cx_t).astype(int)
        v_target = (points_transformed[:, 1] * fy_t / points_transformed[:, 2] + cy_t).astype(int)
        
        # Create aligned depth map
        aligned_depth = np.zeros((h, w), dtype=np.float32)
        
        # Filter valid projections
        valid_proj = ((u_target >= 0) & (u_target < w) & 
                     (v_target >= 0) & (v_target < h) &
                     (points_transformed[:, 2] > 0))
        
        if np.any(valid_proj):
            # Convert back to millimeters
            depth_values = (points_transformed[valid_proj, 2] / depth_scale).astype(np.uint16)
            aligned_depth[v_target[valid_proj], u_target[valid_proj]] = depth_values
        
        return aligned_depth.astype(np.uint16)
    
    def align_depth_to_camera(self, depth_file, rgb_file, target_cam, output_index, output_dir):
        """Align D415 depth to target camera RGB"""
        try:
            # Load images
            depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            rgb_img = cv2.imread(rgb_file)
            
            if depth_img is None or rgb_img is None:
                print(f"❌ Could not load images: {depth_file}, {rgb_file}")
                return False
            
            print(f"📷 Processing alignment {output_index:06d} for {target_cam}")
            print(f"   Depth shape: {depth_img.shape}, RGB shape: {rgb_img.shape}")
            
            # Transform depth to target camera coordinate system
            aligned_depth = self.transform_depth_to_camera(depth_img, 'd415', target_cam)
            
            # Resize both to NeWCRFs standard size (480x640)
            target_size = (640, 480)  # (width, height)
            
            rgb_resized = cv2.resize(rgb_img, target_size)
            depth_resized = cv2.resize(aligned_depth, target_size)
            
            # Clip depth to 1-2 meter range (1000-2000mm)
            depth_final = np.clip(depth_resized, 0, 2000)
            
            # Create output paths
            rgb_output = os.path.join(output_dir, "rgb", f"{output_index:06d}.png")
            depth_output = os.path.join(output_dir, "depth", f"{output_index:06d}.png")
            
            # Save images
            cv2.imwrite(rgb_output, rgb_resized)
            cv2.imwrite(depth_output, depth_final)
            
            return True
            
        except Exception as e:
            print(f"❌ Error aligning {depth_file} to {target_cam}: {e}")
            return False
    
    def create_training_dataset_for_camera(self, cam_name, cam_path):
        """Create training dataset for D415 depth → target camera RGB"""
        print(f"\n📦 Creating training dataset: D415 depth → {cam_name} RGB")
        
        # Create output directory
        output_dir = os.path.join(self.output_base, f"d415_depth_to_{cam_name}")
        rgb_dir = os.path.join(output_dir, "rgb")
        depth_dir = os.path.join(output_dir, "depth")
        
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        
        print(f"📁 Output directory: {output_dir}")
        
        # Find synchronized frames
        synchronized_pairs = self.find_synchronized_frames(cam_name, cam_path)
        
        if not synchronized_pairs:
            print(f"❌ No synchronized frames found for {cam_name}")
            return 0
        
        # Process each pair
        success_count = 0
        filenames_list = []
        
        for i, (depth_file, rgb_file) in enumerate(synchronized_pairs):
            if self.align_depth_to_camera(depth_file, rgb_file, cam_name, i, output_dir):
                success_count += 1
                filenames_list.append(f"{i:06d}")
            
            # Progress update
            if (i + 1) % 5 == 0:
                print(f"✅ Processed {i + 1}/{len(synchronized_pairs)} pairs")
        
        # Create filenames list for NeWCRFs
        filenames_file = os.path.join(output_dir, "train_filenames.txt")
        with open(filenames_file, 'w') as f:
            for filename in filenames_list:
                f.write(f"{filename}\n")
        
        # Create sample visualization
        self.create_alignment_visualization(output_dir, cam_name)
        
        # Generate training command
        self.generate_training_command(output_dir, cam_name)
        
        print(f"✅ {cam_name} dataset complete: {success_count}/{len(synchronized_pairs)} pairs")
        
        return success_count
    
    def create_alignment_visualization(self, output_dir, cam_name):
        """Create visualization to verify alignment quality"""
        print(f"🖼️ Creating alignment visualization for {cam_name}...")
        
        rgb_dir = os.path.join(output_dir, "rgb")
        depth_dir = os.path.join(output_dir, "depth")
        
        # Create overlays for first 3 samples
        for i in range(min(3, len(os.listdir(rgb_dir)))):
            rgb_file = os.path.join(rgb_dir, f"{i:06d}.png")
            depth_file = os.path.join(depth_dir, f"{i:06d}.png")
            
            if os.path.exists(rgb_file) and os.path.exists(depth_file):
                rgb_img = cv2.imread(rgb_file)
                depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                
                # Create depth visualization
                depth_norm = cv2.convertScaleAbs(depth_img, alpha=255.0/2000.0)
                depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                
                # Create overlay
                overlay = cv2.addWeighted(rgb_img, 0.6, depth_colored, 0.4, 0)
                
                # Save visualization
                viz_file = os.path.join(output_dir, f"alignment_check_{cam_name}_{i:03d}.png")
                cv2.imwrite(viz_file, overlay)
                
                print(f"📷 Alignment check saved: {viz_file}")
    
    def generate_training_command(self, output_dir, cam_name):
        """Generate NeWCRFs training command for this camera combination"""
        linux_output_path = output_dir.replace("E:\\vscode\\", "/home/spacetop/")
        
        command = f"""# Training command for D415 depth → {cam_name} RGB
cd /home/spacetop/SpaceTop/NeWCRFs

python newcrfs/train.py \\
    --mode train \\
    --model_name newcrfs_{cam_name} \\
    --encoder large07 \\
    --dataset custom \\
    --data_path {linux_output_path}/rgb \\
    --gt_path {linux_output_path}/depth \\
    --filenames_file {linux_output_path}/train_filenames.txt \\
    --input_height 480 \\
    --input_width 640 \\
    --max_depth 2.0 \\
    --batch_size 4 \\
    --num_epochs 50 \\
    --learning_rate 0.0001 \\
    --save_freq 500 \\
    --log_freq 100 \\
    --num_threads 2"""
        
        # Save command
        command_file = os.path.join(output_dir, f"train_{cam_name}.sh")
        with open(command_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(command)
        
        print(f"💾 Training command saved: {command_file}")
    
    def process_all_cameras(self):
        """Process all available platform cameras"""
        print("\n🚀 Starting Multi-Camera Depth Alignment Pipeline")
        print("=" * 60)
        
        # Check data availability
        available_data = self.check_data_availability()
        
        if available_data['d415_depth'] == 0:
            print("❌ No D415 depth data found!")
            return
        
        # Process each platform camera
        total_datasets = 0
        
        for cam_name, cam_path in self.platform_cameras.items():
            if available_data[cam_name] > 0:
                print(f"\n{cam_name.upper()} CAMERA PROCESSING")
                print("-" * 30)
                
                success_count = self.create_training_dataset_for_camera(cam_name, cam_path)
                if success_count > 0:
                    total_datasets += 1
                    print(f"✅ {cam_name} dataset created with {success_count} samples")
                else:
                    print(f"❌ {cam_name} dataset creation failed")
            else:
                print(f"⚠️ Skipping {cam_name} - no data available")
        
        print(f"\n🎯 Multi-Camera Alignment Complete!")
        print(f"📊 Created {total_datasets} training datasets")
        print(f"📁 All results saved to: {self.output_base}")
        
        # Create combined dataset summary
        self.create_summary_report(total_datasets)
    
    def create_summary_report(self, total_datasets):
        """Create summary report of all generated datasets"""
        summary_file = os.path.join(self.output_base, "alignment_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("Multi-Camera Depth Alignment Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Source: D415 Depth Camera\n")
            f.write(f"Targets: Platform Cameras (cam1, cam2, cam3)\n")
            f.write(f"Created Datasets: {total_datasets}\n\n")
            
            f.write("Generated Training Datasets:\n")
            for cam_name in self.platform_cameras.keys():
                dataset_dir = os.path.join(self.output_base, f"d415_depth_to_{cam_name}")
                if os.path.exists(dataset_dir):
                    rgb_count = len(glob.glob(os.path.join(dataset_dir, "rgb", "*.png")))
                    f.write(f"- d415_depth_to_{cam_name}: {rgb_count} samples\n")
            
            f.write(f"\nNext Steps:\n")
            f.write(f"1. Copy datasets to Linux server\n")
            f.write(f"2. Run individual training commands for each camera\n")
            f.write(f"3. Compare model performance across different camera viewpoints\n")
        
        print(f"📄 Summary report saved: {summary_file}")

def main():
    """Main function"""
    aligner = MultiCameraDepthAligner()
    
    print("🎯 Multi-Camera Depth Alignment")
    print("This will align D415 depth with each platform camera RGB")
    
    # Check if user wants to update calibration
    print("\n⚠️ IMPORTANT: Check your calibration parameters!")
    print(f"Calibration file: {os.path.join(aligner.base_data_path, 'calibration.json')}")
    
    response = input("\nProceed with alignment? (y/n): ").strip().lower()
    if response != 'y':
        print("👋 Exiting. Please update calibration and run again.")
        return
    
    # Run alignment
    aligner.process_all_cameras()
    
    print(f"\n🎓 Multi-Camera Alignment Complete!")
    print(f"📁 Check results in: {aligner.output_base}")

if __name__ == "__main__":
    main()