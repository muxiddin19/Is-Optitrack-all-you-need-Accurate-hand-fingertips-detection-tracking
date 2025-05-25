#!/usr/bin/env python3
"""
Professional Data Capture System
Implements structured recording of aligned RGB-Depth data with proper formatting
"""

import cv2
import numpy as np
import json
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
import threading
from collections import deque

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠️  pyrealsense2 not available")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️  Open3D not available - some features disabled")

class DataCaptureSystem:
    """Professional data capture system for RGB-Depth training data"""
    
    def __init__(self, output_dir="training_data", calibration_file=None):
        self.output_dir = Path(output_dir)
        self.calibration_file = calibration_file
        self.calibration_data = None
        
        # Create structured directories
        self.setup_directories()
        
        # Camera systems
        self.d415_pipeline = None
        self.d405_pipeline = None
        self.platform_cameras = {}
        
        # Recording parameters
        self.recording = False
        self.frame_count = 0
        self.recording_fps = 30
        self.target_range = (1.0, 2.0)  # 1-2 meters optimal range
        
        # Data format settings
        self.depth_scale = 1000  # Save depth in millimeters (16-bit PNG)
        self.use_high_accuracy = True
        
        # Load calibration if provided
        if calibration_file:
            self.load_calibration()
    
    def setup_directories(self):
        """Create structured directory layout"""
        print(f"📁 Setting up data capture directories in {self.output_dir}")
        
        # Main directories
        self.images_dir = self.output_dir / "images"
        self.depths_dir = self.output_dir / "depths" 
        self.aligned_dir = self.output_dir / "aligned"
        self.metadata_dir = self.output_dir / "metadata"
        self.pointclouds_dir = self.output_dir / "pointclouds"
        
        # Create all directories
        for directory in [self.images_dir, self.depths_dir, self.aligned_dir, 
                         self.metadata_dir, self.pointclouds_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Directory structure created:")
        print(f"  📷 RGB Images: {self.images_dir}")
        print(f"  🎯 Depth Maps: {self.depths_dir}")
        print(f"  🔗 Aligned Data: {self.aligned_dir}")
        print(f"  📊 Metadata: {self.metadata_dir}")
        print(f"  ☁️  Point Clouds: {self.pointclouds_dir}")
    
    def load_calibration(self):
        """Load calibration data"""
        try:
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            print(f"✅ Loaded calibration from {self.calibration_file}")
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
    
    def setup_d415_high_accuracy(self, serial_number):
        """Setup D415 with high accuracy preset for 1-2m range"""
        print(f"🔧 Setting up D415 (SN: {serial_number}) with high accuracy preset...")
        
        try:
            self.d415_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial_number)
            
            # Configure streams for optimal quality
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.recording_fps)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.recording_fps)
            
            # Start pipeline
            profile = self.d415_pipeline.start(config)
            
            # Get device for advanced configuration
            device = profile.get_device()
            
            # Configure for high accuracy at 1-2m range
            depth_sensor = device.first_depth_sensor()
            
            # Set high accuracy preset
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
                print("  ✅ High accuracy preset enabled")
            
            # Optimize for close range (1-2m)
            if depth_sensor.supports(rs.option.depth_units):
                # Set depth units for maximum precision
                depth_sensor.set_option(rs.option.depth_units, 0.0001)  # 0.1mm units
                print("  ✅ Depth units set to 0.1mm")
            
            # Enable laser power for better accuracy
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, 360)  # Max laser power
                print("  ✅ Laser power maximized")
            
            # Enable auto-exposure for consistent quality
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            
            # Get intrinsics for data recording
            color_profile = profile.get_stream(rs.stream.color)
            depth_profile = profile.get_stream(rs.stream.depth)
            
            self.d415_color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            self.d415_depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            self.d415_depth_scale = depth_sensor.get_depth_scale()
            
            print(f"  ✅ D415 ready for high-quality data capture")
            print(f"      Depth scale: {self.d415_depth_scale}")
            print(f"      Color intrinsics: fx={self.d415_color_intrinsics.fx:.1f}")
            print(f"      Depth intrinsics: fx={self.d415_depth_intrinsics.fx:.1f}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ D415 setup failed: {e}")
            return False
    
    def setup_d405_close_range(self, serial_number):
        """Setup D405 optimized for close range accuracy"""
        print(f"🔧 Setting up D405 (SN: {serial_number}) for close range...")
        
        try:
            self.d405_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial_number)
            
            # Configure streams
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.recording_fps)
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, self.recording_fps)
            
            # Start pipeline
            profile = self.d405_pipeline.start(config)
            
            # Get device and optimize for close range
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            
            # D405 specific optimizations
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
            
            # Maximize laser power
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, 360)
                
            # Get intrinsics
            depth_profile = profile.get_stream(rs.stream.depth)
            self.d405_depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            self.d405_depth_scale = depth_sensor.get_depth_scale()
            
            print(f"  ✅ D405 ready for close-range depth capture")
            print(f"      Depth scale: {self.d405_depth_scale}")
            print(f"      Depth intrinsics: fx={self.d405_depth_intrinsics.fx:.1f}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ D405 setup failed: {e}")
            return False
    
    def create_intrinsics_file(self):
        """Create intrinsics JSON file for the dataset"""
        intrinsics_data = {
            "dataset_info": {
                "created": datetime.now().isoformat(),
                "format": "RGB-D dataset with aligned frames",
                "depth_format": "16-bit PNG in millimeters",
                "target_range_meters": self.target_range,
                "recording_fps": self.recording_fps
            }
        }
        
        # Add D415 intrinsics if available
        if hasattr(self, 'd415_color_intrinsics'):
            color_intr = self.d415_color_intrinsics
            depth_intr = self.d415_depth_intrinsics
            
            intrinsics_data["d415"] = {
                "color_intrinsics": {
                    "width": color_intr.width,
                    "height": color_intr.height,
                    "fx": color_intr.fx,
                    "fy": color_intr.fy,
                    "ppx": color_intr.ppx,
                    "ppy": color_intr.ppy,
                    "model": str(color_intr.model),
                    "coeffs": list(color_intr.coeffs)
                },
                "depth_intrinsics": {
                    "width": depth_intr.width,
                    "height": depth_intr.height,
                    "fx": depth_intr.fx,
                    "fy": depth_intr.fy,
                    "ppx": depth_intr.ppx,
                    "ppy": depth_intr.ppy,
                    "model": str(depth_intr.model),
                    "coeffs": list(depth_intr.coeffs)
                },
                "depth_scale": self.d415_depth_scale
            }
        
        # Add D405 intrinsics if available
        if hasattr(self, 'd405_depth_intrinsics'):
            depth_intr = self.d405_depth_intrinsics
            
            intrinsics_data["d405"] = {
                "depth_intrinsics": {
                    "width": depth_intr.width,
                    "height": depth_intr.height,
                    "fx": depth_intr.fx,
                    "fy": depth_intr.fy,
                    "ppx": depth_intr.ppx,
                    "ppy": depth_intr.ppy,
                    "model": str(depth_intr.model),
                    "coeffs": list(depth_intr.coeffs)
                },
                "depth_scale": self.d405_depth_scale
            }
        
        # Add calibration data if available
        if self.calibration_data:
            intrinsics_data["calibration"] = self.calibration_data
        
        # Save intrinsics file
        intrinsics_file = self.output_dir / "intrinsics.json"
        with open(intrinsics_file, 'w') as f:
            json.dump(intrinsics_data, f, indent=2)
        
        print(f"✅ Intrinsics file created: {intrinsics_file}")
        return intrinsics_file
    
    def align_depth_to_color_d415(self, color_frame, depth_frame):
        """Align depth to color using D415 hardware alignment"""
        try:
            # Create alignment object
            align_to_color = rs.align(rs.stream.color)
            
            # Note: This would be used in the capture loop with rs.frameset
            # For now, we assume frames are already aligned by hardware
            return depth_frame
            
        except Exception as e:
            print(f"❌ Hardware alignment failed: {e}")
            return depth_frame
    
    def manual_depth_alignment(self, rgb_frame, depth_frame, rgb_camera, depth_camera):
        """Manual depth alignment using calibration data"""
        if not self.calibration_data:
            print("❌ No calibration data for manual alignment")
            return depth_frame
        
        try:
            # Get calibration matrices
            pair_key = f"{depth_camera}_{rgb_camera}"
            if pair_key not in self.calibration_data.get('stereo_pairs', {}):
                print(f"❌ No calibration for {pair_key}")
                return depth_frame
            
            # Get transformation matrices
            stereo_data = self.calibration_data['stereo_pairs'][pair_key]
            R = np.array(stereo_data['rotation_matrix'])
            T = np.array(stereo_data['translation_vector'])
            
            # Get intrinsics
            rgb_intrinsics = np.array(self.calibration_data['intrinsics'][rgb_camera]['camera_matrix'])
            depth_intrinsics = np.array(self.calibration_data['intrinsics'][depth_camera]['camera_matrix'])
            
            # Convert depth to point cloud
            height, width = depth_frame.shape
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            
            # Get depth intrinsic parameters
            fx_d, fy_d = depth_intrinsics[0, 0], depth_intrinsics[1, 1]
            cx_d, cy_d = depth_intrinsics[0, 2], depth_intrinsics[1, 2]
            
            # Convert to 3D points in depth camera coordinates
            z = depth_frame.astype(np.float32) * 0.001  # Convert mm to meters
            x = (u - cx_d) * z / fx_d
            y = (v - cy_d) * z / fy_d
            
            # Transform to RGB camera coordinates
            points_3d = np.stack([x, y, z], axis=-1)
            points_flat = points_3d.reshape(-1, 3)
            
            # Apply transformation
            points_rgb = (R @ points_flat.T).T + T
            
            # Project to RGB image
            fx_rgb, fy_rgb = rgb_intrinsics[0, 0], rgb_intrinsics[1, 1]
            cx_rgb, cy_rgb = rgb_intrinsics[0, 2], rgb_intrinsics[1, 2]
            
            x_proj = points_rgb[:, 0] * fx_rgb / points_rgb[:, 2] + cx_rgb
            y_proj = points_rgb[:, 1] * fy_rgb / points_rgb[:, 2] + cy_rgb
            
            # Create aligned depth image
            aligned_depth = np.zeros_like(depth_frame)
            valid_mask = (points_rgb[:, 2] > 0) & (x_proj >= 0) & (x_proj < width) & (y_proj >= 0) & (y_proj < height)
            
            if np.any(valid_mask):
                x_valid = x_proj[valid_mask].astype(int)
                y_valid = y_proj[valid_mask].astype(int)
                depth_valid = (points_rgb[valid_mask, 2] * 1000).astype(np.uint16)  # Convert back to mm
                
                aligned_depth[y_valid, x_valid] = depth_valid
            
            return aligned_depth
            
        except Exception as e:
            print(f"❌ Manual alignment failed: {e}")
            return depth_frame
    
    def save_frame_pair(self, rgb_frame, depth_frame, frame_idx, camera_name="d415"):
        """Save RGB-Depth frame pair with proper formatting"""
        try:
            # Format frame index with leading zeros
            frame_str = f"{frame_idx:06d}"
            
            # Save RGB frame
            rgb_filename = self.images_dir / f"{frame_str}_color.png"
            cv2.imwrite(str(rgb_filename), rgb_frame)
            
            # Save depth frame as 16-bit PNG in millimeters
            depth_filename = self.depths_dir / f"{frame_str}_depth.png"
            
            # Ensure depth is in millimeters and 16-bit
            if depth_frame.dtype != np.uint16:
                depth_mm = (depth_frame * 1000).astype(np.uint16)
            else:
                depth_mm = depth_frame
            
            cv2.imwrite(str(depth_filename), depth_mm)
            
            # Create aligned visualization for verification
            aligned_filename = self.aligned_dir / f"{frame_str}_aligned.png"
            aligned_vis = self.create_rgb_depth_overlay(rgb_frame, depth_mm)
            cv2.imwrite(str(aligned_filename), aligned_vis)
            
            # Save metadata for this frame
            metadata = {
                "frame_index": frame_idx,
                "timestamp": time.time(),
                "camera": camera_name,
                "rgb_file": f"{frame_str}_color.png",
                "depth_file": f"{frame_str}_depth.png",
                "depth_format": "16-bit PNG, millimeters",
                "depth_stats": {
                    "min_depth_mm": int(np.min(depth_mm[depth_mm > 0])) if np.any(depth_mm > 0) else 0,
                    "max_depth_mm": int(np.max(depth_mm)),
                    "mean_depth_mm": float(np.mean(depth_mm[depth_mm > 0])) if np.any(depth_mm > 0) else 0
                }
            }
            
            metadata_filename = self.metadata_dir / f"{frame_str}_metadata.json"
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save frame {frame_idx}: {e}")
            return False
    
    def create_rgb_depth_overlay(self, rgb_frame, depth_frame):
        """Create RGB-depth overlay for visualization"""
        # Normalize depth for visualization
        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(rgb_frame, 0.7, depth_colored, 0.3, 0)
        
        # Add depth statistics
        valid_depth = depth_frame[depth_frame > 0]
        if len(valid_depth) > 0:
            min_depth = np.min(valid_depth)
            max_depth = np.max(valid_depth)
            mean_depth = np.mean(valid_depth)
            
            cv2.putText(overlay, f"Depth: {min_depth:.0f}-{max_depth:.0f}mm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, f"Mean: {mean_depth:.0f}mm", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay
    
    def save_point_cloud(self, rgb_frame, depth_frame, frame_idx, camera_name="d415"):
        """Save point cloud in PLY format"""
        if not OPEN3D_AVAILABLE:
            return False
        
        try:
            # Get intrinsics based on camera
            if camera_name == "d415" and hasattr(self, 'd415_color_intrinsics'):
                intrinsics = self.d415_color_intrinsics
                depth_scale = self.d415_depth_scale
            else:
                return False
            
            # Create Open3D RGB-D image
            color_o3d = o3d.geometry.Image(rgb_frame)
            depth_o3d = o3d.geometry.Image(depth_frame.astype(np.uint16))
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0/1000.0, convert_rgb_to_intensity=False)
            
            # Create camera intrinsic matrix
            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                intrinsics.width, intrinsics.height, 
                intrinsics.fx, intrinsics.fy, 
                intrinsics.ppx, intrinsics.ppy)
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
            
            # Save point cloud
            frame_str = f"{frame_idx:06d}"
            pcd_filename = self.pointclouds_dir / f"{frame_str}_pointcloud.ply"
            o3d.io.write_point_cloud(str(pcd_filename), pcd)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save point cloud {frame_idx}: {e}")
            return False
    
    def run_d415_data_capture(self):
        """Run D415 data capture session"""
        print(f"\n🎬 Starting D415 data capture session")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Target range: {self.target_range[0]:.1f}-{self.target_range[1]:.1f}m")
        print(f"🎮 Controls:")
        print(f"  • SPACE - Start/Stop recording")
        print(f"  • 's' - Save single frame")
        print(f"  • 'p' - Save point cloud")
        print(f"  • 'i' - Show capture info")
        print(f"  • 'q' - Quit")
        
        # Create intrinsics file
        self.create_intrinsics_file()
        
        # Hardware alignment for D415
        align_to_color = rs.align(rs.stream.color)
        
        capture_count = 0
        recording_start_time = 0
        
        try:
            while True:
                # Capture frames
                frames = self.d415_pipeline.wait_for_frames()
                
                # Apply hardware alignment
                aligned_frames = align_to_color.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Create display
                display_frame = self.create_rgb_depth_overlay(color_image, depth_image)
                
                # Add recording status
                if self.recording:
                    recording_time = time.time() - recording_start_time
                    cv2.putText(display_frame, f"RECORDING: {recording_time:.1f}s", (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Frames: {capture_count}", (10, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "Press SPACE to start recording", (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show range guidance
                valid_depth = depth_image[depth_image > 0]
                if len(valid_depth) > 0:
                    mean_depth_m = np.mean(valid_depth) * self.d415_depth_scale
                    if self.target_range[0] <= mean_depth_m <= self.target_range[1]:
                        range_color = (0, 255, 0)  # Green - optimal range
                        range_text = f"OPTIMAL RANGE: {mean_depth_m:.2f}m"
                    else:
                        range_color = (0, 255, 255)  # Yellow - suboptimal
                        range_text = f"Range: {mean_depth_m:.2f}m (target: {self.target_range[0]:.1f}-{self.target_range[1]:.1f}m)"
                    
                    cv2.putText(display_frame, range_text, (10, display_frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, range_color, 2)
                
                cv2.imshow('D415 Data Capture', display_frame)
                
                # Handle recording
                if self.recording:
                    if self.save_frame_pair(color_image, depth_image, capture_count, "d415"):
                        capture_count += 1
                        print(f"📸 Saved frame {capture_count}")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.recording = not self.recording
                    if self.recording:
                        recording_start_time = time.time()
                        print("🔴 Recording started")
                    else:
                        print(f"⏹️  Recording stopped. Captured {capture_count} frames")
                elif key == ord('s'):
                    if self.save_frame_pair(color_image, depth_image, capture_count, "d415"):
                        capture_count += 1
                        print(f"📸 Single frame saved: {capture_count}")
                elif key == ord('p'):
                    if self.save_point_cloud(color_image, depth_image, capture_count, "d415"):
                        print(f"☁️  Point cloud saved: {capture_count}")
                elif key == ord('i'):
                    self.print_capture_info(capture_count)
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        finally:
            cv2.destroyAllWindows()
            
        print(f"\n✅ Data capture complete!")
        print(f"📊 Total frames captured: {capture_count}")
        print(f"📁 Data saved to: {self.output_dir}")
        
        # Create final dataset summary
        self.create_dataset_summary(capture_count)
    
    def print_capture_info(self, frame_count):
        """Print detailed capture information"""
        print(f"\n{'='*50}")
        print("DATA CAPTURE INFORMATION")
        print('='*50)
        print(f"📊 Frames captured: {frame_count}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Target range: {self.target_range[0]:.1f}-{self.target_range[1]:.1f}m")
        print(f"📷 Recording FPS: {self.recording_fps}")
        print(f"💾 Depth format: 16-bit PNG (millimeters)")
        
        if hasattr(self, 'd415_depth_scale'):
            print(f"📏 D415 depth scale: {self.d415_depth_scale}")
        
        # Directory sizes
        rgb_count = len(list(self.images_dir.glob("*.png")))
        depth_count = len(list(self.depths_dir.glob("*.png")))
        aligned_count = len(list(self.aligned_dir.glob("*.png")))
        
        print(f"\n📁 File counts:")
        print(f"  RGB images: {rgb_count}")
        print(f"  Depth images: {depth_count}")
        print(f"  Aligned visualizations: {aligned_count}")
        
        print('='*50)
    
    def create_dataset_summary(self, total_frames):
        """Create final dataset summary"""
        summary = {
            "dataset_summary": {
                "total_frames": total_frames,
                "created": datetime.now().isoformat(),
                "target_range_meters": self.target_range,
                "recording_fps": self.recording_fps,
                "depth_format": "16-bit PNG in millimeters",
                "directory_structure": {
                    "images": "RGB frames as PNG",
                    "depths": "16-bit depth maps in millimeters", 
                    "aligned": "RGB-depth overlay visualizations",
                    "metadata": "Per-frame JSON metadata",
                    "pointclouds": "PLY point cloud files"
                },
                "filename_format": {
                    "rgb": "XXXXXX_color.png",
                    "depth": "XXXXXX_depth.png", 
                    "metadata": "XXXXXX_metadata.json"
                }
            }
        }
        
        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Dataset summary created: {summary_file}")
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up data capture system...")
        
        if self.d415_pipeline:
            try:
                self.d415_pipeline.stop()
            except:
                pass
                
        if self.d405_pipeline:
            try:
                self.d405_pipeline.stop()
            except:
                pass
        
        for cap in self.platform_cameras.values():
            try:
                cap.release()
            except:
                pass
        
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Professional Data Capture System')
    parser.add_argument('--output-dir', type=str, default='training_data',
                       help='Output directory for captured data')
    parser.add_argument('--calibration', type=str,
                       default='calibration_results/calibration_results.json',
                       help='Calibration file for manual alignment')
    parser.add_argument('--fps', type=int, default=30,
                       help='Recording frame rate')
    parser.add_argument('--min-range', type=float, default=1.0,
                       help='Minimum target range in meters')
    parser.add_argument('--max-range', type=float, default=2.0,
                       help='Maximum target range in meters')
    parser.add_argument('--camera', type=str, choices=['d415', 'd405', 'platform'],
                       default='d415', help='Primary camera for data capture')
    
    args = parser.parse_args()
    
    print("🎯 Professional Data Capture System")
    print("High-Quality RGB-Depth Training Data")
    print("=" * 50)
    
    try:
        # Create data capture system
        capture_system = DataCaptureSystem(
            output_dir=args.output_dir,
            calibration_file=args.calibration
        )
        
        # Set parameters
        capture_system.recording_fps = args.fps
        capture_system.target_range = (args.min_range, args.max_range)
        
        print(f"📁 Output directory: {args.output_dir}")
        print(f"🎯 Target range: {args.min_range:.1f}-{args.max_range:.1f}m")
        print(f"📷 Recording FPS: {args.fps}")
        
        # Setup cameras based on selection
        if args.camera == 'd415':
            # Use your discovered D415 serial
            if capture_system.setup_d415_high_accuracy("821312062833"):
                print("✅ D415 setup complete - starting data capture")
                capture_system.run_d415_data_capture()
            else:
                print("❌ Failed to setup D415")
                
        elif args.camera == 'd405':
            # Setup D405 + platform camera alignment
            if capture_system.setup_d405_close_range("230322270171"):
                print("✅ D405 setup complete")
                # Would need additional platform camera setup for RGB
                print("📝 D405 + Platform camera capture not yet implemented")
            else:
                print("❌ Failed to setup D405")
                
        else:
            print("📝 Platform camera capture not yet implemented")
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            capture_system.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()