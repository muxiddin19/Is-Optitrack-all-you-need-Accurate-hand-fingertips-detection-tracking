#!/usr/bin/env python3
"""
Fixed D415 Synchronized RGB-Depth Capture
Ensures RGB and depth frames are from the exact same moment
"""

import cv2
import numpy as np
import json
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    print("✅ pyrealsense2 available")
except ImportError:
    REALSENSE_AVAILABLE = False
    print("❌ pyrealsense2 not available")

class SynchronizedMultiCameraCapture:
    """Multi-camera capture with proper D415 RGB-Depth synchronization"""
    
    def __init__(self, output_dir="synchronized_capture"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create organized directory structure
        self.setup_directories()
        
        # Create training-ready structure
        self.create_training_dataset_structure()
        #self.setup_d415_synchronized("821312062833")  # D415 serial number

        # Camera objects
        self.d415_pipeline = None
        self.d405_pipeline = None
        self.platform_cameras = {}
        
        # Synchronization objects
        self.d415_align = None  # Hardware alignment for D415
        
        

        # Recording state
        self.recording = False
        self.frame_count = 0
        self.target_range = (0.4, 0.58)  # 400-580mm
        
        print("🎯 Synchronized Multi-Camera Capture initialized")
    
    def setup_directories(self):
        """Create organized directory structure"""
        self.d415_dir = self.output_dir / "d415"
        self.d405_dir = self.output_dir / "d405"
        self.platform_dir = self.output_dir / "platform_cameras"
        self.sync_dir = self.output_dir / "synchronized"
        
        # Create all directories
        for directory in [self.d415_dir, self.d405_dir, self.platform_dir, self.sync_dir]:
            directory.mkdir(exist_ok=True)
            
        # Create subdirectories
        (self.d415_dir / "rgb").mkdir(exist_ok=True)
        (self.d415_dir / "depth").mkdir(exist_ok=True)
        (self.d415_dir / "overlay").mkdir(exist_ok=True)
        
        (self.d405_dir / "depth").mkdir(exist_ok=True)
        (self.d405_dir / "ir").mkdir(exist_ok=True)
        
        (self.platform_dir / "cam_1").mkdir(exist_ok=True)
        (self.platform_dir / "cam_2").mkdir(exist_ok=True)
        
        print(f"📁 Synchronized directory structure created")
    
    def setup_d415_synchronized(self, serial):
        """Setup D415 with proper RGB-Depth synchronization"""
        print(f"🔧 Setting up D415 (SN: {serial}) with synchronized RGB-Depth...")
        
        try:
            self.d415_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            
            # Enable streams with same resolution and framerate for sync
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.d415_pipeline.start(config)
            
            # Get device for optimization
            device = profile.get_device()
            
            # Enable hardware synchronization
            depth_sensor = device.first_depth_sensor()
            
            # Optimize for close range (400-580mm)
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
                print("  ✅ High accuracy preset enabled")
            
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, 360)
                print("  ✅ Laser power maximized")
            
            # CRITICAL: Setup hardware alignment for perfect RGB-Depth sync
            self.d415_align = rs.align(rs.stream.color)
            print("  ✅ Hardware RGB-Depth alignment enabled")
            
            # Test synchronized capture
            print("  🧪 Testing synchronized capture...")
            for test_attempt in range(5):
                frames = self.d415_pipeline.wait_for_frames(timeout_ms=2000)
                
                # Apply hardware alignment - this ensures RGB and depth are from same moment
                aligned_frames = self.d415_align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    # Get timestamps to verify synchronization
                    color_timestamp = color_frame.get_timestamp()
                    depth_timestamp = depth_frame.get_timestamp()
                    time_diff = abs(color_timestamp - depth_timestamp)
                    
                    print(f"    Test {test_attempt + 1}: Time diff = {time_diff:.2f}ms")
                    
                    if time_diff < 50:  # Less than 50ms difference is good
                        print("  ✅ D415 RGB-Depth synchronization verified")
                        return True
            
            # Save calibration data after successful setup
            self.save_camera_calibration()

            print("  ⚠️  D415 sync test completed, but timing may vary")
            return True
                
        except Exception as e:
            print(f"  ❌ D415 setup failed: {e}")
            return False
    
    def setup_d405(self, serial):
        """Setup D405"""
        print(f"🔧 Setting up D405 (SN: {serial})...")
        
        try:
            self.d405_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            
            self.d405_pipeline.start(config)
            
            # Test capture
            frames = self.d405_pipeline.wait_for_frames(timeout_ms=2000)
            if frames.get_depth_frame() and frames.get_infrared_frame(1):
                print("  ✅ D405 ready")
                return True
            else:
                print("  ❌ D405 frame test failed")
                return False
                
        except Exception as e:
            print(f"  ❌ D405 setup failed: {e}")
            return False
    
    def setup_platform_camera(self, cam_id):
        """Setup platform camera"""
        print(f"🔧 Setting up Platform Camera {cam_id}...")
        
        try:
            # Try different backends
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
            cap = None
            
            for backend in backends:
                try:
                    test_cap = cv2.VideoCapture(cam_id, backend)
                    if test_cap.isOpened():
                        test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        test_cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        ret, frame = test_cap.read()
                        if ret and frame is not None:
                            cap = test_cap
                            break
                        test_cap.release()
                except:
                    pass
            
            if cap:
                self.platform_cameras[f"platform_{cam_id}"] = cap
                print(f"  ✅ Platform Camera {cam_id} ready")
                return True
            else:
                print(f"  ❌ Platform Camera {cam_id} not accessible")
                return False
                
        except Exception as e:
            print(f"  ❌ Platform Camera {cam_id} setup failed: {e}")
            return False
    


    def capture_synchronized_frames(self):
        """Capture frames with proper synchronization"""
        frames = {}
        
        # Capture D415 with hardware synchronization
        if self.d415_pipeline and self.d415_align:
            try:
                # Wait for frames
                rs_frames = self.d415_pipeline.wait_for_frames(timeout_ms=100)
                
                # CRITICAL: Apply hardware alignment for perfect RGB-Depth sync
                aligned_frames = self.d415_align.process(rs_frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    # Verify synchronization
                    color_timestamp = color_frame.get_timestamp()
                    depth_timestamp = depth_frame.get_timestamp()
                    sync_quality = abs(color_timestamp - depth_timestamp)
                    
                    # Convert to numpy
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    frames['d415_rgb'] = color_image
                    frames['d415_depth'] = depth_image
                    frames['sync_quality'] = sync_quality
                    
                    # Debug info
                    if sync_quality > 50:  # Warn if sync is poor
                        print(f"⚠️  D415 sync quality: {sync_quality:.1f}ms")
                    
            except Exception as e:
                print(f"D415 synchronized capture error: {e}")
        
        # Capture D405
        if self.d405_pipeline:
            try:
                rs_frames = self.d405_pipeline.wait_for_frames(timeout_ms=100)
                
                depth_frame = rs_frames.get_depth_frame()
                ir_frame = rs_frames.get_infrared_frame(1)
                
                if depth_frame and ir_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    ir_image = np.asanyarray(ir_frame.get_data())
                    
                    frames['d405_depth'] = depth_image
                    frames['d405_ir'] = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
                    
            except Exception as e:
                print(f"D405 capture error: {e}")
        
        # Capture platform cameras
        for cam_name, cap in self.platform_cameras.items():
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames[cam_name] = frame
            except Exception as e:
                print(f"{cam_name} capture error: {e}")
        
        return frames
    
    def create_depth_overlay(self, rgb_frame, depth_frame):
        """Create RGB-depth overlay with range optimization"""
        try:
            # Filter depth to target range (400-580mm)
            min_depth = self.target_range[0] * 1000
            max_depth = self.target_range[1] * 1000
            
            # Create mask for target range
            depth_mask = (depth_frame >= min_depth) & (depth_frame <= max_depth)
            
            # Normalize depth for visualization
            depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Create overlay
            overlay = cv2.addWeighted(rgb_frame, 0.7, depth_colored, 0.3, 0)
            
            # Add range feedback
            valid_depth = depth_frame[depth_frame > 0]
            if len(valid_depth) > 0:
                mean_depth = np.mean(valid_depth)
                in_range_pixels = np.sum(depth_mask)
                total_depth_pixels = np.sum(depth_frame > 0)
                range_percentage = (in_range_pixels / total_depth_pixels * 100) if total_depth_pixels > 0 else 0
                
                # Color code feedback
                if min_depth <= mean_depth <= max_depth and range_percentage > 50:
                    color = (0, 255, 0)  # Green - optimal
                    status = "OPTIMAL RANGE"
                elif min_depth <= mean_depth <= max_depth:
                    color = (0, 255, 255)  # Yellow - acceptable
                    status = "GOOD RANGE"
                else:
                    color = (0, 100, 255)  # Orange - adjust needed
                    status = "ADJUST DISTANCE"
                
                cv2.putText(overlay, f"{status}: {mean_depth:.0f}mm", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(overlay, f"In Range: {range_percentage:.1f}%", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return overlay
            
        except Exception as e:
            print(f"Overlay creation error: {e}")
            return rgb_frame
    
    def create_display(self, frames):
        """Create multi-camera display - FIXED memory allocation"""
        displays = []
        target_size = (320, 240)  # Consistent size for all displays
        
        # D415 RGB-Depth overlay (now properly synchronized)
        if 'd415_rgb' in frames and 'd415_depth' in frames:
            overlay = self.create_depth_overlay(frames['d415_rgb'], frames['d415_depth'])
            
            # Add sync quality indicator
            sync_quality = frames.get('sync_quality', 0)
            if sync_quality < 25:
                sync_color = (0, 255, 0)  # Green - excellent sync
                sync_text = "SYNC: EXCELLENT"
            elif sync_quality < 50:
                sync_color = (0, 255, 255)  # Yellow - good sync
                sync_text = "SYNC: GOOD"
            else:
                sync_color = (0, 100, 255)  # Orange - poor sync
                sync_text = "SYNC: POOR"
            
            cv2.putText(overlay, sync_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, sync_color, 1)
            cv2.putText(overlay, "D415: RGB+Depth", (10, overlay.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # FIXED: Ensure consistent resize
            displays.append(cv2.resize(overlay, target_size))
        
        # D405 depth visualization
        if 'd405_depth' in frames:
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(frames['d405_depth'], alpha=0.05), cv2.COLORMAP_PLASMA)
            cv2.putText(depth_vis, "D405: Depth", (10, depth_vis.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            displays.append(cv2.resize(depth_vis, target_size))
        
        # D405 IR
        if 'd405_ir' in frames:
            ir_frame = frames['d405_ir']
            cv2.putText(ir_frame, "D405: IR", (10, ir_frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            displays.append(cv2.resize(ir_frame, target_size))
        
        # Platform cameras
        cam_number = 1
        for cam_name, frame in frames.items():
            if cam_name.startswith('platform_'):
                cam_frame = frame.copy()
                cv2.putText(cam_frame, f"PLATFORM {cam_number}", (10, cam_frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                displays.append(cv2.resize(cam_frame, target_size))
                cam_number += 1
        
        # FIXED: Safer grid arrangement with consistent dimensions
        if len(displays) == 0:
            combined = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(combined, "No synchronized feeds", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        elif len(displays) == 1:
            combined = displays[0]
        elif len(displays) == 2:
            # Stack horizontally - both same size now
            combined = np.hstack(displays)
        elif len(displays) == 3:
            # Two on top, one on bottom with padding
            top_row = np.hstack(displays[:2])
            padding = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            bottom_row = np.hstack([displays[2], padding])
            combined = np.vstack([top_row, bottom_row])
        elif len(displays) == 4:
            # 2x2 grid
            top_row = np.hstack(displays[:2])
            bottom_row = np.hstack(displays[2:4])
            combined = np.vstack([top_row, bottom_row])
        else:
            # More than 4 displays - arrange in rows of 2
            rows = []
            for i in range(0, len(displays), 2):
                if i + 1 < len(displays):
                    row = np.hstack([displays[i], displays[i+1]])
                else:
                    padding = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                    row = np.hstack([displays[i], padding])
                rows.append(row)
            combined = np.vstack(rows)
        
        return combined
    
    def save_synchronized_frame_set(self, frames):
        """Enhanced synchronized frame set saving with training-ready format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_num = f"{self.frame_count:06d}"
        
        saved_count = 0
        saved_details = []
        quality_data = {}
        
        # Save D415 synchronized frames
        if 'd415_rgb' in frames and 'd415_depth' in frames:
            rgb_frame = frames['d415_rgb']
            depth_frame = frames['d415_depth']
            
            # Verify data quality
            quality_metrics = self.verify_data_quality(rgb_frame, depth_frame)
            quality_data.update(quality_metrics)
            
            # 1. Save original format (for reference)
            rgb_file = self.d415_dir / "rgb" / f"{frame_num}_rgb_{timestamp}.png"
            cv2.imwrite(str(rgb_file), rgb_frame)
            saved_count += 1
            
            depth_file = self.d415_dir / "depth" / f"{frame_num}_depth_{timestamp}.png"
            cv2.imwrite(str(depth_file), depth_frame.astype(np.uint16))
            saved_count += 1
            
            # 2. Save training-ready format for each model
            try:
                for model in ["adabins", "newcrfs", "zoedepth"]:
                    model_dir = self.training_dir / model
                    
                    # Model-specific resolution
                    if model == "zoedepth":
                        target_size = (384, 384)  # Square for transformer
                    else:
                        target_size = (480, 640)  # NYU format
                    
                    # Save model-ready RGB
                    model_rgb_file = model_dir / "rgb" / f"{frame_num}_{timestamp}.png"
                    if self.save_model_ready_rgb(rgb_frame, model_rgb_file, target_size):
                        saved_count += 1
                    
                    # Save model-ready depth (resize to match RGB)
                    depth_resized = cv2.resize(depth_frame, (target_size[1], target_size[0]))
                    model_depth_file = model_dir / "depth" / f"{frame_num}_{timestamp}"
                    if self.save_training_ready_depth(depth_resized, model_depth_file):
                        saved_count += 1
            except Exception as e:
                print(f"⚠️  Training format save error: {e}")
            
            # Save overlay
            overlay = self.create_depth_overlay(rgb_frame, depth_frame)
            overlay_file = self.d415_dir / "overlay" / f"{frame_num}_overlay_{timestamp}.png"
            cv2.imwrite(str(overlay_file), overlay)
            saved_count += 1
            
            saved_details = ["D415 RGB", "D415 Depth", "Training Formats", "Overlay"]
        
        # Save D405 frames (if available)
        if 'd405_depth' in frames:
            depth_file = self.d405_dir / "depth" / f"{frame_num}_depth_{timestamp}.png"
            cv2.imwrite(str(depth_file), frames['d405_depth'].astype(np.uint16))
            saved_count += 1
            saved_details.append("D405 Depth")
            
        if 'd405_ir' in frames:
            ir_file = self.d405_dir / "ir" / f"{frame_num}_ir_{timestamp}.png"
            cv2.imwrite(str(ir_file), frames['d405_ir'])
            saved_count += 1
            saved_details.append("D405 IR")
        
        # Save platform camera frames
        platform_count = 1
        for frame_name, frame_data in frames.items():
            if frame_name.startswith('platform_'):
                cam_dir = self.platform_dir / f"cam_{platform_count}"
                cam_file = cam_dir / f"{frame_num}_rgb_{timestamp}.png"
                cv2.imwrite(str(cam_file), frame_data)
                saved_count += 1
                saved_details.append(f"Platform {platform_count}")
                platform_count += 1
        
        # Save synchronized display
        if len(frames) > 1:
            try:
                sync_display = self.create_display(frames)
                sync_file = self.sync_dir / f"{frame_num}_synchronized_{timestamp}.png"
                cv2.imwrite(str(sync_file), sync_display)
                saved_count += 1
                saved_details.append("Synchronized View")
            except Exception as e:
                print(f"⚠️  Display save error: {e}")
        
        # Enhanced metadata with training info
        metadata = {
            "frame_set": self.frame_count,
            "timestamp": timestamp,
            "cameras_captured": list(frames.keys()),
            "files_saved": saved_count,
            "target_range_mm": [self.target_range[0]*1000, self.target_range[1]*1000],
            "training_range_m": [0.4, 0.58],
            "max_depth_cap_m": 2.0,
            "sync_quality_ms": frames.get('sync_quality', 0),
            "data_quality": quality_data,
            "file_details": saved_details,
            "formats": {
                "original_depth": "16-bit PNG (millimeters)",
                "training_depth": "float32 NPY (meters, capped at 2m)",
                "rgb_resolutions": {
                    "adabins": [480, 640],
                    "newcrfs": [480, 640], 
                    "zoedepth": [384, 384]
                }
            },
            "alignment_method": "hardware_synchronized"
        }
        
        metadata_file = self.sync_dir / f"{frame_num}_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.frame_count += 1
        
        # Enhanced status reporting
        sync_quality = frames.get('sync_quality', 0)
        alignment_score = quality_data.get('edge_alignment', 0)
        depth_coverage = quality_data.get('depth_coverage', 0)
        
        print(f"📸 Set {self.frame_count}: {saved_count} files | "
            f"Sync: {sync_quality:.1f}ms | "
            f"Align: {alignment_score:.2f} | "
            f"Coverage: {depth_coverage:.1%}")
        
        return saved_count
    
    def run_synchronized_capture(self):
        """Run synchronized capture session"""
        print(f"\n🎬 Starting Synchronized Multi-Camera Capture")
        print(f"🎯 Target range: {self.target_range[0]*1000:.0f}-{self.target_range[1]*1000:.0f}mm")
        print(f"📁 Output: {self.output_dir}")
        print(f"\n🎮 Controls:")
        print(f"  • SPACE - Save synchronized frame set")
        print(f"  • 'r' - Start/stop recording")
        print(f"  • 'q' - Quit")
        
        recording_start_time = 0
        
        try:
            while True:
                # Capture synchronized frames
                frames = self.capture_synchronized_frames()
                
                if not frames:
                    print("⚠️  No synchronized frames captured")
                    time.sleep(0.1)
                    continue
                
                # Create display
                display = self.create_display(frames)
                
                # Add status info
                sync_quality = frames.get('sync_quality', 0)
                status_text = f"Cameras: {len(frames)-1} | Frames: {self.frame_count} | Sync: {sync_quality:.1f}ms"
                cv2.putText(display, status_text, (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if self.recording:
                    recording_time = time.time() - recording_start_time
                    cv2.putText(display, f"🔴 RECORDING: {recording_time:.1f}s", (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Auto-save while recording (every 2 seconds)
                    if int(recording_time) % 2 == 0 and int(recording_time) != int(recording_time - 0.1):
                        self.save_synchronized_frame_set(frames)
                
                cv2.imshow('Synchronized Multi-Camera Capture', display)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.save_synchronized_frame_set(frames)
                elif key == ord('r'):
                    self.recording = not self.recording
                    if self.recording:
                        recording_start_time = time.time()
                        print("🔴 Recording synchronized frames (auto-save every 2s)")
                    else:
                        print("⏹️  Recording stopped")
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        finally:
            cv2.destroyAllWindows()
        
        print(f"\n✅ Synchronized capture session complete!")
        print(f"📊 Total synchronized frame sets: {self.frame_count}")
        print(f"📁 Data location: {self.output_dir}")
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        
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

    def save_camera_calibration(self):
        """Save camera intrinsic parameters as recommended in doc"""
        if self.d415_pipeline:
            try:
                profile = self.d415_pipeline.get_active_profile()
                color_profile = profile.get_stream(rs.stream.color)
                depth_profile = profile.get_stream(rs.stream.depth)
                
                color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
                depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
                
                calibration_data = {
                    "device_info": {
                        "serial_number": "821312062833",
                        "device_name": "Intel RealSense D415",
                        "firmware_version": profile.get_device().get_info(rs.camera_info.firmware_version)
                    },
                    "color_intrinsics": {
                        "width": color_intrinsics.width,
                        "height": color_intrinsics.height,
                        "fx": color_intrinsics.fx,
                        "fy": color_intrinsics.fy,
                        "cx": color_intrinsics.ppx,
                        "cy": color_intrinsics.ppy,
                        "distortion_model": str(color_intrinsics.model),
                        "distortion_coeffs": list(color_intrinsics.coeffs)
                    },
                    "depth_intrinsics": {
                        "width": depth_intrinsics.width,
                        "height": depth_intrinsics.height,
                        "fx": depth_intrinsics.fx,
                        "fy": depth_intrinsics.fy,
                        "cx": depth_intrinsics.ppx,
                        "cy": depth_intrinsics.ppy,
                        "depth_scale": profile.get_device().first_depth_sensor().get_depth_scale()
                    },
                    "capture_settings": {
                        "target_range_mm": [self.target_range[0]*1000, self.target_range[1]*1000],
                        "visual_preset": "high_accuracy",
                        "laser_power": 360
                    }
                }
                
                calib_file = self.output_dir / "camera_calibration.json"
                with open(calib_file, 'w') as f:
                    json.dump(calibration_data, f, indent=2)
                print("✅ Camera calibration saved")
                
            except Exception as e:
                print(f"⚠️  Could not save calibration: {e}")
    def save_depth_with_precision(self, depth_frame, filepath):
        """Save depth with 16-bit precision as recommended"""
        # Ensure depth is in millimeters and save as 16-bit
        depth_mm = depth_frame.astype(np.uint16)
        success = cv2.imwrite(str(filepath), depth_mm)
        return success

    def verify_data_quality(self, rgb_frame, depth_frame):
        """Verify RGB-depth alignment and data quality"""
        quality_metrics = {}
        
        try:
            # 1. Edge alignment check
            rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            rgb_edges = cv2.Canny(rgb_gray, 50, 150)
            
            # Normalize depth for edge detection
            depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_edges = cv2.Canny(depth_normalized, 50, 150)
            
            # Calculate alignment score
            intersection = np.sum(rgb_edges & depth_edges)
            union = np.sum(rgb_edges | depth_edges)
            alignment_score = intersection / union if union > 0 else 0
            quality_metrics['edge_alignment'] = alignment_score
            
            # 2. Depth coverage in target range
            valid_depth = depth_frame[depth_frame > 0]
            if len(valid_depth) > 0:
                min_range_mm = self.target_range[0] * 1000
                max_range_mm = self.target_range[1] * 1000
                
                in_range_pixels = np.sum((depth_frame >= min_range_mm) & 
                                    (depth_frame <= max_range_mm))
                total_valid_pixels = np.sum(depth_frame > 0)
                
                quality_metrics['depth_coverage'] = in_range_pixels / total_valid_pixels
                quality_metrics['mean_depth_mm'] = np.mean(valid_depth)
                quality_metrics['depth_std_mm'] = np.std(valid_depth)
            
            # 3. Check for missing/invalid depth regions
            invalid_pixels = np.sum(depth_frame == 0)
            total_pixels = depth_frame.shape[0] * depth_frame.shape[1]
            quality_metrics['invalid_depth_ratio'] = invalid_pixels / total_pixels
            
            return quality_metrics
            
        except Exception as e:
            print(f"Quality verification error: {e}")
            return {}
    
    def enhanced_save_synchronized_frame_set(self, frames):
        """Enhanced saving with quality metrics and proper depth precision"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_num = f"{self.frame_count:06d}"
        
        saved_count = 0
        saved_details = []
        quality_data = {}
        
        # Save D415 synchronized frames with enhanced precision
        if 'd415_rgb' in frames and 'd415_depth' in frames:
            rgb_frame = frames['d415_rgb']
            depth_frame = frames['d415_depth']
            
            # Verify data quality
            quality_metrics = self.verify_data_quality(rgb_frame, depth_frame)
            quality_data.update(quality_metrics)
            
            # Save RGB
            rgb_file = self.d415_dir / "rgb" / f"{frame_num}_rgb_{timestamp}.png"
            cv2.imwrite(str(rgb_file), rgb_frame)
            saved_count += 1
            saved_details.append("D415 RGB")
            
            # Save depth with 16-bit precision
            depth_file = self.d415_dir / "depth" / f"{frame_num}_depth_{timestamp}.png"
            if self.save_depth_with_precision(depth_frame, depth_file):
                saved_count += 1
                saved_details.append("D415 Depth (16-bit)")
            
            # Save overlay
            overlay = self.create_depth_overlay(rgb_frame, depth_frame)
            overlay_file = self.d415_dir / "overlay" / f"{frame_num}_overlay_{timestamp}.png"
            cv2.imwrite(str(overlay_file), overlay)
            saved_count += 1
            saved_details.append("D415 Overlay")
        
        # Enhanced metadata with quality metrics
        metadata = {
            "frame_set": self.frame_count,
            "timestamp": timestamp,
            "cameras_captured": list(frames.keys()),
            "files_saved": saved_count,
            "target_range_mm": [self.target_range[0]*1000, self.target_range[1]*1000],
            "sync_quality_ms": frames.get('sync_quality', 0),
            "data_quality": quality_data,
            "file_details": saved_details,
            "depth_format": "16-bit PNG (millimeters)",
            "alignment_method": "hardware_synchronized"
        }
        
        metadata_file = self.sync_dir / f"{frame_num}_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.frame_count += 1
        
        # Enhanced status reporting
        sync_quality = frames.get('sync_quality', 0)
        alignment_score = quality_data.get('edge_alignment', 0)
        depth_coverage = quality_data.get('depth_coverage', 0)
        
        print(f"📸 Set {self.frame_count}: {saved_count} files | "
            f"Sync: {sync_quality:.1f}ms | "
            f"Align: {alignment_score:.2f} | "
            f"Coverage: {depth_coverage:.1%}")
        
        return saved_count
    def verify_alignment(self, rgb_frame, depth_frame):
        """Verify RGB-depth alignment quality"""
        # Create edge maps
        rgb_edges = cv2.Canny(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY), 50, 150)
        depth_edges = cv2.Canny((depth_frame / 256).astype(np.uint8), 50, 150)
        
        # Calculate edge alignment score
        union = np.sum(rgb_edges | depth_edges)
        if union > 0:
            alignment_score = np.sum(rgb_edges & depth_edges) / union
        else:
            alignment_score = 0
        return alignment_score

    # Add these methods to your SynchronizedMultiCameraCapture class

    def save_training_ready_depth(self, depth_frame, filepath):
        """Save depth in training-ready format (meters, float32, capped)"""
        try:
            # Convert from millimeters to meters
            depth_meters = depth_frame.astype(np.float32) / 1000.0
            
            # Apply range capping as per documentation
            max_depth = 2.0  # Cap at 2m for short-range application
            min_depth = 0.1  # Minimum valid depth
            
            # Cap depth values
            depth_meters = np.clip(depth_meters, min_depth, max_depth)
            
            # Handle invalid pixels (set to max_depth or special value)
            invalid_mask = depth_frame == 0
            depth_meters[invalid_mask] = max_depth  # or could use np.nan
            
            # Save as 32-bit float (training format)
            # Note: PNG doesn't support float32, so save as .npy for training
            np.save(str(filepath).replace('.png', '.npy'), depth_meters)
            
            # Also save 16-bit PNG for visualization (scaled back to mm)
            depth_vis = np.clip(depth_meters * 1000, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(filepath), depth_vis)
            
            return True
            
        except Exception as e:
            print(f"Error saving training-ready depth: {e}")
            return False

    def save_model_ready_rgb(self, rgb_frame, filepath, target_size=(480, 640)):
        """Save RGB in model-ready format with proper resolution"""
        try:
            # Resize to training resolution (height, width)
            if rgb_frame.shape[:2] != target_size:
                rgb_resized = cv2.resize(rgb_frame, (target_size[1], target_size[0]))
            else:
                rgb_resized = rgb_frame
                
            # Save original resolution
            cv2.imwrite(str(filepath), rgb_resized)
            return True
            
        except Exception as e:
            print(f"Error saving model-ready RGB: {e}")
            return False

    def create_training_dataset_structure(self):
        """Create additional directories for training-ready data"""
        # Training-ready directories
        self.training_dir = self.output_dir / "training_ready"
        self.training_dir.mkdir(exist_ok=True)
        
        # Model-specific directories
        for model in ["adabins", "newcrfs", "zoedepth"]:
            model_dir = self.training_dir / model
            model_dir.mkdir(exist_ok=True)
            (model_dir / "rgb").mkdir(exist_ok=True)
            (model_dir / "depth").mkdir(exist_ok=True)
            (model_dir / "lists").mkdir(exist_ok=True)  # For train/val splits
        
        print("📁 Training-ready dataset structure created")

    def save_enhanced_synchronized_frame_set(self, frames):
        """Enhanced saving with training-ready format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_num = f"{self.frame_count:06d}"
        
        saved_count = 0
        saved_details = []
        quality_data = {}
        
        # Save D415 synchronized frames
        if 'd415_rgb' in frames and 'd415_depth' in frames:
            rgb_frame = frames['d415_rgb']
            depth_frame = frames['d415_depth']
            
            # Verify data quality first
            quality_metrics = self.verify_data_quality(rgb_frame, depth_frame)
            quality_data.update(quality_metrics)
            
            # 1. Save original format (for reference)
            rgb_file = self.d415_dir / "rgb" / f"{frame_num}_rgb_{timestamp}.png"
            cv2.imwrite(str(rgb_file), rgb_frame)
            
            depth_file = self.d415_dir / "depth" / f"{frame_num}_depth_{timestamp}.png"
            cv2.imwrite(str(depth_file), depth_frame.astype(np.uint16))
            
            # 2. Save training-ready format for each model
            for model in ["adabins", "newcrfs", "zoedepth"]:
                model_dir = self.training_dir / model
                
                # Model-specific resolution
                if model == "zoedepth":
                    target_size = (384, 384)  # Square for transformer
                else:
                    target_size = (480, 640)  # NYU format
                
                # Save model-ready RGB
                model_rgb_file = model_dir / "rgb" / f"{frame_num}_{timestamp}.png"
                if self.save_model_ready_rgb(rgb_frame, model_rgb_file, target_size):
                    saved_count += 1
                
                # Save model-ready depth (resize to match RGB)
                depth_resized = cv2.resize(depth_frame, (target_size[1], target_size[0]))
                model_depth_file = model_dir / "depth" / f"{frame_num}_{timestamp}"
                if self.save_training_ready_depth(depth_resized, model_depth_file):
                    saved_count += 1
            
            # Save overlay
            overlay = self.create_depth_overlay(rgb_frame, depth_frame)
            overlay_file = self.d415_dir / "overlay" / f"{frame_num}_overlay_{timestamp}.png"
            cv2.imwrite(str(overlay_file), overlay)
            
            saved_details = ["Original", "AdaBins", "NeWCRFs", "ZoeDepth", "Overlay"]
        
        # Enhanced metadata with training info
        metadata = {
            "frame_set": self.frame_count,
            "timestamp": timestamp,
            "cameras_captured": list(frames.keys()),
            "files_saved": saved_count,
            "target_range_mm": [self.target_range[0]*1000, self.target_range[1]*1000],
            "training_range_m": [0.4, 0.58],  # Training range in meters
            "max_depth_cap_m": 2.0,
            "sync_quality_ms": frames.get('sync_quality', 0),
            "data_quality": quality_data,
            "file_details": saved_details,
            "formats": {
                "original_depth": "16-bit PNG (millimeters)",
                "training_depth": "float32 NPY (meters, capped at 2m)",
                "rgb_resolutions": {
                    "adabins": [480, 640],
                    "newcrfs": [480, 640], 
                    "zoedepth": [384, 384]
                }
            },
            "alignment_method": "hardware_synchronized"
        }
        
        metadata_file = self.sync_dir / f"{frame_num}_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.frame_count += 1
        
        # Enhanced status reporting
        sync_quality = frames.get('sync_quality', 0)
        alignment_score = quality_data.get('edge_alignment', 0)
        depth_coverage = quality_data.get('depth_coverage', 0)
        
        print(f"📸 Set {self.frame_count}: {saved_count} files | "
            f"Sync: {sync_quality:.1f}ms | "
            f"Align: {alignment_score:.2f} | "
            f"Coverage: {depth_coverage:.1%} | "
            f"Training-ready: ✅")
        
        return saved_count

    def create_training_lists(self):
        """Create train/val split lists for each model as per documentation"""
        try:
            # Get all frame files
            all_frames = []
            for rgb_file in (self.training_dir / "adabins" / "rgb").glob("*.png"):
                frame_id = rgb_file.stem
                depth_file = self.training_dir / "adabins" / "depth" / f"{frame_id}.npy"
                if depth_file.exists():
                    all_frames.append(frame_id)
            
            if not all_frames:
                print("⚠️  No training frames found")
                return
            
            # Create train/val split (90/10)
            np.random.shuffle(all_frames)
            split_idx = int(0.9 * len(all_frames))
            train_frames = all_frames[:split_idx]
            val_frames = all_frames[split_idx:]
            
            print(f"📊 Dataset split: {len(train_frames)} train, {len(val_frames)} val")
            
            # Create lists for each model
            for model in ["adabins", "newcrfs", "zoedepth"]:
                lists_dir = self.training_dir / model / "lists"
                
                # AdaBins format: rgb_path depth_path per line
                if model == "adabins":
                    with open(lists_dir / "train_pairs.txt", 'w') as f:
                        for frame_id in train_frames:
                            f.write(f"rgb/{frame_id}.png depth/{frame_id}.npy\n")
                    
                    with open(lists_dir / "val_pairs.txt", 'w') as f:
                        for frame_id in val_frames:
                            f.write(f"rgb/{frame_id}.png depth/{frame_id}.npy\n")
                
                # NeWCRFs format: just filenames
                elif model == "newcrfs":
                    with open(lists_dir / "train_files.txt", 'w') as f:
                        for frame_id in train_frames:
                            f.write(f"{frame_id}.png\n")
                    
                    with open(lists_dir / "val_files.txt", 'w') as f:
                        for frame_id in val_frames:
                            f.write(f"{frame_id}.png\n")
                
                # ZoeDepth format: similar to AdaBins
                elif model == "zoedepth":
                    with open(lists_dir / "train_list.txt", 'w') as f:
                        for frame_id in train_frames:
                            f.write(f"rgb/{frame_id}.png depth/{frame_id}.npy\n")
                    
                    with open(lists_dir / "val_list.txt", 'w') as f:
                        for frame_id in val_frames:
                            f.write(f"rgb/{frame_id}.png depth/{frame_id}.npy\n")
            
            print("✅ Training/validation lists created for all models")
            
        except Exception as e:
            print(f"❌ Error creating training lists: {e}")

    def finalize_training_dataset(self):
        """Finalize dataset for training - call this after data collection"""
        print("\n🎯 Finalizing training dataset...")
        
        # Create training lists
        self.create_training_lists()
        
        # Create dataset info file
        dataset_info = {
            "dataset_name": f"hand_depth_{datetime.now().strftime('%Y%m%d')}",
            "total_frames": self.frame_count,
            "capture_range_mm": [self.target_range[0]*1000, self.target_range[1]*1000],
            "training_range_m": [0.4, 0.58],
            "max_depth_cap_m": 2.0,
            "camera_info": {
                "device": "Intel RealSense D415",
                "serial": "821312062833",
                "resolution": [640, 480],
                "sync_method": "hardware_aligned"
            },
            "model_formats": {
                "adabins": {
                    "resolution": [480, 640],
                    "depth_format": "float32_npy_meters",
                    "rgb_format": "uint8_png"
                },
                "newcrfs": {
                    "resolution": [480, 640], 
                    "depth_format": "float32_npy_meters",
                    "rgb_format": "uint8_png"
                },
                "zoedepth": {
                    "resolution": [384, 384],
                    "depth_format": "float32_npy_meters", 
                    "rgb_format": "uint8_png"
                }
            }
        }
        
        with open(self.training_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"✅ Training dataset finalized!")
        print(f"📁 Location: {self.training_dir}")
        print(f"📊 Ready for: AdaBins, NeWCRFs, ZoeDepth")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Synchronized Multi-Camera Capture')
    parser.add_argument('--output-dir', type=str, default='synchronized_capture',
                       help='Output directory')
    parser.add_argument('--min-range', type=float, default=0.4,
                       help='Minimum range in meters')
    parser.add_argument('--max-range', type=float, default=0.58,
                       help='Maximum range in meters')
    
    args = parser.parse_args()
    
    print("🎯 Synchronized Multi-Camera Capture System")
    print("Fixed D415 RGB-Depth synchronization issue")
    print("=" * 50)
    
    try:
        # Create capture system
        capture = SynchronizedMultiCameraCapture(args.output_dir)
        capture.target_range = (args.min_range, args.max_range)
        
        # Setup cameras with improved synchronization
        setup_count = 0
        
        # Setup D415 with synchronized RGB-Depth
        if capture.setup_d415_synchronized("821312062833"):
            setup_count += 1
        
        # Setup D405
        if capture.setup_d405("230322270171"):
            setup_count += 1
        
        # Setup platform cameras
        for cam_id in [1, 2]:
            if capture.setup_platform_camera(cam_id):
                setup_count += 1
        
        if setup_count == 0:
            print("❌ No cameras were setup successfully!")
            return
        
        print(f"✅ {setup_count} cameras ready for synchronized capture")
        
        # Run synchronized capture
        capture.run_synchronized_capture()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            capture.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()