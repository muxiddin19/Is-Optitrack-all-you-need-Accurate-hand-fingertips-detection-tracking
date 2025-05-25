#!/usr/bin/env python3
"""
4-Camera Simultaneous Video Capture System
Optimized for 400-580mm range with continuous recording from all cameras
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
import queue

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

class MultiCameraVideoCapture:
    """Simultaneous video capture from 4 cameras optimized for close range"""
    
    def __init__(self, output_dir="multi_camera_data", calibration_file=None):
        self.output_dir = Path(output_dir)
        self.calibration_file = calibration_file
        self.calibration_data = None
        
        # Camera objects
        self.d415_pipeline = None
        self.d405_pipeline = None
        self.platform_cameras = {}
        
        # Recording parameters
        self.recording = False
        self.target_range = (0.4, 0.58)  # 400-580mm optimized range
        self.recording_fps = 30
        
        # Threading for simultaneous capture
        self.capture_threads = {}
        self.frame_queues = {}
        self.capture_running = False
        
        # Video writers
        self.video_writers = {}
        self.frame_counters = {}
        
        # Setup directories and load calibration
        self.setup_directories()
        if calibration_file:
            self.load_calibration()
    
    def setup_directories(self):
        """Create comprehensive directory structure"""
        print(f"📁 Setting up multi-camera directories in {self.output_dir}")
        
        # Main camera directories
        self.d415_dir = self.output_dir / "d415"
        self.d405_dir = self.output_dir / "d405" 
        self.platform1_dir = self.output_dir / "platform_cam_1"
        self.platform2_dir = self.output_dir / "platform_cam_2"
        
        # Create subdirectories for each camera
        for camera_dir in [self.d415_dir, self.d405_dir, self.platform1_dir, self.platform2_dir]:
            (camera_dir / "rgb").mkdir(parents=True, exist_ok=True)
            (camera_dir / "depth").mkdir(parents=True, exist_ok=True)
            (camera_dir / "aligned").mkdir(parents=True, exist_ok=True)
            (camera_dir / "video").mkdir(parents=True, exist_ok=True)
            (camera_dir / "metadata").mkdir(parents=True, exist_ok=True)
        
        # Synchronized data directory
        self.sync_dir = self.output_dir / "synchronized"
        self.sync_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Multi-camera directory structure created")
    
    def load_calibration(self):
        """Load calibration data"""
        try:
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            print(f"✅ Loaded calibration from {self.calibration_file}")
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
    
    def setup_d415_close_range(self, serial_number):
        """Setup D415 optimized for 400-580mm range"""
        print(f"🔧 Setting up D415 (SN: {serial_number}) for close range (400-580mm)...")
        
        try:
            self.d415_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial_number)
            
            # Configure streams
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.recording_fps)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.recording_fps)
            
            # Start pipeline
            profile = self.d415_pipeline.start(config)
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            
            # Optimize for close range
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
            
            # Set minimum distance for close range
            if depth_sensor.supports(rs.option.min_distance):
                depth_sensor.set_option(rs.option.min_distance, 100)  # 10cm minimum
            
            # Maximize laser power for close range accuracy
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, 360)
            
            # Get parameters
            self.d415_depth_scale = depth_sensor.get_depth_scale()
            color_profile = profile.get_stream(rs.stream.color)
            self.d415_color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            # Setup frame queue
            self.frame_queues['d415'] = queue.Queue(maxsize=60)
            self.frame_counters['d415'] = 0
            
            print(f"  ✅ D415 ready for close-range capture")
            print(f"      Optimized for: {self.target_range[0]*1000:.0f}-{self.target_range[1]*1000:.0f}mm")
            return True
            
        except Exception as e:
            print(f"  ❌ D415 setup failed: {e}")
            return False
    
    def setup_d405_close_range(self, serial_number):
        """Setup D405 optimized for close range"""
        print(f"🔧 Setting up D405 (SN: {serial_number}) for close range...")
        
        try:
            self.d405_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial_number)
            
            # D405 streams
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.recording_fps)
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, self.recording_fps)
            
            # Start pipeline
            profile = self.d405_pipeline.start(config)
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            
            # Optimize for close range
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
                
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, 360)
            
            self.d405_depth_scale = depth_sensor.get_depth_scale()
            
            # Setup frame queue
            self.frame_queues['d405'] = queue.Queue(maxsize=60)
            self.frame_counters['d405'] = 0
            
            print(f"  ✅ D405 ready for close-range depth capture")
            return True
            
        except Exception as e:
            print(f"  ❌ D405 setup failed: {e}")
            return False
    
    def setup_platform_camera(self, camera_id, camera_name):
        """Setup platform camera with optimization"""
        print(f"🔧 Setting up {camera_name} (ID: {camera_id})...")
        
        try:
            # Test different backends for stability
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
            cap = None
            
            for backend in backends:
                try:
                    test_cap = cv2.VideoCapture(camera_id, backend)
                    if test_cap.isOpened():
                        test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        test_cap.set(cv2.CAP_PROP_FPS, self.recording_fps)
                        
                        ret, frame = test_cap.read()
                        if ret and frame is not None:
                            cap = test_cap
                            break
                        test_cap.release()
                except:
                    pass
            
            if cap is None:
                print(f"  ❌ {camera_name} not accessible")
                return False
            
            self.platform_cameras[camera_name] = cap
            
            # Setup frame queue
            self.frame_queues[camera_name] = queue.Queue(maxsize=60)
            self.frame_counters[camera_name] = 0
            
            print(f"  ✅ {camera_name} ready")
            return True
            
        except Exception as e:
            print(f"  ❌ {camera_name} setup failed: {e}")
            return False
    
    def d415_capture_thread(self):
        """D415 capture thread with RGB and depth"""
        print("🎬 Starting D415 capture thread...")
        
        align_to_color = rs.align(rs.stream.color)
        
        while self.capture_running:
            try:
                frames = self.d415_pipeline.wait_for_frames()
                aligned_frames = align_to_color.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    timestamp = time.time()
                    
                    # Convert to numpy
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    # Create aligned visualization
                    aligned_vis = self.create_depth_overlay(color_image, depth_image)
                    
                    frame_data = {
                        'timestamp': timestamp,
                        'rgb': color_image,
                        'depth': depth_image,
                        'aligned': aligned_vis,
                        'camera': 'd415'
                    }
                    
                    # Add to queue if not full
                    if not self.frame_queues['d415'].full():
                        self.frame_queues['d415'].put(frame_data)
                    
            except Exception as e:
                if self.capture_running:
                    print(f"D415 capture error: {e}")
                time.sleep(0.01)
    
    def d405_capture_thread(self):
        """D405 capture thread with depth and IR"""
        print("🎬 Starting D405 capture thread...")
        
        while self.capture_running:
            try:
                frames = self.d405_pipeline.wait_for_frames()
                
                depth_frame = frames.get_depth_frame()
                ir_frame = frames.get_infrared_frame(1)
                
                if depth_frame and ir_frame:
                    timestamp = time.time()
                    
                    # Convert to numpy
                    depth_image = np.asanyarray(depth_frame.get_data())
                    ir_image = np.asanyarray(ir_frame.get_data())
                    
                    # Convert IR to 3-channel for consistency
                    ir_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
                    
                    # Create depth visualization
                    depth_vis = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_PLASMA)
                    
                    frame_data = {
                        'timestamp': timestamp,
                        'rgb': ir_bgr,  # Use IR as RGB substitute
                        'depth': depth_image,
                        'aligned': depth_vis,
                        'camera': 'd405'
                    }
                    
                    if not self.frame_queues['d405'].full():
                        self.frame_queues['d405'].put(frame_data)
                    
            except Exception as e:
                if self.capture_running:
                    print(f"D405 capture error: {e}")
                time.sleep(0.01)
    
    def platform_capture_thread(self, camera_name):
        """Platform camera capture thread"""
        print(f"🎬 Starting {camera_name} capture thread...")
        
        cap = self.platform_cameras[camera_name]
        
        while self.capture_running:
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    timestamp = time.time()
                    
                    frame_data = {
                        'timestamp': timestamp,
                        'rgb': frame,
                        'depth': None,  # No depth for platform cameras
                        'aligned': frame,  # Just RGB
                        'camera': camera_name
                    }
                    
                    if not self.frame_queues[camera_name].full():
                        self.frame_queues[camera_name].put(frame_data)
                
            except Exception as e:
                if self.capture_running:
                    print(f"{camera_name} capture error: {e}")
                time.sleep(0.01)
    
    def create_depth_overlay(self, rgb_frame, depth_frame):
        """Create RGB-depth overlay"""
        # Apply range filter for target distance
        min_depth_mm = self.target_range[0] * 1000
        max_depth_mm = self.target_range[1] * 1000
        
        # Create mask for target range
        depth_mask = (depth_frame >= min_depth_mm) & (depth_frame <= max_depth_mm)
        
        # Normalize depth within target range
        depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)
        if np.any(depth_mask):
            depth_in_range = depth_frame[depth_mask]
            depth_normalized[depth_mask] = cv2.normalize(
                depth_in_range, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(rgb_frame, 0.7, depth_colored, 0.3, 0)
        
        # Add range information
        valid_depth = depth_frame[depth_frame > 0]
        if len(valid_depth) > 0:
            mean_depth = np.mean(valid_depth)
            in_range_count = np.sum(depth_mask)
            total_pixels = np.sum(depth_frame > 0)
            range_percentage = (in_range_count / total_pixels * 100) if total_pixels > 0 else 0
            
            # Color code based on range compliance
            if self.target_range[0] * 1000 <= mean_depth <= self.target_range[1] * 1000:
                text_color = (0, 255, 0)  # Green - optimal
                status = "OPTIMAL"
            else:
                text_color = (0, 255, 255)  # Yellow - suboptimal
                status = "ADJUST DISTANCE"
            
            cv2.putText(overlay, f"{status}: {mean_depth:.0f}mm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(overlay, f"In Range: {range_percentage:.1f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        return overlay
    
    def setup_video_writers(self):
        """Setup video writers for continuous recording"""
        if not self.recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Setup video writers for each camera
        for camera_name in self.frame_queues.keys():
            camera_dir = self.output_dir / camera_name / "video"
            
            # RGB video
            rgb_filename = camera_dir / f"{camera_name}_rgb_{timestamp}.mp4"
            rgb_writer = cv2.VideoWriter(str(rgb_filename), fourcc, self.recording_fps, (640, 480))
            
            # Aligned video (RGB+depth overlay or just RGB for platform cams)
            aligned_filename = camera_dir / f"{camera_name}_aligned_{timestamp}.mp4"
            aligned_writer = cv2.VideoWriter(str(aligned_filename), fourcc, self.recording_fps, (640, 480))
            
            self.video_writers[camera_name] = {
                'rgb': rgb_writer,
                'aligned': aligned_writer,
                'rgb_file': rgb_filename,
                'aligned_file': aligned_filename
            }
        
        print(f"✅ Video writers setup for {len(self.video_writers)} cameras")
    
    def process_frame_queues(self):
        """Process frames from all cameras and save to video"""
        while self.capture_running or any(not q.empty() for q in self.frame_queues.values()):
            try:
                # Process frames from each camera
                for camera_name, frame_queue in self.frame_queues.items():
                    if not frame_queue.empty():
                        frame_data = frame_queue.get_nowait()
                        
                        # Save to video if recording
                        if self.recording and camera_name in self.video_writers:
                            writers = self.video_writers[camera_name]
                            
                            # Write RGB frame
                            writers['rgb'].write(frame_data['rgb'])
                            
                            # Write aligned frame
                            writers['aligned'].write(frame_data['aligned'])
                            
                            # Update counter
                            self.frame_counters[camera_name] += 1
                        
                        # Save individual frames periodically
                        if self.frame_counters[camera_name] % 30 == 0:  # Every 30 frames
                            self.save_frame_set(frame_data, camera_name, self.frame_counters[camera_name])
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Frame processing error: {e}")
    
    def save_frame_set(self, frame_data, camera_name, frame_number):
        """Save individual frame set for detailed analysis"""
        try:
            camera_dir = self.output_dir / camera_name
            frame_str = f"{frame_number:06d}"
            
            # Save RGB frame
            rgb_file = camera_dir / "rgb" / f"{frame_str}_rgb.png"
            cv2.imwrite(str(rgb_file), frame_data['rgb'])
            
            # Save depth frame if available
            if frame_data['depth'] is not None:
                depth_file = camera_dir / "depth" / f"{frame_str}_depth.png"
                cv2.imwrite(str(depth_file), frame_data['depth'])
            
            # Save aligned frame
            aligned_file = camera_dir / "aligned" / f"{frame_str}_aligned.png"
            cv2.imwrite(str(aligned_file), frame_data['aligned'])
            
            # Save metadata
            metadata = {
                "frame_number": frame_number,
                "timestamp": frame_data['timestamp'],
                "camera": camera_name,
                "files": {
                    "rgb": f"{frame_str}_rgb.png",
                    "depth": f"{frame_str}_depth.png" if frame_data['depth'] is not None else None,
                    "aligned": f"{frame_str}_aligned.png"
                }
            }
            
            metadata_file = camera_dir / "metadata" / f"{frame_str}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save frame set {frame_number} for {camera_name}: {e}")
    
    def start_capture_threads(self):
        """Start all capture threads"""
        print("🚀 Starting simultaneous capture from all cameras...")
        
        self.capture_running = True
        
        # Start camera capture threads
        if self.d415_pipeline:
            thread = threading.Thread(target=self.d415_capture_thread, daemon=True)
            thread.start()
            self.capture_threads['d415'] = thread
        
        if self.d405_pipeline:
            thread = threading.Thread(target=self.d405_capture_thread, daemon=True)
            thread.start()
            self.capture_threads['d405'] = thread
        
        for camera_name in self.platform_cameras:
            thread = threading.Thread(target=self.platform_capture_thread, 
                                    args=(camera_name,), daemon=True)
            thread.start()
            self.capture_threads[camera_name] = thread
        
        # Start frame processing thread
        processing_thread = threading.Thread(target=self.process_frame_queues, daemon=True)
        processing_thread.start()
        self.capture_threads['processor'] = processing_thread
        
        time.sleep(2.0)  # Allow threads to initialize
        print(f"✅ {len(self.capture_threads)} threads started")
    
    def stop_capture_threads(self):
        """Stop all capture threads"""
        print("🛑 Stopping capture threads...")
        self.capture_running = False
        
        # Wait for threads to finish
        for thread in self.capture_threads.values():
            thread.join(timeout=2.0)
        
        self.capture_threads.clear()
    
    def create_multi_camera_display(self):
        """Create real-time display from all cameras"""
        displays = []
        
        # Get latest frames from each camera
        for camera_name, frame_queue in self.frame_queues.items():
            try:
                # Get the most recent frame without blocking
                latest_frame = None
                while not frame_queue.empty():
                    try:
                        latest_frame = frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                if latest_frame:
                    display_frame = latest_frame['aligned'].copy()
                    
                    # Add camera info
                    cv2.putText(display_frame, camera_name.upper(), (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Frames: {self.frame_counters[camera_name]}", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    displays.append(cv2.resize(display_frame, (320, 240)))
                    
                    # Put frame back for processing
                    if not frame_queue.full():
                        frame_queue.put(latest_frame)
            
            except Exception as e:
                # Create blank frame if error
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, f"{camera_name}: ERROR", (50, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                displays.append(blank)
        
        # Arrange in 2x2 grid
        if len(displays) >= 4:
            top_row = np.hstack(displays[:2])
            bottom_row = np.hstack(displays[2:4])
            combined = np.vstack([top_row, bottom_row])
        elif len(displays) == 3:
            top_row = np.hstack(displays[:2])
            bottom_row = np.hstack([displays[2], np.zeros((240, 320, 3), dtype=np.uint8)])
            combined = np.vstack([top_row, bottom_row])
        elif len(displays) == 2:
            combined = np.hstack(displays)
        elif len(displays) == 1:
            combined = displays[0]
        else:
            combined = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(combined, "No camera feeds", (200, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined
    
    def run_multi_camera_capture(self):
        """Run multi-camera capture session"""
        print(f"\n🎬 Starting 4-Camera Video Capture System")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Optimized range: {self.target_range[0]*1000:.0f}-{self.target_range[1]*1000:.0f}mm")
        print(f"🎮 Controls:")
        print(f"  • SPACE - Start/Stop video recording")
        print(f"  • 's' - Save current frame set from all cameras")
        print(f"  • 'i' - Show capture statistics")
        print(f"  • 'q' - Quit")
        
        recording_start_time = 0
        
        try:
            while True:
                # Create multi-camera display
                display = self.create_multi_camera_display()
                
                # Add system status
                total_frames = sum(self.frame_counters.values())
                active_cameras = len([q for q in self.frame_queues.values() if not q.empty()])
                
                if self.recording:
                    recording_time = time.time() - recording_start_time
                    cv2.putText(display, f"🔴 RECORDING: {recording_time:.1f}s", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(display, "Press SPACE to start video recording", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(display, f"Total Frames: {total_frames} | Active: {active_cameras}/4", 
                           (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('4-Camera Video Capture System', display)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if not self.recording:
                        # Start recording
                        self.recording = True
                        recording_start_time = time.time()
                        self.setup_video_writers()
                        print("🔴 Video recording started for all cameras")
                    else:
                        # Stop recording
                        self.recording = False
                        self.close_video_writers()
                        print("⏹️  Video recording stopped")
                elif key == ord('s'):
                    self.save_current_frames()
                elif key == ord('i'):
                    self.print_capture_statistics()
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        finally:
            if self.recording:
                self.close_video_writers()
            cv2.destroyAllWindows()
        
        print(f"\n✅ Multi-camera capture complete!")
        self.print_final_summary()
    
    def close_video_writers(self):
        """Close all video writers"""
        for camera_name, writers in self.video_writers.items():
            try:
                writers['rgb'].release()
                writers['aligned'].release()
                print(f"📹 Saved videos for {camera_name}:")
                print(f"  RGB: {writers['rgb_file']}")
                print(f"  Aligned: {writers['aligned_file']}")
            except Exception as e:
                print(f"Error closing video writer for {camera_name}: {e}")
        
        self.video_writers.clear()
    
    def save_current_frames(self):
        """Save current frame set from all cameras"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_count = 0
        
        for camera_name, frame_queue in self.frame_queues.items():
            try:
                if not frame_queue.empty():
                    # Get latest frame
                    latest_frame = None
                    while not frame_queue.empty():
                        try:
                            latest_frame = frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    if latest_frame:
                        # Save frame set
                        self.save_frame_set(latest_frame, camera_name, int(timestamp))
                        saved_count += 1
                        
                        # Put frame back
                        if not frame_queue.full():
                            frame_queue.put(latest_frame)
            
            except Exception as e:
                print(f"Error saving frame for {camera_name}: {e}")
        
        print(f"📸 Saved current frame set from {saved_count} cameras")
    
    def print_capture_statistics(self):
        """Print detailed capture statistics"""
        print(f"\n{'='*60}")
        print("4-CAMERA CAPTURE STATISTICS")
        print('='*60)
        
        total_frames = sum(self.frame_counters.values())
        print(f"📊 Total frames captured: {total_frames}")
        print(f"🎯 Target range: {self.target_range[0]*1000:.0f}-{self.target_range[1]*1000:.0f}mm")
        print(f"📷 Recording FPS: {self.recording_fps}")
        print(f"🔴 Currently recording: {'YES' if self.recording else 'NO'}")
        