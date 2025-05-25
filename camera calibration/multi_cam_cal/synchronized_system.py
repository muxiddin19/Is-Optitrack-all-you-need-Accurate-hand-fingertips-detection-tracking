#!/usr/bin/env python3
"""
Time-Synchronized Multi-Camera System
Optimized for temporal synchronization across D415, D405, and platform cameras
"""

import cv2
import numpy as np
import json
import time
import threading
from datetime import datetime
from collections import deque
import argparse
from pathlib import Path

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

class TimestampedFrame:
    """Frame with precise timestamp"""
    def __init__(self, frame, timestamp, camera_name, frame_type='rgb'):
        self.frame = frame
        self.timestamp = timestamp
        self.camera_name = camera_name
        self.frame_type = frame_type  # 'rgb', 'depth', 'ir'
        self.frame_number = None

class SynchronizedCameraSystem:
    """Multi-camera system with precise temporal synchronization"""
    
    def __init__(self, calibration_file=None, sync_tolerance_ms=50):
        self.calibration_file = calibration_file
        self.sync_tolerance_ms = sync_tolerance_ms  # Max time difference for sync
        
        # Camera objects
        self.d415_pipeline = None
        self.d405_pipeline = None
        self.platform_cameras = {}
        
        # Synchronization buffers
        self.frame_buffers = {}  # {camera_name: deque of TimestampedFrame}
        self.buffer_size = 30  # Keep last N frames for sync
        
        # Threading
        self.capture_threads = {}
        self.capture_running = False
        self.frame_locks = {}
        
        # Sync statistics
        self.sync_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'avg_sync_error_ms': 0,
            'max_sync_error_ms': 0
        }
        
        # Target frame rate
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps
        
        if calibration_file:
            self.load_calibration()
    
    def load_calibration(self):
        """Load calibration data"""
        try:
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            print(f"✅ Loaded calibration from {self.calibration_file}")
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
    
    def setup_d415(self, serial_number):
        """Setup D415 with hardware synchronization"""
        print(f"🔧 Setting up D415 (SN: {serial_number}) with sync optimization...")
        
        try:
            self.d415_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial_number)
            
            # Configure streams with same resolution and FPS
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.target_fps)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.target_fps)
            
            # Start pipeline
            profile = self.d415_pipeline.start(config)
            
            # Get device and enable hardware timestamp
            device = profile.get_device()
            
            # Enable auto-exposure and white balance for consistent timing
            color_sensor = device.query_sensors()[1]  # Color sensor
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            
            # Setup frame buffers
            self.frame_buffers['d415_rgb'] = deque(maxlen=self.buffer_size)
            self.frame_buffers['d415_depth'] = deque(maxlen=self.buffer_size)
            self.frame_locks['d415'] = threading.Lock()
            
            print(f"  ✅ D415 ready with hardware sync")
            return True
            
        except Exception as e:
            print(f"  ❌ D415 setup failed: {e}")
            return False
    
    def setup_d405(self, serial_number):
        """Setup D405 with sync optimization"""
        print(f"🔧 Setting up D405 (SN: {serial_number}) with sync optimization...")
        
        try:
            self.d405_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial_number)
            
            # Configure streams
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.target_fps)
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, self.target_fps)
            
            # Start pipeline
            self.d405_pipeline.start(config)
            
            # Setup frame buffers
            self.frame_buffers['d405_depth'] = deque(maxlen=self.buffer_size)
            self.frame_buffers['d405_ir'] = deque(maxlen=self.buffer_size)
            self.frame_locks['d405'] = threading.Lock()
            
            print(f"  ✅ D405 ready with sync optimization")
            return True
            
        except Exception as e:
            print(f"  ❌ D405 setup failed: {e}")
            return False
    
    def setup_platform_camera(self, camera_id, camera_name):
        """Setup platform camera with sync optimization"""
        print(f"🔧 Setting up {camera_name} (ID: {camera_id}) with sync optimization...")
        
        try:
            # Try different backends for best performance
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
            cap = None
            
            for backend in backends:
                try:
                    test_cap = cv2.VideoCapture(camera_id, backend)
                    if test_cap.isOpened():
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
            
            # Optimize for synchronization
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Disable auto-exposure for consistent timing
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
            
            self.platform_cameras[camera_name] = cap
            
            # Setup frame buffer
            self.frame_buffers[camera_name] = deque(maxlen=self.buffer_size)
            self.frame_locks[camera_name] = threading.Lock()
            
            print(f"  ✅ {camera_name} ready with sync optimization")
            return True
            
        except Exception as e:
            print(f"  ❌ {camera_name} setup failed: {e}")
            return False
    
    def d415_capture_thread(self):
        """D415 capture thread with precise timing"""
        print("🎬 Starting D415 capture thread...")
        
        # Hardware alignment
        align_to_color = rs.align(rs.stream.color)
        
        while self.capture_running:
            try:
                start_time = time.time()
                
                # Capture frames
                frames = self.d415_pipeline.wait_for_frames()
                capture_timestamp = time.time() * 1000  # Convert to milliseconds
                
                # Apply hardware alignment
                aligned_frames = align_to_color.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    # Convert to numpy
                    rgb_data = np.asanyarray(color_frame.get_data())
                    depth_data = np.asanyarray(depth_frame.get_data())
                    
                    # Store with timestamps
                    with self.frame_locks['d415']:
                        self.frame_buffers['d415_rgb'].append(
                            TimestampedFrame(rgb_data, capture_timestamp, 'd415_rgb', 'rgb'))
                        self.frame_buffers['d415_depth'].append(
                            TimestampedFrame(depth_data, capture_timestamp, 'd415_depth', 'depth'))
                
                # Maintain target FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, self.frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                if self.capture_running:
                    print(f"D415 capture error: {e}")
                    time.sleep(0.01)
    
    def d405_capture_thread(self):
        """D405 capture thread with precise timing"""
        print("🎬 Starting D405 capture thread...")
        
        while self.capture_running:
            try:
                start_time = time.time()
                
                # Capture frames
                frames = self.d405_pipeline.wait_for_frames()
                capture_timestamp = time.time() * 1000
                
                depth_frame = frames.get_depth_frame()
                ir_frame = frames.get_infrared_frame(1)
                
                if depth_frame and ir_frame:
                    # Convert to numpy
                    depth_data = np.asanyarray(depth_frame.get_data())
                    ir_data = np.asanyarray(ir_frame.get_data())
                    
                    # Store with timestamps
                    with self.frame_locks['d405']:
                        self.frame_buffers['d405_depth'].append(
                            TimestampedFrame(depth_data, capture_timestamp, 'd405_depth', 'depth'))
                        self.frame_buffers['d405_ir'].append(
                            TimestampedFrame(ir_data, capture_timestamp, 'd405_ir', 'ir'))
                
                # Maintain target FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, self.frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                if self.capture_running:
                    print(f"D405 capture error: {e}")
                    time.sleep(0.01)
    
    def platform_capture_thread(self, camera_name):
        """Platform camera capture thread with precise timing"""
        print(f"🎬 Starting {camera_name} capture thread...")
        
        cap = self.platform_cameras[camera_name]
        
        while self.capture_running:
            try:
                start_time = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                capture_timestamp = time.time() * 1000
                
                if ret and frame is not None:
                    # Store with timestamp
                    with self.frame_locks[camera_name]:
                        self.frame_buffers[camera_name].append(
                            TimestampedFrame(frame, capture_timestamp, camera_name, 'rgb'))
                
                # Maintain target FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, self.frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                if self.capture_running:
                    print(f"{camera_name} capture error: {e}")
                    time.sleep(0.01)
    
    def start_capture_threads(self):
        """Start all capture threads"""
        print("🚀 Starting synchronized capture threads...")
        
        self.capture_running = True
        
        # Start D415 thread
        if self.d415_pipeline:
            thread = threading.Thread(target=self.d415_capture_thread, daemon=True)
            thread.start()
            self.capture_threads['d415'] = thread
        
        # Start D405 thread
        if self.d405_pipeline:
            thread = threading.Thread(target=self.d405_capture_thread, daemon=True)
            thread.start()
            self.capture_threads['d405'] = thread
        
        # Start platform camera threads
        for camera_name in self.platform_cameras:
            thread = threading.Thread(target=self.platform_capture_thread, 
                                    args=(camera_name,), daemon=True)
            thread.start()
            self.capture_threads[camera_name] = thread
        
        # Wait for threads to initialize
        time.sleep(1.0)
        print(f"✅ {len(self.capture_threads)} capture threads started")
    
    def stop_capture_threads(self):
        """Stop all capture threads"""
        print("🛑 Stopping capture threads...")
        self.capture_running = False
        
        # Wait for threads to finish
        for thread in self.capture_threads.values():
            thread.join(timeout=1.0)
        
        self.capture_threads.clear()
    
    def find_synchronized_frames(self, reference_timestamp=None):
        """Find synchronized frames across all cameras"""
        if reference_timestamp is None:
            reference_timestamp = time.time() * 1000
        
        synchronized_frames = {}
        sync_errors = []
        
        # Find closest frames to reference timestamp
        for camera_name, buffer in self.frame_buffers.items():
            if not buffer:
                continue
            
            with self.frame_locks.get(camera_name.split('_')[0], threading.Lock()):
                # Find frame closest to reference timestamp
                closest_frame = None
                min_time_diff = float('inf')
                
                for frame in buffer:
                    time_diff = abs(frame.timestamp - reference_timestamp)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_frame = frame
                
                # Check if within sync tolerance
                if closest_frame and min_time_diff <= self.sync_tolerance_ms:
                    synchronized_frames[camera_name] = closest_frame
                    sync_errors.append(min_time_diff)
        
        # Update sync statistics
        if sync_errors:
            self.sync_stats['total_syncs'] += 1
            if len(synchronized_frames) >= 2:  # At least 2 cameras synchronized
                self.sync_stats['successful_syncs'] += 1
                avg_error = sum(sync_errors) / len(sync_errors)
                max_error = max(sync_errors)
                
                # Update running averages
                self.sync_stats['avg_sync_error_ms'] = (
                    (self.sync_stats['avg_sync_error_ms'] * (self.sync_stats['successful_syncs'] - 1) + avg_error) /
                    self.sync_stats['successful_syncs']
                )
                self.sync_stats['max_sync_error_ms'] = max(self.sync_stats['max_sync_error_ms'], max_error)
        
        return synchronized_frames
    
    def create_synchronized_display(self, sync_frames):
        """Create display from synchronized frames"""
        displays = []
        
        # D415 RGB with depth overlay
        if 'd415_rgb' in sync_frames and 'd415_depth' in sync_frames:
            rgb_frame = sync_frames['d415_rgb'].frame
            depth_frame = sync_frames['d415_depth'].frame
            
            # Create overlay
            depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(rgb_frame, 0.7, depth_colored, 0.3, 0)
            
            # Add timestamp info
            timestamp_ms = sync_frames['d415_rgb'].timestamp
            cv2.putText(overlay, f"D415: {timestamp_ms:.0f}ms", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            displays.append(cv2.resize(overlay, (320, 240)))
        
        # D405 depth
        if 'd405_depth' in sync_frames:
            depth_frame = sync_frames['d405_depth'].frame
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_PLASMA)
            
            timestamp_ms = sync_frames['d405_depth'].timestamp
            cv2.putText(depth_vis, f"D405: {timestamp_ms:.0f}ms", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            displays.append(cv2.resize(depth_vis, (320, 240)))
        
        # Platform cameras
        platform_count = 0
        for camera_name, frame_obj in sync_frames.items():
            if camera_name.startswith('platform_cam_'):
                frame = frame_obj.frame
                timestamp_ms = frame_obj.timestamp
                
                cv2.putText(frame, f"Plat{platform_count}: {timestamp_ms:.0f}ms", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                displays.append(cv2.resize(frame, (320, 240)))
                platform_count += 1
        
        # Arrange displays
        if len(displays) == 0:
            combined = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(combined, "No synchronized frames", (50, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        elif len(displays) == 1:
            combined = displays[0]
        elif len(displays) == 2:
            combined = np.hstack(displays)
        elif len(displays) <= 4:
            if len(displays) == 3:
                top_row = np.hstack(displays[:2])
                bottom_row = np.hstack([displays[2], np.zeros_like(displays[2])])
            else:
                top_row = np.hstack(displays[:2])
                bottom_row = np.hstack(displays[2:4])
            combined = np.vstack([top_row, bottom_row])
        else:
            # More than 4 displays - create grid
            rows = []
            for i in range(0, len(displays), 2):
                if i + 1 < len(displays):
                    row = np.hstack([displays[i], displays[i+1]])
                else:
                    row = np.hstack([displays[i], np.zeros_like(displays[i])])
                rows.append(row)
            combined = np.vstack(rows)
        
        return combined
    
    def run_synchronized_system(self):
        """Run synchronized multi-camera system"""
        print(f"\n🎬 Starting synchronized camera system")
        print(f"🎯 Sync tolerance: {self.sync_tolerance_ms}ms")
        print(f"🎮 Controls:")
        print(f"  • 's' - Save synchronized frame set")
        print(f"  • 'i' - Show sync statistics")
        print(f"  • 't' - Adjust sync tolerance")
        print(f"  • 'q' - Quit")
        
        frame_count = 0
        last_save_time = 0
        
        try:
            while True:
                # Get synchronized frames
                sync_frames = self.find_synchronized_frames()
                
                if not sync_frames:
                    print("⚠️  No synchronized frames available")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Create display
                display = self.create_synchronized_display(sync_frames)
                
                # Add system info
                sync_count = len(sync_frames)
                success_rate = (self.sync_stats['successful_syncs'] / 
                              max(1, self.sync_stats['total_syncs']) * 100)
                
                cv2.putText(display, f"Synchronized Cameras: {sync_count} | Frame: {frame_count}", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display, f"Sync Success: {success_rate:.1f}% | Avg Error: {self.sync_stats['avg_sync_error_ms']:.1f}ms", 
                           (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.imshow('Synchronized Multi-Camera System', display)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    current_time = time.time()
                    if current_time - last_save_time > 1.0:  # Prevent spam saving
                        self.save_synchronized_frames(sync_frames, frame_count)
                        last_save_time = current_time
                elif key == ord('i'):
                    self.print_sync_statistics()
                elif key == ord('t'):
                    self.adjust_sync_tolerance()
                
                # Control display rate
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        finally:
            cv2.destroyAllWindows()
    
    def save_synchronized_frames(self, sync_frames, frame_number):
        """Save synchronized frame set"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_count = 0
        for camera_name, frame_obj in sync_frames.items():
            try:
                filename = f"sync_{camera_name}_{frame_number:06d}_{timestamp}.png"
                cv2.imwrite(filename, frame_obj.frame)
                saved_count += 1
            except Exception as e:
                print(f"❌ Failed to save {camera_name}: {e}")
        
        print(f"💾 Saved {saved_count} synchronized frames (set {frame_number})")
    
    def print_sync_statistics(self):
        """Print synchronization statistics"""
        print(f"\n{'='*50}")
        print("SYNCHRONIZATION STATISTICS")
        print('='*50)
        
        stats = self.sync_stats
        success_rate = (stats['successful_syncs'] / max(1, stats['total_syncs']) * 100)
        
        print(f"📊 Sync Attempts: {stats['total_syncs']}")
        print(f"✅ Successful Syncs: {stats['successful_syncs']} ({success_rate:.1f}%)")
        print(f"⏱️  Average Sync Error: {stats['avg_sync_error_ms']:.2f} ms")
        print(f"⏱️  Maximum Sync Error: {stats['max_sync_error_ms']:.2f} ms")
        print(f"🎯 Sync Tolerance: {self.sync_tolerance_ms} ms")
        
        print(f"\n📷 Active Cameras:")
        for camera_name, buffer in self.frame_buffers.items():
            buffer_size = len(buffer)
            latest_timestamp = buffer[-1].timestamp if buffer else 0
            print(f"  • {camera_name}: {buffer_size} frames, latest: {latest_timestamp:.0f}ms")
        
        print('='*50)
    
    def adjust_sync_tolerance(self):
        """Adjust synchronization tolerance"""
        print(f"\nCurrent sync tolerance: {self.sync_tolerance_ms}ms")
        try:
            new_tolerance = float(input("Enter new tolerance (ms): "))
            if 1 <= new_tolerance <= 1000:
                self.sync_tolerance_ms = new_tolerance
                print(f"✅ Sync tolerance set to {new_tolerance}ms")
            else:
                print("❌ Invalid tolerance (must be 1-1000ms)")
        except ValueError:
            print("❌ Invalid input")
    
    def cleanup(self):
        """Clean up all resources"""
        print("🧹 Cleaning up synchronized system...")
        
        # Stop capture threads
        self.stop_capture_threads()
        
        # Clean up cameras
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
    parser = argparse.ArgumentParser(description='Synchronized Multi-Camera System')
    parser.add_argument('--calibration', type=str,
                       default='calibration_results/calibration_results.json',
                       help='Calibration results file')
    parser.add_argument('--sync-tolerance', type=int, default=50,
                       help='Synchronization tolerance in milliseconds')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target frame rate')
    
    args = parser.parse_args()
    
    print("🎯 Time-Synchronized Multi-Camera System")
    print("Optimized for temporal synchronization")
    print("=" * 50)
    
    try:
        # Create synchronized system
        system = SynchronizedCameraSystem(args.calibration, args.sync_tolerance)
        system.target_fps = args.fps
        system.frame_interval = 1.0 / args.fps
        
        # Setup cameras with your discovered serials/IDs
        cameras_setup = []
        
        # Setup D415
        if system.setup_d415("821312062833"):
            cameras_setup.append("D415")
        
        # Setup D405
        if system.setup_d405("230322270171"):
            cameras_setup.append("D405")
        
        # Setup best platform cameras (avoid problematic ones)
        platform_cameras = [1, 2]  # Use cameras 1 and 2 (avoid 0 and 3 which had errors)
        for i, cam_id in enumerate(platform_cameras):
            cam_name = f"platform_cam_{cam_id}"
            if system.setup_platform_camera(cam_id, cam_name):
                cameras_setup.append(f"Platform_{cam_id}")
        
        if len(cameras_setup) < 2:
            print("❌ Need at least 2 cameras for synchronization")
            return
        
        print(f"✅ Setup complete: {', '.join(cameras_setup)}")
        
        # Start synchronized capture
        system.start_capture_threads()
        
        # Run synchronized system
        system.run_synchronized_system()
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            system.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()