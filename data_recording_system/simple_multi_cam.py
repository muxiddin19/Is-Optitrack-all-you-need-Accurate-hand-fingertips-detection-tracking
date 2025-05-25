#!/usr/bin/env python3
"""
Simple Multi-Camera Capture - Working Version
Simplified implementation to ensure it works with your setup
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

class SimpleMultiCameraCapture:
    """Simplified multi-camera capture that actually works"""
    
    def __init__(self, output_dir="simple_capture"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")
        
        # Camera objects
        self.d415_pipeline = None
        self.d405_pipeline = None
        self.platform_cameras = {}
        
        # Recording state
        self.recording = False
        self.frame_count = 0
        self.target_range = (0.4, 0.58)  # 400-580mm
        
        print("🎯 Simple Multi-Camera Capture initialized")
    
    def test_cameras(self):
        """Test which cameras are available"""
        print("\n🔍 Testing camera availability...")
        
        available_cameras = []
        
        # Test D415
        if REALSENSE_AVAILABLE:
            try:
                ctx = rs.context()
                devices = ctx.query_devices()
                
                for device in devices:
                    name = device.get_info(rs.camera_info.name)
                    serial = device.get_info(rs.camera_info.serial_number)
                    print(f"  Found: {name} (SN: {serial})")
                    
                    if "D415" in name:
                        available_cameras.append(("D415", serial))
                    elif "D405" in name:
                        available_cameras.append(("D405", serial))
                        
            except Exception as e:
                print(f"  RealSense test error: {e}")
        
        # Test platform cameras
        for cam_id in range(5):
            try:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        available_cameras.append(("Platform", cam_id, f"{w}x{h}"))
                        print(f"  Found: Platform Camera {cam_id} ({w}x{h})")
                    cap.release()
            except Exception as e:
                pass
        
        print(f"✅ Found {len(available_cameras)} cameras total")
        return available_cameras
    
    def setup_d415(self, serial):
        """Simple D415 setup"""
        print(f"🔧 Setting up D415 (SN: {serial})...")
        
        try:
            self.d415_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            
            # Basic configuration
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.d415_pipeline.start(config)
            
            # Test capture
            frames = self.d415_pipeline.wait_for_frames(timeout_ms=2000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame and depth_frame:
                print("  ✅ D415 ready")
                return True
            else:
                print("  ❌ D415 frame test failed")
                return False
                
        except Exception as e:
            print(f"  ❌ D415 setup failed: {e}")
            return False
    
    def setup_d405(self, serial):
        """Simple D405 setup"""
        print(f"🔧 Setting up D405 (SN: {serial})...")
        
        try:
            self.d405_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            
            # D405 configuration
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            
            # Start pipeline
            self.d405_pipeline.start(config)
            
            # Test capture
            frames = self.d405_pipeline.wait_for_frames(timeout_ms=2000)
            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame(1)
            
            if depth_frame and ir_frame:
                print("  ✅ D405 ready")
                return True
            else:
                print("  ❌ D405 frame test failed")
                return False
                
        except Exception as e:
            print(f"  ❌ D405 setup failed: {e}")
            return False
    
    def setup_platform_camera(self, cam_id):
        """Simple platform camera setup"""
        print(f"🔧 Setting up Platform Camera {cam_id}...")
        
        try:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                # Basic settings
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test capture
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.platform_cameras[f"platform_{cam_id}"] = cap
                    print(f"  ✅ Platform Camera {cam_id} ready")
                    return True
                else:
                    cap.release()
                    print(f"  ❌ Platform Camera {cam_id} frame test failed")
                    return False
            else:
                print(f"  ❌ Platform Camera {cam_id} not accessible")
                return False
                
        except Exception as e:
            print(f"  ❌ Platform Camera {cam_id} setup failed: {e}")
            return False
    
    def capture_all_frames(self):
        """Capture frames from all active cameras"""
        frames = {}
        
        # Capture D415
        if self.d415_pipeline:
            try:
                rs_frames = self.d415_pipeline.wait_for_frames(timeout_ms=100)
                
                color_frame = rs_frames.get_color_frame()
                depth_frame = rs_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    frames['d415_rgb'] = color_image
                    frames['d415_depth'] = depth_image
                    
            except Exception as e:
                print(f"D415 capture error: {e}")
        
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
        """Create simple RGB-depth overlay"""
        try:
            # Filter depth to target range
            min_depth = self.target_range[0] * 1000  # Convert to mm
            max_depth = self.target_range[1] * 1000
            
            # Create mask for target range
            depth_mask = (depth_frame >= min_depth) & (depth_frame <= max_depth)
            
            # Normalize and colorize depth
            depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Create overlay
            overlay = cv2.addWeighted(rgb_frame, 0.7, depth_colored, 0.3, 0)
            
            # Add range info
            valid_depth = depth_frame[depth_frame > 0]
            if len(valid_depth) > 0:
                mean_depth = np.mean(valid_depth)
                
                if min_depth <= mean_depth <= max_depth:
                    color = (0, 255, 0)  # Green - good range
                    status = "OPTIMAL RANGE"
                else:
                    color = (0, 255, 255)  # Yellow - adjust needed
                    status = "ADJUST DISTANCE"
                
                cv2.putText(overlay, f"{status}: {mean_depth:.0f}mm", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return overlay
            
        except Exception as e:
            print(f"Overlay creation error: {e}")
            return rgb_frame
    
    def create_display(self, frames):
        """Create multi-camera display"""
        displays = []
        
        # D415 with depth overlay
        if 'd415_rgb' in frames and 'd415_depth' in frames:
            overlay = self.create_depth_overlay(frames['d415_rgb'], frames['d415_depth'])
            cv2.putText(overlay, "D415: RGB+Depth", (10, overlay.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            displays.append(cv2.resize(overlay, (320, 240)))
        
        # D405 depth visualization
        if 'd405_depth' in frames:
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(frames['d405_depth'], alpha=0.05), cv2.COLORMAP_PLASMA)
            cv2.putText(depth_vis, "D405: Depth", (10, depth_vis.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            displays.append(cv2.resize(depth_vis, (320, 240)))
        
        # D405 IR
        if 'd405_ir' in frames:
            ir_frame = frames['d405_ir']
            cv2.putText(ir_frame, "D405: IR", (10, ir_frame.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            displays.append(cv2.resize(ir_frame, (320, 240)))
        
        # Platform cameras
        for cam_name, frame in frames.items():
            if cam_name.startswith('platform_'):
                cam_frame = frame.copy()
                cv2.putText(cam_frame, cam_name.upper(), (10, cam_frame.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                displays.append(cv2.resize(cam_frame, (320, 240)))
        
        # Arrange displays
        if len(displays) == 0:
            combined = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(combined, "No camera feeds", (50, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        elif len(displays) == 1:
            combined = displays[0]
        elif len(displays) == 2:
            combined = np.hstack(displays)
        elif len(displays) <= 4:
            if len(displays) == 3:
                top_row = np.hstack(displays[:2])
                bottom_row = np.hstack([displays[2], np.zeros((240, 320, 3), dtype=np.uint8)])
            else:
                top_row = np.hstack(displays[:2])
                bottom_row = np.hstack(displays[2:4])
            combined = np.vstack([top_row, bottom_row])
        else:
            # More than 4 displays
            rows = []
            for i in range(0, len(displays), 2):
                if i + 1 < len(displays):
                    row = np.hstack([displays[i], displays[i+1]])
                else:
                    row = np.hstack([displays[i], np.zeros((240, 320, 3), dtype=np.uint8)])
                rows.append(row)
            combined = np.vstack(rows)
        
        return combined
    
    def save_frame_set(self, frames):
        """Save current frame set"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_num = f"{self.frame_count:06d}"
        
        saved_count = 0
        
        for frame_name, frame_data in frames.items():
            try:
                filename = self.output_dir / f"{frame_num}_{frame_name}_{timestamp}.png"
                cv2.imwrite(str(filename), frame_data)
                saved_count += 1
            except Exception as e:
                print(f"Error saving {frame_name}: {e}")
        
        self.frame_count += 1
        print(f"📸 Saved {saved_count} frames (set {self.frame_count})")
    
    def run_capture(self):
        """Run the capture session"""
        print(f"\n🎬 Starting Simple Multi-Camera Capture")
        print(f"🎯 Target range: {self.target_range[0]*1000:.0f}-{self.target_range[1]*1000:.0f}mm")
        print(f"📁 Output: {self.output_dir}")
        print(f"\n🎮 Controls:")
        print(f"  • SPACE - Save current frame set")
        print(f"  • 'r' - Start/stop recording")  
        print(f"  • 'q' - Quit")
        
        recording_start_time = 0
        
        try:
            while True:
                # Capture frames from all cameras
                frames = self.capture_all_frames()
                
                if not frames:
                    print("⚠️  No frames captured")
                    time.sleep(0.1)
                    continue
                
                # Create display
                display = self.create_display(frames)
                
                # Add status info
                status_text = f"Cameras: {len(frames)} | Frames saved: {self.frame_count}"
                cv2.putText(display, status_text, (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if self.recording:
                    recording_time = time.time() - recording_start_time
                    cv2.putText(display, f"🔴 RECORDING: {recording_time:.1f}s", (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Auto-save while recording
                    if int(recording_time) % 2 == 0:  # Save every 2 seconds
                        self.save_frame_set(frames)
                
                cv2.imshow('Simple Multi-Camera Capture', display)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.save_frame_set(frames)
                elif key == ord('r'):
                    self.recording = not self.recording
                    if self.recording:
                        recording_start_time = time.time()
                        print("🔴 Recording started (auto-save every 2s)")
                    else:
                        print("⏹️  Recording stopped")
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        finally:
            cv2.destroyAllWindows()
        
        print(f"\n✅ Capture session complete!")
        print(f"📊 Total frame sets saved: {self.frame_count}")
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

def main():
    parser = argparse.ArgumentParser(description='Simple Multi-Camera Capture')
    parser.add_argument('--output-dir', type=str, default='simple_capture',
                       help='Output directory')
    parser.add_argument('--min-range', type=float, default=0.4,
                       help='Minimum range in meters')
    parser.add_argument('--max-range', type=float, default=0.58,
                       help='Maximum range in meters')
    
    args = parser.parse_args()
    
    print("🎯 Simple Multi-Camera Capture System")
    print("=" * 50)
    
    try:
        # Create capture system
        capture = SimpleMultiCameraCapture(args.output_dir)
        capture.target_range = (args.min_range, args.max_range)
        
        # Test available cameras
        available = capture.test_cameras()
        
        if not available:
            print("❌ No cameras found!")
            return
        
        # Setup cameras interactively
        print(f"\nSetup cameras? (y/n): ", end="")
        if input().lower().startswith('y'):
            
            # Try to setup D415
            d415_serials = [item[1] for item in available if item[0] == "D415"]
            if d415_serials:
                capture.setup_d415(d415_serials[0])
            
            # Try to setup D405  
            d405_serials = [item[1] for item in available if item[0] == "D405"]
            if d405_serials:
                capture.setup_d405(d405_serials[0])
            
            # Try to setup platform cameras
            platform_ids = [item[1] for item in available if item[0] == "Platform"]
            for cam_id in platform_ids[:2]:  # Setup first 2 platform cameras
                capture.setup_platform_camera(cam_id)
        
        # Check if any cameras were setup
        total_cameras = 0
        if capture.d415_pipeline:
            total_cameras += 1
        if capture.d405_pipeline:
            total_cameras += 1
        total_cameras += len(capture.platform_cameras)
        
        if total_cameras == 0:
            print("❌ No cameras were setup successfully!")
            return
        
        print(f"✅ {total_cameras} cameras ready for capture")
        
        # Run capture
        capture.run_capture()
        
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