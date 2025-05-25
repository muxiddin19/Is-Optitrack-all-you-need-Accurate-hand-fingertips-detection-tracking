#!/usr/bin/env python3
"""
Four-Camera System Manager
D415 (RGB+Depth) + D405 (Depth) + 2 Platform Cameras (RGB)
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import time
import threading
from datetime import datetime

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠️  pyrealsense2 not available")

class FourCameraSystem:
    """Unified 4-camera system with optimal alignment"""
    
    def __init__(self, calibration_file=None):
        self.calibration_file = calibration_file
        self.calibration_data = None
        
        # Camera objects
        self.d415_pipeline = None
        self.d405_pipeline = None
        self.platform_cameras = {}  # {camera_name: cv2.VideoCapture}
        
        # RealSense configurations
        self.d415_config = None
        self.d405_config = None
        self.d415_align = None  # Hardware alignment for D415
        
        # Camera information
        self.camera_info = {}
        self.active_cameras = {}
        
        # System parameters
        self.depth_scale_d415 = 0.001
        self.depth_scale_d405 = 0.001
        
        if calibration_file:
            self.load_calibration()
    
    def load_calibration(self):
        """Load calibration data"""
        try:
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            print(f"✅ Loaded calibration from {self.calibration_file}")
            
            # Extract camera info
            for cam_name, cam_data in self.calibration_data.get('cameras', {}).items():
                self.camera_info[cam_name] = cam_data
                
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
    
    def discover_system_cameras(self):
        """Discover all system cameras"""
        print("🔍 Discovering 4-camera system...")
        print("=" * 50)
        
        # Discover RealSense devices
        rs_devices = self.discover_realsense_devices()
        
        # Discover platform cameras
        platform_cams = self.discover_platform_cameras()
        
        # Print system summary
        self.print_system_summary(rs_devices, platform_cams)
        
        return len(rs_devices) + len(platform_cams) >= 3  # Minimum viable system
    
    def discover_realsense_devices(self):
        """Discover RealSense devices (D415, D405)"""
        if not REALSENSE_AVAILABLE:
            print("🎯 RealSense not available")
            return []
        
        print("🎯 Discovering RealSense devices...")
        
        found_devices = []
        
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            for device in devices:
                name = device.get_info(rs.camera_info.name)
                serial = device.get_info(rs.camera_info.serial_number)
                
                if "D415" in name or "D405" in name:
                    device_info = {
                        'name': name,
                        'serial': serial,
                        'type': 'd415' if 'D415' in name else 'd405'
                    }
                    found_devices.append(device_info)
                    print(f"  ✅ {name} (SN: {serial})")
                    
        except Exception as e:
            print(f"  ❌ RealSense discovery error: {e}")
        
        return found_devices
    
    def discover_platform_cameras(self):
        """Discover platform RGB cameras"""
        print("📷 Discovering platform cameras...")
        
        found_cameras = []
        
        for camera_id in range(6):  # Test camera IDs 0-5
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(camera_id, backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            h, w = frame.shape[:2]
                            camera_info = {
                                'name': f'platform_cam_{camera_id}',
                                'id': camera_id,
                                'resolution': (w, h),
                                'backend': backend
                            }
                            found_cameras.append(camera_info)
                            print(f"  ✅ Platform Camera {camera_id}: {w}x{h}")
                            cap.release()
                            break
                        cap.release()
                except Exception:
                    pass
        
        return found_cameras
    
    def print_system_summary(self, rs_devices, platform_cams):
        """Print system discovery summary"""
        print(f"\n{'='*50}")
        print("4-CAMERA SYSTEM SUMMARY")
        print('='*50)
        
        print(f"🎯 RealSense Devices ({len(rs_devices)}):")
        for device in rs_devices:
            streams = "RGB+Depth" if device['type'] == 'd415' else "Depth+IR"  
            print(f"  • {device['name']}: {streams} (SN: {device['serial']})")
        
        print(f"\n📷 Platform Cameras ({len(platform_cams)}):")
        for cam in platform_cams:
            print(f"  • {cam['name']}: {cam['resolution'][0]}x{cam['resolution'][1]} (ID: {cam['id']})")
        
        total = len(rs_devices) + len(platform_cams)
        print(f"\n📊 Total Cameras: {total}")
        
        if total >= 4:
            print("✅ Full 4-camera system available!")
        elif total >= 3:
            print("✅ 3-camera system available (minimum viable)")
        else:
            print("❌ Insufficient cameras for system")
        
        print('='*50)
    
    def setup_d415(self, serial_number):
        """Setup D415 with RGB and depth streams"""
        print(f"🔧 Setting up D415 (SN: {serial_number})...")
        
        try:
            self.d415_pipeline = rs.pipeline()
            self.d415_config = rs.config()
            self.d415_config.enable_device(serial_number)
            
            # Configure streams
            self.d415_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.d415_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.d415_config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            
            # Start pipeline
            profile = self.d415_pipeline.start(self.d415_config)
            
            # Get depth scale
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self.depth_scale_d415 = depth_sensor.get_depth_scale()
            
            # Setup hardware alignment
            self.d415_align = rs.align(rs.stream.color)
            
            # Test capture
            frames = self.d415_pipeline.wait_for_frames(timeout_ms=2000)
            if frames.get_color_frame() and frames.get_depth_frame():
                print(f"  ✅ D415 ready (RGB+Depth+IR)")
                return True
            else:
                print(f"  ❌ D415 frame test failed")
                return False
                
        except Exception as e:
            print(f"  ❌ D415 setup failed: {e}")
            return False
    
    def setup_d405(self, serial_number):
        """Setup D405 with depth and IR streams"""
        print(f"🔧 Setting up D405 (SN: {serial_number})...")
        
        try:
            self.d405_pipeline = rs.pipeline()
            self.d405_config = rs.config()
            self.d405_config.enable_device(serial_number)
            
            # D405 doesn't have RGB, only depth and IR
            self.d405_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.d405_config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            
            # Start pipeline
            profile = self.d405_pipeline.start(self.d405_config)
            
            # Get depth scale
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self.depth_scale_d405 = depth_sensor.get_depth_scale()
            
            # Test capture
            frames = self.d405_pipeline.wait_for_frames(timeout_ms=2000)
            if frames.get_depth_frame() and frames.get_infrared_frame(1):
                print(f"  ✅ D405 ready (Depth+IR)")
                return True
            else:
                print(f"  ❌ D405 frame test failed")
                return False
                
        except Exception as e:
            print(f"  ❌ D405 setup failed: {e}")
            return False
    
    def setup_platform_camera(self, camera_info):
        """Setup platform camera"""
        camera_name = camera_info['name']
        camera_id = camera_info['id']
        backend = camera_info['backend']
        
        print(f"🔧 Setting up {camera_name} (ID: {camera_id})...")
        
        try:
            cap = cv2.VideoCapture(camera_id, backend)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            ret, frame = cap.read()
            if ret and frame is not None:
                self.platform_cameras[camera_name] = cap
                print(f"  ✅ {camera_name} ready")
                return True
            else:
                print(f"  ❌ {camera_name} frame test failed")
                cap.release()
                return False
                
        except Exception as e:
            print(f"  ❌ {camera_name} setup failed: {e}")
            return False
    
    def setup_system(self, selected_devices):
        """Setup the complete 4-camera system"""
        print(f"\n🔧 Setting up 4-camera system...")
        
        setup_success = True
        
        for device in selected_devices:
            if device['type'] == 'd415':
                if not self.setup_d415(device['serial']):
                    setup_success = False
            elif device['type'] == 'd405':
                if not self.setup_d405(device['serial']):
                    setup_success = False
            elif device['type'] == 'platform':
                if not self.setup_platform_camera(device):
                    setup_success = False
        
        if setup_success:
            print(f"✅ 4-camera system ready!")
        else:
            print(f"⚠️  Some cameras failed to setup")
        
        return setup_success
    
    def capture_all_frames(self):
        """Capture frames from all active cameras"""
        frames = {}
        
        # Capture D415 frames (RGB + Depth)
        if self.d415_pipeline:
            try:
                rs_frames = self.d415_pipeline.wait_for_frames(timeout_ms=100)
                
                # Apply hardware alignment
                aligned_frames = self.d415_align.process(rs_frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    frames['d415_rgb'] = np.asanyarray(color_frame.get_data())
                    frames['d415_depth'] = np.asanyarray(depth_frame.get_data())
                    
            except Exception as e:
                print(f"D415 capture error: {e}")
        
        # Capture D405 frames (Depth + IR)
        if self.d405_pipeline:
            try:
                rs_frames = self.d405_pipeline.wait_for_frames(timeout_ms=100)
                
                depth_frame = rs_frames.get_depth_frame()
                ir_frame = rs_frames.get_infrared_frame(1)
                
                if depth_frame and ir_frame:
                    frames['d405_depth'] = np.asanyarray(depth_frame.get_data())
                    frames['d405_ir'] = np.asanyarray(ir_frame.get_data())
                    
            except Exception as e:
                print(f"D405 capture error: {e}")
        
        # Capture platform camera frames
        for cam_name, cap in self.platform_cameras.items():
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames[cam_name] = frame
            except Exception as e:
                print(f"{cam_name} capture error: {e}")
        
        return frames
    
    def create_system_visualization(self, frames):
        """Create 4-camera system visualization"""
        displays = []
        
        # D415 RGB with depth overlay (primary view)
        if 'd415_rgb' in frames and 'd415_depth' in frames:
            rgb_frame = frames['d415_rgb']
            depth_frame = frames['d415_depth']
            
            # Create depth overlay
            depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Blend with RGB
            overlay = cv2.addWeighted(rgb_frame, 0.7, depth_colored, 0.3, 0)
            cv2.putText(overlay, "D415: RGB + Depth", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            displays.append(cv2.resize(overlay, (320, 240)))
        
        # D405 Depth visualization
        if 'd405_depth' in frames:
            depth_frame = frames['d405_depth']
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_PLASMA)
            cv2.putText(depth_vis, "D405: Depth", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            displays.append(cv2.resize(depth_vis, (320, 240)))
        
        # Platform cameras
        platform_count = 0
        for cam_name, frame in frames.items():
            if cam_name.startswith('platform_cam_'):
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Platform {platform_count+1}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                displays.append(cv2.resize(display_frame, (320, 240)))
                platform_count += 1
        
        # Arrange in 2x2 grid
        if len(displays) >= 4:
            top_row = np.hstack(displays[:2])
            bottom_row = np.hstack(displays[2:4])
            combined = np.vstack([top_row, bottom_row])
        elif len(displays) == 3:
            top_row = np.hstack(displays[:2])
            bottom_row = np.hstack([displays[2], np.zeros_like(displays[2])])
            combined = np.vstack([top_row, bottom_row])
        elif len(displays) == 2:
            combined = np.hstack(displays)
        elif len(displays) == 1:
            combined = displays[0]
        else:
            combined = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(combined, "No frames available", (200, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined
    
    def run_system_monitor(self):
        """Run live 4-camera system monitor"""
        print(f"\n🎬 Starting 4-camera system monitor")
        print(f"🎮 Controls:")
        print(f"  • 's' - Save all frames")
        print(f"  • 'i' - Show system info")
        print(f"  • 'q' - Quit")
        
        frame_count = 0
        
        try:
            while True:
                # Capture from all cameras
                all_frames = self.capture_all_frames()
                
                if not all_frames:
                    print("❌ No frames captured")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Create visualization
                display = self.create_system_visualization(all_frames)
                
                # Add system info
                cv2.putText(display, f"4-Camera System | Frame: {frame_count}", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display, f"Active: {len(all_frames)} streams", (10, display.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('4-Camera System Monitor', display)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_all_frames(all_frames, frame_count)
                elif key == ord('i'):
                    self.print_system_info()
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        finally:
            cv2.destroyAllWindows()
    
    def save_all_frames(self, frames, frame_number):
        """Save frames from all cameras"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_count = 0
        for stream_name, frame in frames.items():
            try:
                filename = f"frame_{stream_name}_{frame_number:06d}_{timestamp}.png"
                cv2.imwrite(filename, frame)
                saved_count += 1
            except Exception as e:
                print(f"❌ Failed to save {stream_name}: {e}")
        
        print(f"💾 Saved {saved_count} frames (set {frame_number})")
    
    def print_system_info(self):
        """Print detailed system information"""
        print(f"\n{'='*50}")
        print("4-CAMERA SYSTEM STATUS")
        print('='*50)
        
        print(f"📊 Active Streams:")
        if self.d415_pipeline:
            print(f"  ✅ D415: RGB + Depth (scale: {self.depth_scale_d415})")
        if self.d405_pipeline:
            print(f"  ✅ D405: Depth + IR (scale: {self.depth_scale_d405})")
        
        for cam_name in self.platform_cameras:
            print(f"  ✅ {cam_name}: RGB")
        
        if self.calibration_data:
            print(f"\n🎯 Calibration Status:")
            intrinsics_count = len(self.calibration_data.get('intrinsics', {}))
            stereo_count = len(self.calibration_data.get('stereo_pairs', {}))
            print(f"  • {intrinsics_count} intrinsic calibrations")
            print(f"  • {stereo_count} stereo calibrations")
        
        print('='*50)
    
    def cleanup(self):
        """Clean up all resources"""
        print("🧹 Cleaning up 4-camera system...")
        
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
    parser = argparse.ArgumentParser(description='4-Camera System Manager')
    parser.add_argument('--calibration', type=str,
                       default='calibration_results/calibration_results.json',
                       help='Calibration results file')
    parser.add_argument('--discover-only', action='store_true',
                       help='Only discover cameras')
    
    args = parser.parse_args()
    
    print("🎯 4-Camera System Manager")
    print("D415 + D405 + 2 Platform Cameras")
    print("=" * 50)
    
    try:
        # Create system manager
        system = FourCameraSystem(args.calibration)
        
        # Discover cameras
        if not system.discover_system_cameras():
            print("❌ Insufficient cameras for system")
            return
        
        if args.discover_only:
            return
        
        # Interactive setup (you'd customize this based on your specific devices)
        print(f"\n🔧 System setup required...")
        print(f"Modify the script to specify your exact device serials/IDs")
        
        # Example setup - customize with your actual device identifiers
        selected_devices = [
            {'type': 'd415', 'serial': 'YOUR_D415_SERIAL'},
            {'type': 'd405', 'serial': 'YOUR_D405_SERIAL'},
            {'type': 'platform', 'name': 'platform_cam_1', 'id': 1, 'backend': cv2.CAP_DSHOW},
            {'type': 'platform', 'name': 'platform_cam_3', 'id': 3, 'backend': cv2.CAP_DSHOW}
        ]
        
        # For now, just run discovery
        print("📝 Customize the script with your specific camera serials/IDs")
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        try:
            system.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()