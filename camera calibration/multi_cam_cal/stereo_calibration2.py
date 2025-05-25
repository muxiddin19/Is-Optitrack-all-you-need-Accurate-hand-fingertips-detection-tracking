#!/usr/bin/env python3
"""
Multi-Camera Calibration Tool
Supports RGB cameras + RealSense devices (D405, D415, etc.)
"""

import cv2
import numpy as np
import os
import pickle
import json
import argparse
from pathlib import Path
import time
from datetime import datetime

# Optional RealSense support
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠️  pyrealsense2 not available. RealSense cameras will not be detected.")

class MultiCameraCalibrationTool:
    """Complete multi-camera calibration system"""
    
    def __init__(self, output_path="calibration_results"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Camera storage
        self.rgb_cameras = {}
        self.realsense_devices = {}
        self.camera_info = {}
        
        # Calibration parameters
        self.checkerboard_size = (7, 4)  # Internal corners
        self.square_size = 0.025  # 25mm
        
        # Data storage
        self.calibration_images = {}
        self.image_points = {}
        self.object_points = []
        
        # Results
        self.intrinsic_results = {}
        self.stereo_results = {}
    
    def test_camera_backends(self, camera_id):
        """Test different OpenCV backends"""
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Any Available")
        ]
        
        for backend, name in backends:
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffering
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        cap.release()
                        print(f"    ✅ {name}: {width}x{height}")
                        return backend
                    cap.release()
            except Exception as e:
                print(f"    ❌ {name}: {e}")
        
        return None
    
    def discover_rgb_cameras(self, max_test=5):
        """Discover RGB cameras with robust backend testing"""
        print("📷 Discovering RGB cameras...")
        
        found_cameras = []
        
        for camera_id in range(max_test):
            print(f"  Testing camera {camera_id}...")
            
            # Find working backend
            backend = self.test_camera_backends(camera_id)
            
            if backend is not None:
                try:
                    # Open with working backend
                    cap = cv2.VideoCapture(camera_id, backend)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        
                        camera_name = f"rgb_cam_{camera_id}"
                        self.rgb_cameras[camera_name] = cap
                        self.camera_info[camera_name] = {
                            'type': 'rgb',
                            'id': camera_id,
                            'backend': backend,
                            'resolution': (width, height)
                        }
                        
                        found_cameras.append(camera_name)
                        print(f"    ✅ Found RGB camera {camera_id}: {width}x{height}")
                    else:
                        cap.release()
                        
                except Exception as e:
                    print(f"    ❌ Error setting up camera {camera_id}: {e}")
        
        return found_cameras
    
    def discover_realsense_devices(self):
        """Discover RealSense devices"""
        if not REALSENSE_AVAILABLE:
            print("🎯 RealSense not available - skipping")
            return []
        
        print("🎯 Discovering RealSense devices...")
        found_devices = []
        
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                print("    No RealSense devices found")
                return found_devices
            
            for device in devices:
                try:
                    serial = device.get_info(rs.camera_info.serial_number)
                    name = device.get_info(rs.camera_info.name)
                    
                    print(f"  Testing {name} (SN: {serial})...")
                    
                    # Create and test pipeline
                    pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_device(serial)
                    
                    # Configure based on device type
                    if "D405" in name:
                        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
                        streams = "depth+ir"
                    elif "D415" in name or "D435" in name:
                        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
                        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                        streams = "depth+ir+rgb"
                    else:
                        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
                        streams = "depth+ir"
                    
                    # Test pipeline
                    profile = pipeline.start(config)
                    
                    # Test frame capture
                    success = False
                    for attempt in range(3):
                        try:
                            frames = pipeline.wait_for_frames(timeout_ms=2000)
                            ir_frame = frames.get_infrared_frame(1)
                            if ir_frame:
                                success = True
                                break
                        except Exception as e:
                            print(f"      Frame test {attempt+1} failed: {e}")
                    
                    pipeline.stop()
                    
                    if success:
                        device_name = f"rs_{serial}"
                        self.realsense_devices[device_name] = {
                            'pipeline': pipeline,
                            'config': config,
                            'serial': serial
                        }
                        self.camera_info[device_name] = {
                            'type': 'realsense',
                            'model': name,
                            'serial': serial,
                            'streams': streams
                        }
                        
                        found_devices.append(device_name)
                        print(f"    ✅ {name} ready ({streams})")
                    else:
                        print(f"    ❌ {name} failed frame test")
                        
                except Exception as e:
                    print(f"    ❌ Error with {name}: {e}")
                    
        except Exception as e:
            print(f"❌ RealSense discovery error: {e}")
        
        return found_devices
    
    def discover_all_cameras(self):
        """Discover all available cameras"""
        print("🔍 Discovering all cameras...")
        print("=" * 50)
        
        rgb_cameras = self.discover_rgb_cameras()
        rs_devices = self.discover_realsense_devices()
        
        # Print summary
        print(f"\n{'='*50}")
        print("DISCOVERY SUMMARY")
        print('='*50)
        
        if rgb_cameras:
            print(f"📷 RGB Cameras ({len(rgb_cameras)}):")
            for cam in rgb_cameras:
                info = self.camera_info[cam]
                print(f"  • {cam}: ID {info['id']}, {info['resolution'][0]}x{info['resolution'][1]}")
        
        if rs_devices:
            print(f"🎯 RealSense Devices ({len(rs_devices)}):")
            for dev in rs_devices:
                info = self.camera_info[dev]
                print(f"  • {dev}: {info['model']} ({info['streams']})")
        
        total = len(rgb_cameras) + len(rs_devices)
        print(f"\n📊 Total: {total} cameras discovered")
        
        if total == 0:
            print("\n❌ No cameras found!")
            print("Troubleshooting:")
            print("- Check USB connections")
            print("- Close other camera applications")
            print("- Try different USB ports")
            if not REALSENSE_AVAILABLE:
                print("- Install pyrealsense2: pip install pyrealsense2")
        
        print('='*50)
        return total > 0
    
    def select_cameras(self):
        """Interactive camera selection"""
        if not self.camera_info:
            return []
        
        cameras = list(self.camera_info.keys())
        
        print(f"\n{'='*40}")
        print("SELECT CAMERAS FOR CALIBRATION")
        print('='*40)
        
        for i, camera in enumerate(cameras):
            info = self.camera_info[camera]
            if info['type'] == 'rgb':
                desc = f"RGB Camera {info['id']}"
            else:
                desc = f"{info['model']} ({info['streams']})"
            print(f"{i+1:2d}. {camera}: {desc}")
        
        print(f"\nOptions:")
        print(f"  • Numbers separated by spaces: '1 3 4'")
        print(f"  • 'all' for all cameras")
        print(f"  • 'q' to quit")
        
        while True:
            try:
                choice = input(f"\nYour selection: ").strip()
                
                if choice.lower() == 'q':
                    return []
                elif choice.lower() == 'all':
                    selected = cameras
                    break
                else:
                    indices = [int(x) - 1 for x in choice.split()]
                    selected = [cameras[i] for i in indices if 0 <= i < len(cameras)]
                    if selected:
                        break
                    print("❌ Invalid selection")
                    
            except (ValueError, IndexError):
                print("❌ Invalid input")
        
        print(f"\n✅ Selected: {selected}")
        return selected
    
    def setup_cameras(self, selected):
        """Setup selected cameras"""
        print(f"\n🔧 Setting up {len(selected)} cameras...")
        
        active = {}
        
        for camera_name in selected:
            info = self.camera_info[camera_name]
            
            try:
                if info['type'] == 'rgb':
                    cap = self.rgb_cameras[camera_name]
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        active[camera_name] = cap
                        print(f"  ✅ {camera_name}")
                    else:
                        print(f"  ❌ {camera_name}: Frame test failed")
                        
                elif info['type'] == 'realsense':
                    rs_config = self.realsense_devices[camera_name]
                    pipeline = rs_config['pipeline']
                    config = rs_config['config']
                    
                    pipeline.start(config)
                    
                    # Test frames
                    for attempt in range(3):
                        try:
                            frames = pipeline.wait_for_frames(timeout_ms=1000)
                            if frames.get_infrared_frame(1):
                                active[camera_name] = pipeline
                                print(f"  ✅ {camera_name}")
                                break
                        except:
                            if attempt == 2:
                                print(f"  ❌ {camera_name}: Frame test failed")
                                pipeline.stop()
                            
            except Exception as e:
                print(f"  ❌ {camera_name}: {e}")
        
        print(f"✅ {len(active)} cameras ready")
        return active
    
    def capture_frame(self, camera_name, camera_obj):
        """Safely capture frame from camera"""
        info = self.camera_info[camera_name]
        
        try:
            if info['type'] == 'rgb':
                ret, frame = camera_obj.read()
                if ret and frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    return frame, gray
                    
            elif info['type'] == 'realsense':
                frames = camera_obj.wait_for_frames(timeout_ms=100)
                ir_frame = frames.get_infrared_frame(1)
                if ir_frame:
                    ir_data = np.asanyarray(ir_frame.get_data())
                    frame = cv2.cvtColor(ir_data, cv2.COLOR_GRAY2BGR)
                    return frame, ir_data
                    
        except Exception:
            pass
            
        return None, None
    
    def capture_calibration_data(self, active_cameras):
        """Capture calibration images"""
        print(f"\n{'='*60}")
        print("CALIBRATION IMAGE CAPTURE")
        print('='*60)
        print("Controls:")
        print("  SPACE - Capture when all cameras see checkerboard")
        print("  Q     - Finish and proceed to calibration")
        print("  R     - Reset if cameras get stuck")
        print("\nTarget: 15-20 good image sets from different angles")
        print('='*60)
        
        # Initialize storage
        for camera_name in active_cameras:
            self.calibration_images[camera_name] = []
            self.image_points[camera_name] = []
        
        capture_count = 0
        
        while True:
            # Capture from all cameras
            frames = {}
            corners = {}
            patterns_found = {}
            
            for camera_name, camera_obj in active_cameras.items():
                frame, gray = self.capture_frame(camera_name, camera_obj)
                
                if frame is not None and gray is not None:
                    # Find checkerboard
                    found, corner_points = cv2.findChessboardCorners(
                        gray, self.checkerboard_size, None)
                    
                    patterns_found[camera_name] = found
                    
                    if found:
                        # Refine corners
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        refined = cv2.cornerSubPix(gray, corner_points, (11, 11), (-1, -1), criteria)
                        corners[camera_name] = refined
                        
                        # Draw on frame
                        cv2.drawChessboardCorners(frame, self.checkerboard_size, refined, found)
                        cv2.putText(frame, "PATTERN FOUND", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "No pattern", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    frames[camera_name] = frame
            
            # Check if all cameras found pattern
            all_found = (len(patterns_found) == len(active_cameras) and 
                        all(patterns_found.values()))
            
            # Create display
            if frames:
                self.create_display(frames, capture_count, all_found)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("🔄 Resetting...")
                time.sleep(0.5)
            elif key == ord(' ') and all_found:
                # Store data
                for camera_name, frame in frames.items():
                    self.calibration_images[camera_name].append(frame)
                    self.image_points[camera_name].append(corners[camera_name])
                
                # Object points
                if len(self.object_points) <= capture_count:
                    objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
                    objp *= self.square_size
                    self.object_points.append(objp)
                
                capture_count += 1
                print(f"✅ Captured set {capture_count}")
                
                # Save images
                timestamp = datetime.now().strftime("%H%M%S")
                for camera_name, frame in frames.items():
                    filename = self.output_path / f"{camera_name}_{capture_count:03d}_{timestamp}.png"
                    cv2.imwrite(str(filename), frame)
        
        cv2.destroyAllWindows()
        print(f"\n📸 Captured {capture_count} image sets")
        return capture_count >= 8
    
    def create_display(self, frames, count, ready):
        """Create multi-camera display"""
        display_frames = []
        
        for name, frame in frames.items():
            # Resize for display
            small = cv2.resize(frame, (320, 240))
            
            # Add info
            cv2.putText(small, name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(small, f"Sets: {count}", (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if ready:
                cv2.putText(small, "READY!", (250, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            display_frames.append(small)
        
        # Arrange in grid
        if len(display_frames) == 1:
            combined = display_frames[0]
        elif len(display_frames) == 2:
            combined = np.hstack(display_frames)
        elif len(display_frames) <= 4:
            if len(display_frames) == 3:
                top = np.hstack(display_frames[:2])
                bottom = np.hstack([display_frames[2], np.zeros_like(display_frames[2])])
            else:
                top = np.hstack(display_frames[:2])
                bottom = np.hstack(display_frames[2:4])
            combined = np.vstack([top, bottom])
        else:
            # 5+ cameras: arrange in rows of 3
            rows = []
            for i in range(0, len(display_frames), 3):
                row = display_frames[i:i+3]
                while len(row) < 3:
                    row.append(np.zeros_like(display_frames[0]))
                rows.append(np.hstack(row))
            combined = np.vstack(rows)
        
        cv2.imshow('Multi-Camera Calibration', combined)
    
    def calibrate_intrinsics(self):
        """Calibrate each camera individually"""
        print(f"\n🔧 Performing intrinsic calibration...")
        
        success_count = 0
        
        for camera_name, image_points in self.image_points.items():
            if len(image_points) < 8:
                print(f"❌ {camera_name}: Only {len(image_points)} images")
                continue
            
            print(f"📐 Calibrating {camera_name}...")
            
            try:
                # Get image size
                first_image = self.calibration_images[camera_name][0]
                h, w = first_image.shape[:2]
                
                # Calibrate
                ret, mtx, dist, _, _ = cv2.calibrateCamera(
                    self.object_points[:len(image_points)], 
                    image_points, (w, h), None, None)
                
                # Store results
                self.intrinsic_results[camera_name] = {
                    'camera_matrix': mtx,
                    'distortion': dist,
                    'rms_error': ret,
                    'image_size': (w, h),
                    'num_images': len(image_points)
                }
                
                success_count += 1
                print(f"  ✅ RMS error: {ret:.4f} pixels")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
        
        print(f"\n✅ Successfully calibrated {success_count} cameras")
        return success_count > 0
    
    def calibrate_stereo_pairs(self):
        """Perform stereo calibration between camera pairs"""
        cameras = list(self.intrinsic_results.keys())
        
        if len(cameras) < 2:
            print("❌ Need at least 2 cameras for stereo calibration")
            return False
        
        pairs = [(cameras[i], cameras[j]) for i in range(len(cameras)) 
                for j in range(i+1, len(cameras))]
        
        print(f"\n🔗 Performing stereo calibration for {len(pairs)} pairs...")
        
        success_count = 0
        
        for cam1, cam2 in pairs:
            try:
                print(f"🔗 {cam1} ↔ {cam2}...")
                
                # Get common images
                points1 = self.image_points[cam1]
                points2 = self.image_points[cam2]
                common = min(len(points1), len(points2))
                
                if common < 8:
                    print(f"  ❌ Only {common} common images")
                    continue
                
                # Prepare data
                obj_pts = self.object_points[:common]
                pts1 = points1[:common]
                pts2 = points2[:common]
                
                mtx1 = self.intrinsic_results[cam1]['camera_matrix']
                dist1 = self.intrinsic_results[cam1]['distortion']
                mtx2 = self.intrinsic_results[cam2]['camera_matrix']
                dist2 = self.intrinsic_results[cam2]['distortion']
                
                img_size = self.intrinsic_results[cam1]['image_size']
                
                # Stereo calibrate
                ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
                    obj_pts, pts1, pts2, mtx1, dist1, mtx2, dist2,
                    img_size, flags=cv2.CALIB_FIX_INTRINSIC)
                
                # Store results
                self.stereo_results[(cam1, cam2)] = {
                    'rotation_matrix': R,
                    'translation_vector': T,
                    'rms_error': ret,
                    'num_images': common
                }
                
                success_count += 1
                print(f"  ✅ RMS error: {ret:.4f} pixels")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
        
        print(f"\n✅ Successfully calibrated {success_count} stereo pairs")
        return success_count > 0
    
    def save_results(self):
        """Save all calibration results"""
        print(f"\n💾 Saving results to {self.output_path}...")
        
        # Prepare data
        results = {
            'timestamp': datetime.now().isoformat(),
            'checkerboard_size': self.checkerboard_size,
            'square_size': self.square_size,
            'cameras': {},
            'intrinsics': {},
            'stereo_pairs': {}
        }
        
        # Camera info
        for name, info in self.camera_info.items():
            if name in self.intrinsic_results:
                results['cameras'][name] = {
                    'type': info['type'],
                    'model': info.get('model', 'Unknown'),
                    'id': info.get('id', 'N/A')
                }
        
        # Intrinsic results
        for name, data in self.intrinsic_results.items():
            results['intrinsics'][name] = {
                'camera_matrix': data['camera_matrix'].tolist(),
                'distortion': data['distortion'].tolist(),
                'rms_error': data['rms_error'],
                'image_size': data['image_size'],
                'num_images': data['num_images']
            }
        
        # Stereo results
        for (cam1, cam2), data in self.stereo_results.items():
            pair_key = f"{cam1}_{cam2}"
            results['stereo_pairs'][pair_key] = {
                'camera1': cam1,
                'camera2': cam2,
                'rotation_matrix': data['rotation_matrix'].tolist(),
                'translation_vector': data['translation_vector'].tolist(),
                'rms_error': data['rms_error'],
                'num_images': data['num_images']
            }
        
        # Save JSON
        json_file = self.output_path / "calibration_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save pickle
        pkl_file = self.output_path / "calibration_results.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"✅ Results saved:")
        print(f"  JSON: {json_file}")
        print(f"  Pickle: {pkl_file}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print calibration summary"""
        print(f"\n{'='*60}")
        print("CALIBRATION SUMMARY")
        print('='*60)
        
        print(f"📷 Intrinsic Calibrations ({len(self.intrinsic_results)}):")
        for name, data in self.intrinsic_results.items():
            cam_type = self.camera_info[name].get('model', 'RGB Camera')
            print(f"  • {name} ({cam_type}): RMS = {data['rms_error']:.4f} px, {data['num_images']} images")
        
        print(f"\n🔗 Stereo Calibrations ({len(self.stereo_results)}):")
        for (cam1, cam2), data in self.stereo_results.items():
            print(f"  • {cam1} ↔ {cam2}: RMS = {data['rms_error']:.4f} px, {data['num_images']} images")
        
        print('='*60)
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        
        # Close RGB cameras
        for cap in self.rgb_cameras.values():
            try:
                cap.release()
            except:
                pass
        
        # Stop RealSense pipelines
        for rs_data in self.realsense_devices.values():
            try:
                rs_data['pipeline'].stop()
            except:
                pass
        
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Camera Calibration Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s                              # Interactive mode
  python %(prog)s --output-path my_setup       # Custom output directory
  python %(prog)s --discover-only              # Just discover cameras
        """)
    
    parser.add_argument('--output-path', type=str, default='calibration_results',
                       help='Output directory for calibration results')
    parser.add_argument('--discover-only', action='store_true',
                       help='Only discover cameras, don\'t calibrate')
    
    args = parser.parse_args()
    
    print("🎯 Multi-Camera Calibration Tool")
    print("Supports RGB cameras + RealSense devices")
    print("=" * 50)
    
    # Create calibration tool
    calibrator = MultiCameraCalibrationTool(args.output_path)
    
    try:
        # Discover cameras
        if not calibrator.discover_all_cameras():
            return 1
        
        if args.discover_only:
            return 0
        
        # Select cameras
        selected = calibrator.select_cameras()
        if not selected:
            print("No cameras selected")
            return 1
        
        # Setup cameras
        active = calibrator.setup_cameras(selected)
        if not active:
            print("❌ No cameras could be setup")
            return 1
        
        # Capture calibration data
        if not calibrator.capture_calibration_data(active):
            print("❌ Not enough calibration data captured")
            return 1
        
        # Perform calibrations
        if not calibrator.calibrate_intrinsics():
            print("❌ Intrinsic calibration failed")
            return 1
        
        calibrator.calibrate_stereo_pairs()
        
        # Save results
        calibrator.save_results()
        
        print("\n🎉 Calibration completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    finally:
        calibrator.cleanup()

if __name__ == "__main__":
    exit(main())