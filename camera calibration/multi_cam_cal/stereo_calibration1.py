import cv2
import numpy as np
import os
import pickle
import json
import argparse
from pathlib import Path
import pyrealsense2 as rs
from datetime import datetime
import threading
import time

class MultiCameraCalibrationSystem:
    """
    Multi-camera calibration system supporting:
    - Multiple RGB/Platform cameras
    - Multiple RealSense devices (D405, D415, etc.)
    - Individual intrinsic calibration
    - Pairwise stereo calibration between any two cameras
    """
    
    def __init__(self, calibration_path="multi_camera_calibration"):
        self.calibration_path = Path(calibration_path)
        self.calibration_path.mkdir(exist_ok=True)
        
        # Camera configurations
        self.rgb_cameras = {}  # {camera_id: cv2.VideoCapture}
        self.realsense_devices = {}  # {serial_number: pipeline}
        self.camera_info = {}  # Store camera metadata
        
        # Calibration parameters
        self.checkerboard_size = (9, 6)  # Internal corners (width, height)
        self.square_size = 0.025  # 25mm squares
        
        # Calibration data storage
        self.calibration_images = {}  # {camera_id: [images]}
        self.image_points = {}  # {camera_id: [corner_points]}
        self.object_points = []  # 3D points (same for all cameras)
        
        # Calibration results
        self.intrinsic_results = {}  # {camera_id: {matrix, distortion, error}}
        self.stereo_results = {}  # {(cam1, cam2): {R, T, error}}
        
    def discover_cameras(self):
        """Discover all available cameras"""
        print("🔍 Discovering available cameras...")
        
        # Discover RGB/Platform cameras
        self.discover_rgb_cameras()
        
        # Discover RealSense devices
        self.discover_realsense_devices()
        
        # Print summary
        self.print_camera_summary()
        
    def discover_rgb_cameras(self, max_cameras=10):
        """Discover RGB/Platform cameras"""
        print("📷 Scanning for RGB cameras...")
        
        for camera_id in range(max_cameras):
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                # Test if we can actually capture
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    
                    # Store camera info
                    camera_name = f"rgb_cam_{camera_id}"
                    self.rgb_cameras[camera_name] = cap
                    self.camera_info[camera_name] = {
                        'type': 'rgb',
                        'id': camera_id,
                        'resolution': (width, height),
                        'status': 'available'
                    }
                    
                    print(f"  ✅ Found RGB Camera {camera_id}: {width}x{height}")
                else:
                    cap.release()
            else:
                cap.release()
    
    def discover_realsense_devices(self):
        """Discover RealSense devices"""
        print("🎯 Scanning for RealSense devices...")
        
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            for i, device in enumerate(devices):
                serial = device.get_info(rs.camera_info.serial_number)
                name = device.get_info(rs.camera_info.name)
                
                # Create pipeline for this device
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                
                # Configure streams based on device type
                if "D405" in name:
                    # D405: Depth + IR (no RGB)
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
                    stream_type = "depth_ir"
                elif "D415" in name or "D435" in name:
                    # D415/D435: Depth + IR + RGB
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    stream_type = "depth_ir_rgb"
                else:
                    # Generic configuration
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
                    stream_type = "depth_ir"
                
                try:
                    # Test pipeline
                    profile = pipeline.start(config)
                    pipeline.stop()
                    
                    # Store device info
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
                        'streams': stream_type,
                        'status': 'available'
                    }
                    
                    print(f"  ✅ Found {name} (SN: {serial}) - {stream_type}")
                    
                except Exception as e:
                    print(f"  ❌ Failed to configure {name}: {e}")
                    
        except Exception as e:
            print(f"❌ RealSense discovery error: {e}")
    
    def print_camera_summary(self):
        """Print summary of all discovered cameras"""
        print(f"\n{'='*60}")
        print("DISCOVERED CAMERAS SUMMARY")
        print('='*60)
        
        if not self.camera_info:
            print("❌ No cameras found!")
            return
        
        # Group by type
        rgb_cams = [(k, v) for k, v in self.camera_info.items() if v['type'] == 'rgb']
        rs_cams = [(k, v) for k, v in self.camera_info.items() if v['type'] == 'realsense']
        
        print(f"📷 RGB/Platform Cameras ({len(rgb_cams)}):")
        for name, info in rgb_cams:
            print(f"  • {name}: Camera ID {info['id']}, {info['resolution'][0]}x{info['resolution'][1]}")
        
        print(f"\n🎯 RealSense Devices ({len(rs_cams)}):")
        for name, info in rs_cams:
            print(f"  • {name}: {info['model']} (SN: {info['serial']}) - {info['streams']}")
        
        print(f"\n📊 Total: {len(self.camera_info)} cameras available")
        print('='*60)
    
    def select_cameras_for_calibration(self):
        """Interactive camera selection for calibration"""
        if not self.camera_info:
            print("❌ No cameras available for calibration")
            return []
        
        print(f"\n{'='*50}")
        print("SELECT CAMERAS FOR CALIBRATION")
        print('='*50)
        
        cameras = list(self.camera_info.keys())
        for i, camera_name in enumerate(cameras):
            info = self.camera_info[camera_name]
            if info['type'] == 'rgb':
                desc = f"RGB Camera {info['id']} ({info['resolution'][0]}x{info['resolution'][1]})"
            else:
                desc = f"{info['model']} - {info['streams']}"
            print(f"{i+1}. {camera_name}: {desc}")
        
        print("\nEnter camera numbers separated by spaces (e.g., '1 3 4'):")
        print("Or 'all' to select all cameras")
        
        try:
            selection = input("Selection: ").strip()
            
            if selection.lower() == 'all':
                selected = cameras
            else:
                indices = [int(x) - 1 for x in selection.split()]
                selected = [cameras[i] for i in indices if 0 <= i < len(cameras)]
            
            print(f"\n✅ Selected cameras: {selected}")
            return selected
            
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            return []
    
    def setup_selected_cameras(self, selected_cameras):
        """Setup and start selected cameras"""
        print(f"\n🔧 Setting up {len(selected_cameras)} cameras...")
        
        active_cameras = {}
        
        for camera_name in selected_cameras:
            info = self.camera_info[camera_name]
            
            if info['type'] == 'rgb':
                # RGB camera already opened during discovery
                if camera_name in self.rgb_cameras:
                    active_cameras[camera_name] = self.rgb_cameras[camera_name]
                    print(f"  ✅ {camera_name} ready")
                
            elif info['type'] == 'realsense':
                # Start RealSense pipeline
                try:
                    rs_config = self.realsense_devices[camera_name]
                    pipeline = rs_config['pipeline']
                    config = rs_config['config']
                    
                    pipeline.start(config)
                    active_cameras[camera_name] = pipeline
                    print(f"  ✅ {camera_name} started")
                    
                except Exception as e:
                    print(f"  ❌ Failed to start {camera_name}: {e}")
        
        return active_cameras
    
    def capture_calibration_images(self, active_cameras):
        """Capture synchronized calibration images from all cameras"""
        print(f"\n{'='*60}")
        print("MULTI-CAMERA CALIBRATION IMAGE CAPTURE")
        print('='*60)
        print("Instructions:")
        print("1. Position checkerboard visible to ALL selected cameras")
        print("2. Press SPACE when pattern is detected in all cameras")
        print("3. Capture 15-20 good sets from different angles/distances")
        print("4. Press 'q' to finish and proceed to calibration")
        print('='*60)
        
        capture_count = 0
        
        # Initialize storage
        for camera_name in active_cameras:
            self.calibration_images[camera_name] = []
            self.image_points[camera_name] = []
        
        while True:
            # Capture frames from all cameras
            frames = {}
            pattern_found = {}
            corner_points = {}
            
            for camera_name, camera_obj in active_cameras.items():
                info = self.camera_info[camera_name]
                
                try:
                    if info['type'] == 'rgb':
                        # RGB camera
                        ret, frame = camera_obj.read()
                        if ret:
                            frames[camera_name] = frame
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        else:
                            continue
                            
                    elif info['type'] == 'realsense':
                        # RealSense device
                        rs_frames = camera_obj.wait_for_frames(timeout_ms=100)
                        
                        # Use IR frame for pattern detection (works for both D405 and D415)
                        ir_frame = rs_frames.get_infrared_frame(1)
                        if ir_frame:
                            ir_image = np.asanyarray(ir_frame.get_data())
                            frame = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
                            frames[camera_name] = frame
                            gray = ir_image
                        else:
                            continue
                    
                    # Find chessboard corners
                    ret_corners, corners = cv2.findChessboardCorners(
                        gray, self.checkerboard_size, None)
                    
                    pattern_found[camera_name] = ret_corners
                    if ret_corners:
                        # Refine corners
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        corner_points[camera_name] = corners_refined
                        
                        # Draw corners on frame
                        cv2.drawChessboardCorners(frame, self.checkerboard_size, corners_refined, ret_corners)
                        cv2.putText(frame, "PATTERN FOUND", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "No pattern", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    frames[camera_name] = frame
                    
                except Exception as e:
                    print(f"Frame capture error for {camera_name}: {e}")
                    continue
            
            # Check if all cameras found the pattern
            all_patterns_found = len(pattern_found) > 0 and all(pattern_found.values())
            
            # Create combined display
            if frames:
                display_frames = []
                for camera_name, frame in frames.items():
                    # Add camera name and status
                    cv2.putText(frame, camera_name, (10, frame.shape[0] - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Captured: {capture_count}", (10, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if all_patterns_found:
                        cv2.putText(frame, "READY - Press SPACE", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Resize for display
                    frame_resized = cv2.resize(frame, (320, 240))
                    display_frames.append(frame_resized)
                
                # Arrange frames in grid
                if len(display_frames) <= 2:
                    combined = np.hstack(display_frames)
                elif len(display_frames) <= 4:
                    top_row = np.hstack(display_frames[:2])
                    bottom_row = np.hstack(display_frames[2:4])
                    if len(display_frames) == 3:
                        bottom_row = np.hstack([display_frames[2], np.zeros_like(display_frames[2])])
                    combined = np.vstack([top_row, bottom_row])
                else:
                    # For more than 4 cameras, arrange in rows of 3
                    rows = []
                    for i in range(0, len(display_frames), 3):
                        row_frames = display_frames[i:i+3]
                        while len(row_frames) < 3:
                            row_frames.append(np.zeros_like(display_frames[0]))
                        rows.append(np.hstack(row_frames))
                    combined = np.vstack(rows)
                
                cv2.imshow('Multi-Camera Calibration', combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') and all_patterns_found:
                # Store calibration data
                for camera_name in active_cameras:
                    if camera_name in frames and camera_name in corner_points:
                        self.calibration_images[camera_name].append(frames[camera_name])
                        self.image_points[camera_name].append(corner_points[camera_name])
                
                # Generate object points (same for all cameras)
                if len(self.object_points) <= capture_count:
                    objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
                    objp *= self.square_size
                    self.object_points.append(objp)
                
                capture_count += 1
                print(f"✅ Captured set {capture_count} from {len(active_cameras)} cameras")
                
                # Save images
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for camera_name, frame in frames.items():
                    filename = self.calibration_path / f"{camera_name}_calib_{capture_count:03d}_{timestamp}.png"
                    cv2.imwrite(str(filename), frame)
        
        cv2.destroyAllWindows()
        print(f"\n📸 Captured {capture_count} image sets from {len(active_cameras)} cameras")
        return capture_count >= 10
    
    def perform_intrinsic_calibration(self):
        """Perform intrinsic calibration for each camera"""
        print(f"\n🔧 Performing intrinsic calibration for {len(self.image_points)} cameras...")
        
        for camera_name, image_points in self.image_points.items():
            if len(image_points) < 10:
                print(f"❌ {camera_name}: Not enough images ({len(image_points)})")
                continue
            
            print(f"📐 Calibrating {camera_name}...")
            
            # Get image dimensions from first calibration image
            first_image = self.calibration_images[camera_name][0]
            height, width = first_image.shape[:2]
            
            # Perform calibration
            ret, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
                self.object_points[:len(image_points)], image_points, 
                (width, height), None, None)
            
            # Store results
            self.intrinsic_results[camera_name] = {
                'camera_matrix': camera_matrix,
                'distortion': distortion,
                'rms_error': ret,
                'image_size': (width, height)
            }
            
            print(f"  ✅ {camera_name}: RMS error = {ret:.4f} pixels")
    
    def perform_stereo_calibration(self, camera_pairs=None):
        """Perform stereo calibration between camera pairs"""
        if not camera_pairs:
            # Generate all possible pairs
            cameras = list(self.intrinsic_results.keys())
            camera_pairs = [(cameras[i], cameras[j]) for i in range(len(cameras)) 
                           for j in range(i+1, len(cameras))]
        
        print(f"\n🔗 Performing stereo calibration for {len(camera_pairs)} pairs...")
        
        for cam1, cam2 in camera_pairs:
            if cam1 not in self.intrinsic_results or cam2 not in self.intrinsic_results:
                print(f"❌ Missing intrinsic calibration for {cam1} or {cam2}")
                continue
            
            print(f"🔗 Stereo calibrating {cam1} ↔ {cam2}...")
            
            # Get calibration data
            cam1_points = self.image_points[cam1]
            cam2_points = self.image_points[cam2]
            
            # Ensure same number of images
            min_images = min(len(cam1_points), len(cam2_points))
            cam1_points = cam1_points[:min_images]
            cam2_points = cam2_points[:min_images]
            object_points = self.object_points[:min_images]
            
            # Get intrinsic parameters
            cam1_matrix = self.intrinsic_results[cam1]['camera_matrix']
            cam1_dist = self.intrinsic_results[cam1]['distortion']
            cam2_matrix = self.intrinsic_results[cam2]['camera_matrix']
            cam2_dist = self.intrinsic_results[cam2]['distortion']
            
            # Image size
            image_size = self.intrinsic_results[cam1]['image_size']
            
            # Stereo calibration
            ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
                object_points, cam1_points, cam2_points,
                cam1_matrix, cam1_dist, cam2_matrix, cam2_dist,
                image_size, flags=cv2.CALIB_FIX_INTRINSIC)
            
            # Store results
            self.stereo_results[(cam1, cam2)] = {
                'rotation_matrix': R,
                'translation_vector': T,
                'rms_error': ret
            }
            
            print(f"  ✅ {cam1} ↔ {cam2}: RMS error = {ret:.4f} pixels")
    
    def save_all_calibrations(self):
        """Save all calibration results"""
        print(f"\n💾 Saving calibration results...")
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'checkerboard_size': self.checkerboard_size,
            'square_size': self.square_size,
            'camera_info': {},
            'intrinsic_calibrations': {},
            'stereo_calibrations': {}
        }
        
        # Convert camera info (remove non-serializable objects)
        for camera_name, info in self.camera_info.items():
            if camera_name in self.intrinsic_results:
                save_data['camera_info'][camera_name] = {
                    'type': info['type'],
                    'model': info.get('model', 'Unknown'),
                    'serial': info.get('serial', 'N/A'),
                    'id': info.get('id', 'N/A')
                }
        
        # Convert intrinsic results
        for camera_name, results in self.intrinsic_results.items():
            save_data['intrinsic_calibrations'][camera_name] = {
                'camera_matrix': results['camera_matrix'].tolist(),
                'distortion': results['distortion'].tolist(),
                'rms_error': results['rms_error'],
                'image_size': results['image_size']
            }
        
        # Convert stereo results
        for (cam1, cam2), results in self.stereo_results.items():
            pair_key = f"{cam1}_{cam2}"
            save_data['stereo_calibrations'][pair_key] = {
                'camera1': cam1,
                'camera2': cam2,
                'rotation_matrix': results['rotation_matrix'].tolist(),
                'translation_vector': results['translation_vector'].tolist(),
                'rms_error': results['rms_error']
            }
        
        # Save as JSON
        json_file = self.calibration_path / "multi_camera_calibration.json"
        with open(json_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Save as pickle
        pickle_file = self.calibration_path / "multi_camera_calibration.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ Calibration results saved:")
        print(f"   JSON: {json_file}")
        print(f"   Pickle: {pickle_file}")
        
        # Print summary
        self.print_calibration_summary()
    
    def print_calibration_summary(self):
        """Print calibration summary"""
        print(f"\n{'='*60}")
        print("CALIBRATION SUMMARY")
        print('='*60)
        
        print(f"📷 Intrinsic Calibrations ({len(self.intrinsic_results)}):")
        for camera_name, results in self.intrinsic_results.items():
            info = self.camera_info[camera_name]
            camera_type = f"{info['type']} - {info.get('model', 'Unknown')}"
            print(f"  • {camera_name} ({camera_type}): RMS = {results['rms_error']:.4f} px")
        
        print(f"\n🔗 Stereo Calibrations ({len(self.stereo_results)}):")
        for (cam1, cam2), results in self.stereo_results.items():
            print(f"  • {cam1} ↔ {cam2}: RMS = {results['rms_error']:.4f} px")
        
        print('='*60)
    
    def cleanup(self):
        """Clean up all camera resources"""
        print("🧹 Cleaning up cameras...")
        
        # Close RGB cameras
        for camera_name, cap in self.rgb_cameras.items():
            if cap:
                cap.release()
        
        # Stop RealSense pipelines
        for camera_name, rs_config in self.realsense_devices.items():
            try:
                pipeline = rs_config['pipeline']
                pipeline.stop()
            except:
                pass
        
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Multi-Camera Calibration System')
    parser.add_argument('--calibration-path', type=str, default='multi_camera_calibration',
                       help='Path to save calibration results')
    parser.add_argument('--discover-only', action='store_true',
                       help='Only discover cameras, don\'t calibrate')
    
    args = parser.parse_args()
    
    # Initialize calibration system
    calibrator = MultiCameraCalibrationSystem(args.calibration_path)
    
    try:
        # Discover all cameras
        calibrator.discover_cameras()
        
        if args.discover_only:
            return
        
        # Select cameras for calibration
        selected_cameras = calibrator.select_cameras_for_calibration()
        if not selected_cameras:
            print("❌ No cameras selected")
            return
        
        # Setup selected cameras
        active_cameras = calibrator.setup_selected_cameras(selected_cameras)
        if not active_cameras:
            print("❌ Failed to setup cameras")
            return
        
        # Capture calibration images
        if calibrator.capture_calibration_images(active_cameras):
            # Perform calibrations
            calibrator.perform_intrinsic_calibration()
            calibrator.perform_stereo_calibration()
            
            # Save results
            calibrator.save_all_calibrations()
            
            print("✅ Multi-camera calibration completed successfully!")
        else:
            print("❌ Not enough calibration images captured")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        calibrator.cleanup()

if __name__ == "__main__":
    main()