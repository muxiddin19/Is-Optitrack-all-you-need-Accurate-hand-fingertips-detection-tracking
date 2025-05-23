import cv2
import numpy as np
import os
import pickle
import json
import argparse
from pathlib import Path
import pyrealsense2 as rs

class StereoCalibrationTool:
    """
    Stereo calibration tool for RGB camera + RealSense D405 depth camera
    """
    
    def __init__(self, calibration_path="stereo_calibration_results"):
        self.calibration_path = Path(calibration_path)
        self.calibration_path.mkdir(exist_ok=True)
        
        # Camera objects
        self.rgb_cap = None
        self.rs_pipeline = None
        self.rs_config = None
        
        # Calibration parameters
        self.checkerboard_size = (7, 4)  # Internal corners (width, height)
        self.square_size = 0.033  # 25mm squares
        
        # Storage for calibration data
        self.rgb_image_points = []
        self.depth_image_points = []
        self.object_points = []
        
        # Calibration results
        self.rgb_camera_matrix = None
        self.rgb_distortion = None
        self.depth_camera_matrix = None
        self.depth_distortion = None
        self.rotation_matrix = None
        self.translation_vector = None
        
    def setup_cameras(self, rgb_camera_id=0):
        """Setup RGB camera and RealSense D405"""
        print("🔍 Setting up cameras...")
        
        # Setup RGB camera
        self.rgb_cap = cv2.VideoCapture(rgb_camera_id)
        if not self.rgb_cap.isOpened():
            print(f"❌ Failed to open RGB camera {rgb_camera_id}")
            return False
        
        # Set RGB camera resolution (adjust as needed)
        self.rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Setup RealSense D405
        try:
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()
            
            # Configure streams - D405 has infrared cameras for depth
            self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.rs_config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # Left IR
            
            # Start pipeline
            profile = self.rs_pipeline.start(self.rs_config)
            
            # Get depth sensor intrinsics
            depth_profile = profile.get_stream(rs.stream.depth)
            depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            
            print(f"✅ RealSense D405 initialized")
            print(f"   Depth resolution: {depth_intrinsics.width}x{depth_intrinsics.height}")
            print(f"   Depth focal length: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}")
            
        except Exception as e:
            print(f"❌ Failed to initialize RealSense: {e}")
            return False
        
        print("✅ Both cameras ready for calibration")
        return True
    
    def capture_calibration_images(self):
        """Capture synchronized images from both cameras"""
        print(f"\n{'='*50}")
        print("STEREO CALIBRATION IMAGE CAPTURE")
        print('='*50)
        print("Instructions:")
        print("1. Position checkerboard in view of BOTH cameras")
        print("2. Press SPACE to capture when both cameras detect the pattern")
        print("3. Capture at least 15-20 good pairs from different angles")
        print("4. Press 'q' to finish and proceed to calibration")
        print('='*50)
        
        capture_count = 0
        
        while True:
            # Get RGB frame
            ret_rgb, rgb_frame = self.rgb_cap.read()
            if not ret_rgb:
                print("❌ Failed to capture RGB frame")
                continue
            
            # Get RealSense frames
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=1000)
                ir_frame = frames.get_infrared_frame(1)  # Left IR camera
                
                if not ir_frame:
                    print("❌ Failed to get IR frame")
                    continue
                
                # Convert IR frame to numpy array
                ir_image = np.asanyarray(ir_frame.get_data())
                ir_image_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
                
            except Exception as e:
                print(f"❌ RealSense frame error: {e}")
                continue
            
            # Find chessboard corners in both images
            gray_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            gray_ir = ir_image
            
            # Find corners
            ret_rgb_corners, corners_rgb = cv2.findChessboardCorners(
                gray_rgb, self.checkerboard_size, None)
            ret_ir_corners, corners_ir = cv2.findChessboardCorners(
                gray_ir, self.checkerboard_size, None)
            
            # Create display frames
            display_rgb = rgb_frame.copy()
            display_ir = ir_image_bgr.copy()
            
            # Draw corners if found
            pattern_found_both = False
            if ret_rgb_corners:
                cv2.drawChessboardCorners(display_rgb, self.checkerboard_size, 
                                        corners_rgb, ret_rgb_corners)
                cv2.putText(display_rgb, "RGB: PATTERN FOUND", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_rgb, "RGB: No pattern", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if ret_ir_corners:
                cv2.drawChessboardCorners(display_ir, self.checkerboard_size, 
                                        corners_ir, ret_ir_corners)
                cv2.putText(display_ir, "IR: PATTERN FOUND", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_ir, "IR: No pattern", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Check if both cameras found the pattern
            if ret_rgb_corners and ret_ir_corners:
                pattern_found_both = True
                cv2.putText(display_rgb, "READY TO CAPTURE - Press SPACE", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_ir, "READY TO CAPTURE - Press SPACE", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add capture count
            cv2.putText(display_rgb, f"Captured: {capture_count}", (10, display_rgb.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_ir, f"Captured: {capture_count}", (10, display_ir.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display images side by side
            combined = np.hstack([display_rgb, display_ir])
            cv2.imshow('Stereo Calibration - RGB (Left) | IR (Right)', combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') and pattern_found_both:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_rgb_refined = cv2.cornerSubPix(gray_rgb, corners_rgb, (11, 11), (-1, -1), criteria)
                corners_ir_refined = cv2.cornerSubPix(gray_ir, corners_ir, (11, 11), (-1, -1), criteria)
                
                # Store the points
                self.rgb_image_points.append(corners_rgb_refined)
                self.depth_image_points.append(corners_ir_refined)
                
                # Generate object points (3D points of checkerboard)
                objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
                objp *= self.square_size
                self.object_points.append(objp)
                
                capture_count += 1
                print(f"✅ Captured pair {capture_count}")
                
                # Save the images for reference
                rgb_filename = self.calibration_path / f"rgb_calib_{capture_count:03d}.png"
                ir_filename = self.calibration_path / f"ir_calib_{capture_count:03d}.png"
                cv2.imwrite(str(rgb_filename), rgb_frame)
                cv2.imwrite(str(ir_filename), ir_image)
        
        cv2.destroyAllWindows()
        print(f"\n📸 Captured {capture_count} image pairs")
        return capture_count >= 10  # Minimum pairs needed
    
    def perform_calibration(self):
        """Perform stereo calibration"""
        if len(self.object_points) < 10:
            print("❌ Not enough calibration images. Need at least 10 pairs.")
            return False
        
        print(f"\n🔧 Performing stereo calibration with {len(self.object_points)} image pairs...")
        
        # Get image dimensions
        rgb_ret, rgb_frame = self.rgb_cap.read()
        rgb_height, rgb_width = rgb_frame.shape[:2]
        
        # For IR camera, use same dimensions (D405 IR and depth have same resolution)
        ir_height, ir_width = rgb_height, rgb_width
        
        # Step 1: Calibrate each camera individually
        print("📐 Calibrating RGB camera...")
        ret_rgb, self.rgb_camera_matrix, self.rgb_distortion, _, _ = cv2.calibrateCamera(
            self.object_points, self.rgb_image_points, (rgb_width, rgb_height), None, None)
        
        print("📐 Calibrating IR/Depth camera...")
        ret_ir, self.depth_camera_matrix, self.depth_distortion, _, _ = cv2.calibrateCamera(
            self.object_points, self.depth_image_points, (ir_width, ir_height), None, None)
        
        print(f"✅ RGB calibration RMS error: {ret_rgb:.4f}")
        print(f"✅ IR calibration RMS error: {ret_ir:.4f}")
        
        # Step 2: Stereo calibration
        print("🔗 Performing stereo calibration...")
        
        # Stereo calibration flags
        flags = cv2.CALIB_FIX_INTRINSIC  # Fix individual camera calibrations
        
        ret_stereo, _, _, _, _, self.rotation_matrix, self.translation_vector, _, _ = cv2.stereoCalibrate(
            self.object_points, self.rgb_image_points, self.depth_image_points,
            self.rgb_camera_matrix, self.rgb_distortion,
            self.depth_camera_matrix, self.depth_distortion,
            (rgb_width, rgb_height), flags=flags)
        
        print(f"✅ Stereo calibration RMS error: {ret_stereo:.4f}")
        
        # Save calibration results
        self.save_calibration_results()
        
        return ret_stereo < 1.0  # Consider good if < 1 pixel error
    
    def save_calibration_results(self):
        """Save all calibration results"""
        print("💾 Saving calibration results...")
        
        # Create calibration data dictionary
        calibration_data = {
            'rgb_camera_matrix': self.rgb_camera_matrix.tolist(),
            'rgb_distortion': self.rgb_distortion.tolist(),
            'depth_camera_matrix': self.depth_camera_matrix.tolist(),
            'depth_distortion': self.depth_distortion.tolist(),
            'rotation_matrix': self.rotation_matrix.tolist(),
            'translation_vector': self.translation_vector.tolist(),
            'checkerboard_size': self.checkerboard_size,
            'square_size': self.square_size,
            'num_images': len(self.object_points)
        }
        
        # Save as JSON
        json_file = self.calibration_path / "stereo_calibration.json"
        with open(json_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        # Save as pickle for easy loading
        pickle_file = self.calibration_path / "stereo_calibration.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print(f"✅ Calibration saved to:")
        print(f"   JSON: {json_file}")
        print(f"   Pickle: {pickle_file}")
        
        # Print summary
        print(f"\n{'='*50}")
        print("CALIBRATION SUMMARY")
        print('='*50)
        print(f"RGB Camera Matrix:")
        print(self.rgb_camera_matrix)
        print(f"\nDepth Camera Matrix:")
        print(self.depth_camera_matrix)
        print(f"\nRotation Matrix (Depth to RGB):")
        print(self.rotation_matrix)
        print(f"\nTranslation Vector (Depth to RGB):")
        print(self.translation_vector.ravel())
        print('='*50)
    
    def test_calibration(self):
        """Test calibration by showing live alignment"""
        if self.rotation_matrix is None:
            print("❌ No calibration data available. Run calibration first.")
            return
        
        print(f"\n{'='*50}")
        print("TESTING STEREO CALIBRATION")
        print('='*50)
        print("This will show live alignment between RGB and depth.")
        print("Press 'q' to quit test.")
        print('='*50)
        
        while True:
            # Get RGB frame
            ret_rgb, rgb_frame = self.rgb_cap.read()
            if not ret_rgb:
                continue
            
            # Get RealSense frames
            try:
                frames = self.rs_pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                ir_frame = frames.get_infrared_frame(1)
                
                if not depth_frame or not ir_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                ir_image = np.asanyarray(ir_frame.get_data())
                
                # Convert depth to colormap for visualization
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Display all three images
                ir_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
                
                # Resize to same size if needed
                height, width = rgb_frame.shape[:2]
                depth_colormap = cv2.resize(depth_colormap, (width, height))
                ir_bgr = cv2.resize(ir_bgr, (width, height))
                
                # Combine images
                top_row = np.hstack([rgb_frame, ir_bgr])
                bottom_row = np.hstack([depth_colormap, np.zeros_like(rgb_frame)])
                combined = np.vstack([top_row, bottom_row])
                
                # Add labels
                cv2.putText(combined, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, "IR", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, "Depth", (10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('Calibration Test - RGB | IR | Depth', combined)
                
            except Exception as e:
                print(f"❌ Frame error: {e}")
                continue
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def cleanup(self):
        """Clean up resources"""
        if self.rgb_cap:
            self.rgb_cap.release()
        if self.rs_pipeline:
            self.rs_pipeline.stop()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Stereo Calibration Tool for RGB + RealSense D405')
    parser.add_argument('--rgb-camera', type=int, default=0, help='RGB camera ID')
    parser.add_argument('--calibration-path', type=str, default='stereo_calibration_results',
                       help='Path to save calibration results')
    parser.add_argument('--test-only', action='store_true', help='Test existing calibration')
    
    args = parser.parse_args()
    
    # Initialize calibration tool
    calibrator = StereoCalibrationTool(args.calibration_path)
    
    try:
        if not calibrator.setup_cameras(args.rgb_camera):
            print("❌ Failed to setup cameras")
            return
        
        if args.test_only:
            # Load existing calibration
            pickle_file = Path(args.calibration_path) / "stereo_calibration.pkl"
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                
                calibrator.rgb_camera_matrix = np.array(data['rgb_camera_matrix'])
                calibrator.rgb_distortion = np.array(data['rgb_distortion'])
                calibrator.depth_camera_matrix = np.array(data['depth_camera_matrix'])
                calibrator.depth_distortion = np.array(data['depth_distortion'])
                calibrator.rotation_matrix = np.array(data['rotation_matrix'])
                calibrator.translation_vector = np.array(data['translation_vector'])
                
                calibrator.test_calibration()
            else:
                print("❌ No existing calibration found")
        else:
            # Full calibration process
            if calibrator.capture_calibration_images():
                if calibrator.perform_calibration():
                    print("✅ Calibration completed successfully!")
                    calibrator.test_calibration()
                else:
                    print("❌ Calibration failed or has high error")
            else:
                print("❌ Not enough calibration images captured")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        calibrator.cleanup()

if __name__ == "__main__":
    main()