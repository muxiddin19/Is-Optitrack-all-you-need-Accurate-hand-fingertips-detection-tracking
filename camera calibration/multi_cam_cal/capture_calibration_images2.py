#!/usr/bin/env python3
"""
Fixed camera calibration image capture for both webcams and RealSense cameras
Handles the MSMF and RealSense API issues properly
"""

import cv2
import pyrealsense2 as rs
import numpy as np
import os
from pathlib import Path
import argparse
from datetime import datetime

class CalibrationImageCapture:
    """
    Robust calibration image capture supporting both webcams and RealSense
    """
    
    def __init__(self, camera_type='webcam', camera_id=0, realsense_serial=None):
        """
        Initialize camera capture
        
        Args:
            camera_type: 'webcam' or 'realsense'
            camera_id: Webcam ID (for webcam type)
            realsense_serial: Serial number for RealSense (for realsense type)
        """
        self.camera_type = camera_type
        self.camera_id = camera_id
        self.realsense_serial = realsense_serial
        
        # Checkerboard parameters
        self.checkerboard_size = (7, 4)  # Internal corners (width, height)
        self.square_size = 33  # mm
        
        # Initialize camera
        self.cap = None
        self.pipeline = None
        self.setup_camera()
        
    def setup_camera(self):
        """Setup camera based on type"""
        if self.camera_type == 'webcam':
            self.setup_webcam()
        elif self.camera_type == 'realsense':
            self.setup_realsense()
        else:
            raise ValueError(f"Unknown camera type: {self.camera_type}")
    
    def setup_webcam(self):
        """Setup webcam with proper error handling"""
        print(f"Setting up webcam {self.camera_id}...")
        
        # Try different backends to avoid MSMF issues
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.camera_id, backend)
                if self.cap.isOpened():
                    # Test if we can read a frame
                    ret, frame = self.cap.read()
                    if ret:
                        print(f"✅ Webcam {self.camera_id} opened successfully with backend {backend}")
                        
                        # Set properties
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        # Get actual resolution
                        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"Camera resolution: {width}x{height}")
                        
                        return
                    else:
                        self.cap.release()
                        self.cap = None
                else:
                    if self.cap:
                        self.cap.release()
                        self.cap = None
            except Exception as e:
                print(f"Failed to open webcam with backend {backend}: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        raise RuntimeError(f"Could not open webcam {self.camera_id} with any backend")
    
    def setup_realsense(self):
        """Setup RealSense camera"""
        print(f"Setting up RealSense camera...")
        
        try:
            # Create pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable specific device if serial provided
            if self.realsense_serial:
                config.enable_device(self.realsense_serial)
                print(f"Using RealSense device: {self.realsense_serial}")
            
            # Configure streams
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.pipeline.start(config)
            
            # Get device info
            device = profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            device_serial = device.get_info(rs.camera_info.serial_number)
            
            print(f"✅ RealSense opened: {device_name} ({device_serial})")
            
            # Create alignment object
            self.align = rs.align(rs.stream.color)
            
        except Exception as e:
            print(f"❌ Failed to setup RealSense: {e}")
            raise
    
    def get_frame(self):
        """Get frame from camera based on type"""
        if self.camera_type == 'webcam':
            return self.get_webcam_frame()
        elif self.camera_type == 'realsense':
            return self.get_realsense_frame()
    
    def get_webcam_frame(self):
        """Get frame from webcam"""
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def get_realsense_frame(self):
        """Get frame from RealSense"""
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            
            # Align frames
            aligned_frames = self.align.process(frames)
            
            # Get color frame
            color_frame = aligned_frames.get_color_frame()
            
            if not color_frame:
                return False, None
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            return True, color_image
            
        except Exception as e:
            print(f"Error getting RealSense frame: {e}")
            return False, None
    
    def find_corners(self, image):
        """Find checkerboard corners in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find corners with different flags for robustness
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_NORMALIZE_IMAGE + 
                cv2.CALIB_CB_FAST_CHECK)
        
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags)
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return ret, corners
    
    def capture_calibration_images(self, output_dir="calibration_images", target_images=20):
        """
        Interactive calibration image capture
        
        Args:
            output_dir: Directory to save images
            target_images: Target number of images to capture
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n=== Calibration Image Capture ===")
        print(f"Camera type: {self.camera_type}")
        print(f"Target images: {target_images}")
        print(f"Checkerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} corners")
        print(f"Output directory: {output_path.absolute()}")
        print("\nControls:")
        print("  SPACE - Capture image (when corners detected)")
        print("  ESC   - Exit")
        print("  'c'   - Force capture without corner detection")
        print("\nTips for good calibration:")
        print("  • Move checkerboard to different positions")
        print("  • Include corners, edges, and center of image")
        print("  • Vary distance and orientation")
        print("  • Ensure good lighting and focus")
        print("\nPress any key to start...")
        cv2.waitKey(0)
        
        captured_count = 0
        
        try:
            while captured_count < target_images:
                # Get frame
                ret, frame = self.get_frame()
                
                if not ret or frame is None:
                    print("Failed to get frame, retrying...")
                    cv2.waitKey(100)
                    continue
                
                # Find checkerboard corners
                corners_found, corners = self.find_corners(frame)
                
                # Create display frame
                display_frame = frame.copy()
                
                # Draw corners if found
                if corners_found:
                    cv2.drawChessboardCorners(display_frame, self.checkerboard_size, corners, corners_found)
                    status_text = "CORNERS DETECTED - Press SPACE to capture"
                    status_color = (0, 255, 0)
                else:
                    status_text = "CORNERS NOT DETECTED - Move checkerboard"
                    status_color = (0, 0, 255)
                
                # Add status text
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Add capture count
                cv2.putText(display_frame, f"Captured: {captured_count}/{target_images}", 
                           (10, display_frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add controls
                cv2.putText(display_frame, "SPACE: Capture, ESC: Exit, C: Force capture", 
                           (10, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Calibration Capture', display_frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("Capture cancelled by user")
                    break
                elif key == ord(' ') and corners_found:
                    # Capture with corner detection
                    self.save_image(frame, output_path, captured_count)
                    captured_count += 1
                    print(f"✅ Captured image {captured_count}/{target_images}")
                    cv2.waitKey(500)  # Brief pause
                elif key == ord('c'):
                    # Force capture without corner detection
                    self.save_image(frame, output_path, captured_count, force=True)
                    captured_count += 1
                    print(f"⚠️  Force captured image {captured_count}/{target_images} (no corners)")
                    cv2.waitKey(500)  # Brief pause
        
        except KeyboardInterrupt:
            print("\nCapture interrupted by user")
        except Exception as e:
            print(f"Error during capture: {e}")
        finally:
            cv2.destroyAllWindows()
        
        print(f"\n=== Capture Summary ===")
        print(f"Total images captured: {captured_count}")
        print(f"Images saved to: {output_path.absolute()}")
        
        if captured_count >= 10:
            print("✅ Sufficient images for calibration")
        else:
            print("⚠️  Recommendation: Capture at least 15-20 images for good calibration")
        
        return captured_count
    
    def save_image(self, image, output_path, count, force=False):
        """Save calibration image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_{count:03d}_{timestamp}.png"
        filepath = output_path / filename
        
        success = cv2.imwrite(str(filepath), image)
        if not success:
            print(f"❌ Failed to save {filename}")
        
        return success
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()

def list_available_cameras():
    """List all available cameras (webcams and RealSense)"""
    print("=== Available Cameras ===")
    
    # Check webcams
    print("\nWebcams:")
    webcam_count = 0
    for i in range(5):  # Check first 5 camera IDs
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow to避免MSMF issues
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"  Camera {i}: {width}x{height}")
                    webcam_count += 1
                cap.release()
            else:
                cap.release()
        except:
            pass
    
    if webcam_count == 0:
        print("  No webcams found")
    
    # Check RealSense cameras
    print("\nRealSense cameras:")
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("  No RealSense cameras found")
        else:
            for i, device in enumerate(devices):
                name = device.get_info(rs.camera_info.name)
                serial = device.get_info(rs.camera_info.serial_number)
                print(f"  RealSense {i}: {name} (Serial: {serial})")
    except Exception as e:
        print(f"  Error checking RealSense cameras: {e}")

def main():
    parser = argparse.ArgumentParser(description='Calibration Image Capture')
    parser.add_argument('--type', choices=['webcam', 'realsense'], default='webcam',
                       help='Camera type')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Webcam ID (for webcam type)')
    parser.add_argument('--realsense-serial', type=str,
                       help='RealSense serial number (for realsense type)')
    parser.add_argument('--output-dir', type=str, default='calibration_images',
                       help='Output directory for images')
    parser.add_argument('--num-images', type=int, default=20,
                       help='Target number of images to capture')
    parser.add_argument('--list-cameras', action='store_true',
                       help='List available cameras and exit')
    
    args = parser.parse_args()
    
    if args.list_cameras:
        list_available_cameras()
        return
    
    # Initialize capture
    try:
        capturer = CalibrationImageCapture(
            camera_type=args.type,
            camera_id=args.camera_id,
            realsense_serial=args.realsense_serial
        )
        
        # Capture images
        capturer.capture_calibration_images(args.output_dir, args.num_images)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'capturer' in locals():
            capturer.cleanup()

if __name__ == "__main__":
    main()