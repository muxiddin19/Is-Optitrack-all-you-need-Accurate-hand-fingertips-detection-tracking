import cv2
import numpy as np
import os
import pickle
import json
import argparse
from pathlib import Path

class MultiCameraUndistortion:
    """
    Live undistortion tool for multiple cameras with their respective calibrations
    """
    
    def __init__(self, calibration_base_path="calibration_results"):
        self.calibration_base_path = Path(calibration_base_path)
        self.camera_configs = {}
        self.current_camera = None
        self.current_cap = None
        self.undistortion_maps = {}
        self.correction_enabled = True
        
        # Load all available camera calibrations
        self.load_camera_calibrations()
    
    def load_camera_calibrations(self):
        """Load calibration data for all available cameras"""
        print("🔍 Scanning for camera calibrations...")
        
        if not self.calibration_base_path.exists():
            print(f"❌ Calibration directory not found: {self.calibration_base_path}")
            return
        
        # Look for camera folders
        camera_folders = [d for d in self.calibration_base_path.iterdir() 
                         if d.is_dir() and d.name.startswith('camera_')]
        
        for camera_folder in camera_folders:
            camera_name = camera_folder.name
            
            # Try to load calibration data (pickle format)
            pkl_file = camera_folder / "calibration_data.pkl"
            json_file = camera_folder / "calibration_data.json"
            
            calibration_data = None
            
            if pkl_file.exists():
                try:
                    with open(pkl_file, 'rb') as f:
                        calibration_data = pickle.load(f)
                    print(f"✅ Loaded {camera_name} calibration (pickle)")
                except Exception as e:
                    print(f"❌ Failed to load {camera_name} pickle: {e}")
            
            elif json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    calibration_data = {
                        'camera_matrix': np.array(data['camera_matrix']),
                        'distortion_coefficients': np.array(data['distortion_coefficients']),
                        'reprojection_error': data.get('reprojection_error', 0)
                    }
                    print(f"✅ Loaded {camera_name} calibration (JSON)")
                except Exception as e:
                    print(f"❌ Failed to load {camera_name} JSON: {e}")
            
            if calibration_data:
                # Extract camera ID from folder name (e.g., camera_2 -> 2)
                try:
                    camera_id = int(camera_name.split('_')[1])
                except:
                    camera_id = 0
                
                self.camera_configs[camera_name] = {
                    'camera_id': camera_id,
                    'calibration_data': calibration_data,
                    'folder_path': camera_folder
                }
        
        if not self.camera_configs:
            print("❌ No camera calibrations found!")
        else:
            print(f"📷 Found calibrations for: {list(self.camera_configs.keys())}")
    
    def list_available_cameras(self):
        """List all available calibrated cameras"""
        print(f"\n{'='*50}")
        print("AVAILABLE CALIBRATED CAMERAS")
        print('='*50)
        
        for i, (camera_name, config) in enumerate(self.camera_configs.items()):
            calib = config['calibration_data']
            print(f"{i+1}. {camera_name}")
            print(f"   Camera ID: {config['camera_id']}")
            print(f"   Reprojection Error: {calib.get('reprojection_error', 'N/A'):.4f} pixels")
            print(f"   Calibration Path: {config['folder_path']}")
            print()
    
    def test_camera_access(self, camera_id):
        """Test if camera can be accessed"""
        print(f"🔍 Testing camera {camera_id} access...")
        
        # Try different backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        cap.release()
                        print(f"✅ Camera {camera_id} accessible: {width}x{height}")
                        return True
                cap.release()
            except Exception as e:
                print(f"❌ Backend failed: {e}")
        
        print(f"❌ Camera {camera_id} not accessible")
        return False
    
    def setup_camera(self, camera_name):
        """Setup specific camera for live undistortion"""
        if camera_name not in self.camera_configs:
            print(f"❌ Camera {camera_name} not found in calibrations")
            return False
        
        config = self.camera_configs[camera_name]
        camera_id = config['camera_id']
        
        # Test camera access first
        if not self.test_camera_access(camera_id):
            return False
        
        # Close previous camera if any
        if self.current_cap:
            self.current_cap.release()
        
        # Open camera with best backend
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                self.current_cap = cv2.VideoCapture(camera_id, backend)
                if self.current_cap.isOpened():
                    ret, frame = self.current_cap.read()
                    if ret and frame is not None:
                        print(f"✅ Opened camera {camera_id} with backend {backend}")
                        break
                self.current_cap.release()
                self.current_cap = None
            except Exception as e:
                print(f"Backend {backend} failed: {e}")
        
        if not self.current_cap:
            print(f"❌ Failed to open camera {camera_id}")
            return False
        
        # Setup undistortion maps
        self.current_camera = camera_name
        self.setup_undistortion_maps()
        
        return True
    
    def setup_undistortion_maps(self):
        """Setup undistortion maps for current camera"""
        config = self.camera_configs[self.current_camera]
        calib = config['calibration_data']
        
        mtx = calib['camera_matrix']
        dist = calib['distortion_coefficients']
        
        # Get camera resolution
        width = int(self.current_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.current_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📐 Camera resolution: {width}x{height}")
        print(f"🎯 Reprojection error: {calib.get('reprojection_error', 'N/A')} pixels")
        
        # Calculate optimal camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
        
        # Create undistortion maps
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width, height), 5)
        
        self.undistortion_maps[self.current_camera] = {
            'mapx': mapx,
            'mapy': mapy,
            'roi': roi,
            'original_size': (width, height)
        }
        
        print(f"✅ Undistortion maps created for {self.current_camera}")
    
    def run_live_undistortion(self, camera_name):
        """Run live undistortion for specific camera"""
        if not self.setup_camera(camera_name):
            return
        
        print(f"\n{'='*50}")
        print(f"LIVE UNDISTORTION - {camera_name.upper()}")
        print('='*50)
        print("Controls:")
        print("  'd' - Toggle distortion correction ON/OFF")
        print("  's' - Save current frame")
        print("  'i' - Show camera info")
        print("  'q' - Quit")
        print("  ESC - Quit")
        print('='*50)
        
        maps = self.undistortion_maps[self.current_camera]
        frame_count = 0
        
        while True:
            ret, frame = self.current_cap.read()
            
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Create display frame
            if self.correction_enabled:
                # Apply undistortion
                undistorted = cv2.remap(frame, maps['mapx'], maps['mapy'], cv2.INTER_LINEAR)
                
                # Crop the image
                x, y, w, h = maps['roi']
                if w > 0 and h > 0:
                    undistorted = undistorted[y:y+h, x:x+w]
                    # Resize back to original size for consistent display
                    undistorted = cv2.resize(undistorted, maps['original_size'])
                
                display_frame = undistorted
                status_text = "UNDISTORTED"
                status_color = (0, 255, 0)
            else:
                display_frame = frame
                status_text = "ORIGINAL"
                status_color = (0, 0, 255)
            
            # Add overlay information
            self.add_overlay(display_frame, camera_name, status_text, status_color, frame_count)
            
            # Display frame
            cv2.imshow(f'Live Undistortion - {camera_name}', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('d'):
                self.correction_enabled = not self.correction_enabled
                print(f"🔄 Distortion correction: {'ON' if self.correction_enabled else 'OFF'}")
            elif key == ord('s'):
                self.save_frame(display_frame, camera_name)
            elif key == ord('i'):
                self.show_camera_info(camera_name)
        
        self.cleanup()
    
    def add_overlay(self, frame, camera_name, status_text, status_color, frame_count):
        """Add information overlay to frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Camera name and status
        cv2.putText(frame, f"{camera_name.upper()}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (w-150, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls hint
        cv2.putText(frame, "Press 'i' for info, 'd' to toggle, 'q' to quit", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def save_frame(self, frame, camera_name):
        """Save current frame"""
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "undistorted" if self.correction_enabled else "original"
        filename = f"frame_{camera_name}_{status}_{timestamp}.png"
        
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"💾 Saved frame: {filename}")
        else:
            print(f"❌ Failed to save frame: {filename}")
    
    def show_camera_info(self, camera_name):
        """Display detailed camera information"""
        config = self.camera_configs[camera_name]
        calib = config['calibration_data']
        
        print(f"\n{'='*40}")
        print(f"CAMERA INFO - {camera_name.upper()}")
        print('='*40)
        print(f"Camera ID: {config['camera_id']}")
        print(f"Reprojection Error: {calib.get('reprojection_error', 'N/A')} pixels")
        print(f"Camera Matrix:")
        print(calib['camera_matrix'])
        print(f"Distortion Coefficients:")
        print(calib['distortion_coefficients'].ravel())
        
        # Current camera settings
        if self.current_cap:
            width = int(self.current_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.current_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.current_cap.get(cv2.CAP_PROP_FPS))
            print(f"Current Resolution: {width}x{height}")
            print(f"Current FPS: {fps}")
        print('='*40)
    
    def run_camera_selector(self):
        """Interactive camera selection"""
        if not self.camera_configs:
            print("❌ No calibrated cameras found!")
            return
        
        while True:
            self.list_available_cameras()
            
            try:
                choice = input("Select camera number (or 'q' to quit): ").strip()
                
                if choice.lower() == 'q':
                    break
                
                camera_index = int(choice) - 1
                camera_names = list(self.camera_configs.keys())
                
                if 0 <= camera_index < len(camera_names):
                    selected_camera = camera_names[camera_index]
                    print(f"🎯 Selected: {selected_camera}")
                    self.run_live_undistortion(selected_camera)
                else:
                    print("❌ Invalid selection!")
                    
            except ValueError:
                print("❌ Please enter a valid number!")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    
    def cleanup(self):
        """Clean up resources"""
        if self.current_cap:
            self.current_cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Multi-Camera Live Undistortion Tool')
    parser.add_argument('--calibration-path', type=str, default='calibration_results',
                       help='Path to calibration results directory')
    parser.add_argument('--camera', type=str, 
                       help='Specific camera to test (e.g., camera_2)')
    parser.add_argument('--list-cameras', action='store_true',
                       help='List available calibrated cameras and exit')
    
    args = parser.parse_args()
    
    # Initialize undistortion tool
    undistorter = MultiCameraUndistortion(args.calibration_path)
    
    try:
        if args.list_cameras:
            undistorter.list_available_cameras()
        elif args.camera:
            if args.camera in undistorter.camera_configs:
                undistorter.run_live_undistortion(args.camera)
            else:
                print(f"❌ Camera '{args.camera}' not found in calibrations")
                undistorter.list_available_cameras()
        else:
            # Interactive mode
            undistorter.run_camera_selector()
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        undistorter.cleanup()

if __name__ == "__main__":
    main()