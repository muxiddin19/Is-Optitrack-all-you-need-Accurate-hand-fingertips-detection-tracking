import cv2
import os
import time
import argparse
import numpy as np

# Try to import pyrealsense2 for RealSense cameras, but don't fail if not available
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 module not found. RealSense camera support disabled.")

# Default capture parameters
DEFAULT_CHESSBOARD_SIZE = (7, 4)  # Number of inner corners per chessboard row and column
DEFAULT_OUTPUT_DIRECTORY = 'calibration_images'  # Directory to save calibration images

# Camera type constants
CAMERA_TYPE_WEBCAM = 'webcam'
CAMERA_TYPE_REALSENSE = 'realsense'

def list_available_cameras(max_to_check=10):
    """
    List all available camera devices.
    Returns a list of available camera IDs.
    """
    available_cameras = []
    
    print("Checking for available webcams...")
    for i in range(max_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                resolution = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                available_cameras.append((i, resolution, CAMERA_TYPE_WEBCAM))
            cap.release()
    
    # Check for RealSense cameras if the library is available
    if REALSENSE_AVAILABLE:
        print("Checking for RealSense cameras...")
        ctx = rs.context()
        devices = ctx.query_devices()
        device_serial_numbers = []
        
        for i in range(len(devices)):
            device = devices[i]
            serial_number = device.get_info(rs.camera_info.serial_number)
            device_serial_numbers.append(serial_number)
            
            # Add these as special IDs after the webcams
            camera_id = f"rs_{serial_number}"
            product_line = device.get_info(rs.camera_info.product_line)
            resolution = "Unknown (RealSense)"
            try:
                name = device.get_info(rs.camera_info.name)
                resolution = f"{name} ({product_line})"
            except:
                pass
            
            available_cameras.append((camera_id, resolution, CAMERA_TYPE_REALSENSE))
    
    return available_cameras

def initialize_realsense_pipeline(serial_number):
    """Initialize a RealSense pipeline for a specific device"""
    if not REALSENSE_AVAILABLE:
        print("Error: pyrealsense2 not available")
        return None, None
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable the specific device
    config.enable_device(serial_number)
    
    # Enable color stream - adjust resolution if needed
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline_profile = pipeline.start(config)
        return pipeline, pipeline_profile
    except Exception as e:
        print(f"Error initializing RealSense camera {serial_number}: {e}")
        return None, None

def capture_calibration_images(camera_id, chessboard_size=DEFAULT_CHESSBOARD_SIZE, 
                              output_directory=DEFAULT_OUTPUT_DIRECTORY,
                              adaptive_threshold=False,
                              normalize_image=False):
    """
    Capture images of a chessboard pattern for camera calibration.
    Run in an infinite loop until user presses 'q' or Escape to quit.
    Press 'c' to capture an image.
    
    Args:
        camera_id: ID of the camera to use (int for webcams, string like 'rs_123456' for RealSense)
        chessboard_size: Tuple containing the number of inner corners in the chessboard (width, height)
        output_directory: Directory to save calibration images
        adaptive_threshold: Whether to use adaptive thresholding to improve chessboard detection
        normalize_image: Whether to normalize the image to improve chessboard detection
    """
    # Determine camera type
    camera_type = CAMERA_TYPE_WEBCAM
    rs_serial = None
    if isinstance(camera_id, str) and camera_id.startswith('rs_'):
        camera_type = CAMERA_TYPE_REALSENSE
        rs_serial = camera_id[3:]  # Remove 'rs_' prefix
    
    # Create camera-specific output directory
    camera_output_dir = os.path.join(output_directory, f"camera_{camera_id}")
    if not os.path.exists(camera_output_dir):
        os.makedirs(camera_output_dir)
    
    # Open camera based on type
    cap = None
    pipeline = None
    
    if camera_type == CAMERA_TYPE_WEBCAM:
        cap = cv2.VideoCapture(int(camera_id))
        if not cap.isOpened():
            print(f"Error: Could not open webcam {camera_id}")
            return False
        
        # Get camera resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {camera_id} resolution: {width}x{height}")
        
    elif camera_type == CAMERA_TYPE_REALSENSE:
        if not REALSENSE_AVAILABLE:
            print(f"Error: RealSense support not available. Cannot open camera {camera_id}")
            return False
        
        pipeline, profile = initialize_realsense_pipeline(rs_serial)
        if not pipeline:
            print(f"Error: Could not initialize RealSense camera {rs_serial}")
            return False
        
        # Get stream profile details
        stream_profile = profile.get_stream(rs.stream.color)
        width = stream_profile.width()
        height = stream_profile.height()
        print(f"RealSense camera {rs_serial} resolution: {width}x{height}")
    
    # Counter for captured images
    img_counter = 0
    
    print(f"=== Calibrating Camera {camera_id} ===")
    print("Press 'c' to capture an image")
    print("Press 't' to toggle adaptive thresholding")
    print("Press 'n' to toggle image normalization")
    print("Press '+'/'-' to adjust brightness/contrast")
    print("Press 'q' or Escape to quit calibration for this camera")
    print(f"Images will be saved to {camera_output_dir}")
    
    # Image processing parameters
    brightness_adjustment = 0
    contrast_adjustment = 1.0
    
    while True:
        # Capture frame based on camera type
        frame = None
        if camera_type == CAMERA_TYPE_WEBCAM:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam")
                break
        elif camera_type == CAMERA_TYPE_REALSENSE:
            try:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    print("Error: Failed to get color frame from RealSense")
                    continue
                
                # Convert to numpy array for OpenCV
                frame = np.asanyarray(color_frame.get_data())
            except Exception as e:
                print(f"Error capturing RealSense frame: {e}")
                break
        
        # Apply brightness/contrast adjustments
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast_adjustment, beta=brightness_adjustment)
        
        # Convert to grayscale for chessboard detection
        gray = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply image preprocessing based on flags
        processed_gray = gray.copy()
        
        # Apply normalization if enabled
        if normalize_image:
            processed_gray = cv2.normalize(processed_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply adaptive thresholding if enabled
        if adaptive_threshold:
            processed_gray = cv2.adaptiveThreshold(
                processed_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        
        # Find chessboard corners with different flags for better detection
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret_chess, corners = cv2.findChessboardCorners(processed_gray, chessboard_size, flags)
        
        # If not found with the enhanced flags, try basic detection on original grayscale
        if not ret_chess:
            ret_chess, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # Create a copy of the frame for display
        display_frame = adjusted_frame.copy()
        
        # Draw corners if found
        if ret_chess:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(display_frame, chessboard_size, corners, ret_chess)
            cv2.putText(display_frame, "Chessboard detected!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No chessboard detected", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display camera ID and capture counter
        cv2.putText(display_frame, f"Camera ID: {camera_id}", (50, height - 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Captured: {img_counter}", (50, height - 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display processing settings
        cv2.putText(display_frame, f"Adaptive Threshold: {'ON' if adaptive_threshold else 'OFF'}", 
                   (50, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Normalize: {'ON' if normalize_image else 'OFF'}", 
                   (50, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Display the frame
        cv2.imshow(f'Camera {camera_id} Calibration', display_frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' or Escape to quit
        if key == ord('q') or key == 27:  # 27 is the ASCII code for Escape
            print(f"Finished calibration for camera {camera_id}")
            break
        
        # 't' to toggle adaptive thresholding
        elif key == ord('t'):
            adaptive_threshold = not adaptive_threshold
            print(f"Adaptive thresholding: {'ON' if adaptive_threshold else 'OFF'}")
        
        # 'n' to toggle normalization
        elif key == ord('n'):
            normalize_image = not normalize_image
            print(f"Image normalization: {'ON' if normalize_image else 'OFF'}")
        
        # '+' to increase brightness
        elif key == ord('+') or key == ord('='):
            brightness_adjustment += 5
            print(f"Brightness adjustment: {brightness_adjustment}")
        
        # '-' to decrease brightness
        elif key == ord('-'):
            brightness_adjustment -= 5
            print(f"Brightness adjustment: {brightness_adjustment}")
        
        # '[' to decrease contrast
        elif key == ord('['):
            contrast_adjustment = max(0.1, contrast_adjustment - 0.1)
            print(f"Contrast adjustment: {contrast_adjustment:.1f}")
        
        # ']' to increase contrast
        elif key == ord(']'):
            contrast_adjustment += 0.1
            print(f"Contrast adjustment: {contrast_adjustment:.1f}")
        
        # 'c' to capture
        elif key == ord('c'):
            # Only save if chessboard is detected
            if ret_chess:
                # Save both the original and annotated images
                img_name_original = os.path.join(camera_output_dir, f"calib_cam{camera_id}_img{img_counter:02d}.jpg")
                img_name_annotated = os.path.join(camera_output_dir, f"calib_cam{camera_id}_img{img_counter:02d}_annotated.jpg")
                
                cv2.imwrite(img_name_original, frame)
                cv2.imwrite(img_name_annotated, display_frame)
                
                print(f"Captured {img_name_original}")
                img_counter += 1
            else:
                print("No chessboard detected. Image not saved.")
    
    # Release resources based on camera type
    if camera_type == CAMERA_TYPE_WEBCAM and cap is not None:
        cap.release()
    elif camera_type == CAMERA_TYPE_REALSENSE and pipeline is not None:
        pipeline.stop()
    
    cv2.destroyAllWindows()
    
    print(f"Captured {img_counter} images for camera {camera_id}")
    return True

def camera_id_parser(camera_id_str):
    """Parse camera ID string and return appropriate type."""
    if camera_id_str.startswith('rs_'):
        return camera_id_str
    else:
        try:
            return int(camera_id_str)
        except ValueError:
            return camera_id_str

def main():
    """
    Main function to run the multi-camera calibration process.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Multi-camera Calibration Tool')
    parser.add_argument('--camera-ids', type=str, nargs='+', 
                        help='List of camera IDs to calibrate (e.g., 0 1 2 rs_123456)')
    parser.add_argument('--chessboard-width', type=int, default=DEFAULT_CHESSBOARD_SIZE[0],
                        help=f'Number of inner corners along width of chessboard (default: {DEFAULT_CHESSBOARD_SIZE[0]})')
    parser.add_argument('--chessboard-height', type=int, default=DEFAULT_CHESSBOARD_SIZE[1],
                        help=f'Number of inner corners along height of chessboard (default: {DEFAULT_CHESSBOARD_SIZE[1]})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIRECTORY,
                        help=f'Directory to save calibration images (default: {DEFAULT_OUTPUT_DIRECTORY})')
    parser.add_argument('--scan', action='store_true',
                        help='Scan and list available cameras')
    parser.add_argument('--adaptive-threshold', action='store_true',
                        help='Enable adaptive thresholding for better chessboard detection')
    parser.add_argument('--normalize', action='store_true',
                        help='Enable image normalization for better chessboard detection')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    chessboard_size = (args.chessboard_width, args.chessboard_height)
    
    # Scan for available cameras if requested
    if args.scan:
        available_cameras = list_available_cameras()
        if available_cameras:
            print("\nAvailable cameras:")
            for cam_id, resolution, cam_type in available_cameras:
                print(f"Camera ID: {cam_id} - Resolution: {resolution} - Type: {cam_type}")
            print("\nTo calibrate specific cameras, run:")
            print(f"python {os.path.basename(__file__)} --camera-ids", end=" ")
            print(" ".join(str(cam_id) for cam_id, _, _ in available_cameras))
        else:
            print("No cameras detected.")
        return
    
    # If no camera IDs are provided, ask user interactively
    camera_ids = args.camera_ids
    if not camera_ids:
        available_cameras = list_available_cameras()
        if not available_cameras:
            print("No cameras detected.")
            return
        
        print("\nAvailable cameras:")
        for cam_id, resolution, cam_type in available_cameras:
            print(f"Camera ID: {cam_id} - Resolution: {resolution} - Type: {cam_type}")
        
        try:
            input_str = input("\nEnter camera IDs to calibrate (space-separated, e.g., '0 1 2 rs_123456'): ")
            camera_ids = input_str.strip().split()
        except ValueError:
            print("Invalid input. Using default camera (ID 0).")
            camera_ids = ['0']
    
    # Convert camera IDs to appropriate types
    parsed_camera_ids = [camera_id_parser(cam_id) for cam_id in camera_ids]
    
    # Ensure we have at least one camera to calibrate
    if not parsed_camera_ids:
        print("No cameras specified for calibration.")
        return
    
    print(f"\nPreparing to calibrate {len(parsed_camera_ids)} camera(s): {parsed_camera_ids}")
    print(f"Using chessboard size: {chessboard_size}")
    print(f"Images will be saved to: {args.output_dir}")
    
    for camera_id in parsed_camera_ids:
        print(f"\n{'='*50}")
        print(f"Starting calibration for Camera {camera_id}")
        print(f"{'='*50}")
        
        success = capture_calibration_images(
            camera_id=camera_id,
            chessboard_size=chessboard_size,
            output_directory=args.output_dir,
            adaptive_threshold=args.adaptive_threshold,
            normalize_image=args.normalize
        )
        
        if not success:
            print(f"Failed to calibrate Camera {camera_id}. Skipping to next camera.")
    
    print("\nAll camera calibrations completed.")
    print(f"Calibration images have been saved to: {args.output_dir}")
    print("You can now use these images with OpenCV's calibration functions to compute camera parameters.")

if __name__ == "__main__":
    main()
