def run_demo():
    # Define some constants
    resolution_width = 640  # Lower resolution for better performance
    resolution_height = 480  # Lower resolution for better performance
    frame_rate = 15  # fps

    dispose_frames_for_stablisation = 30  # frames

    chessboard_width = 4  # squares
    chessboard_height = 7  # squares
    square_size = 0.033  # meters

    # Define webcam IDs - start with empty list for auto-detection
    webcam_ids = []  # Auto-detect available webcams

    try:
        # Initialize RealSense device manager
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
        rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

        device_manager = DeviceManager(rs.context(), rs_config)
        device_manager.enable_all_devices()

        # Initialize webcam manager
        webcam_manager = WebcamManager(webcam_ids)
        webcam_manager.enable_all_webcams(resolution_width, resolution_height)

        # If no webcams were successfully enabled, continue with just RealSense
        if len(webcam_manager._enabled_webcams) == 0:
            print("No webcams were successfully enabled. Continuing with RealSense only.")

        # Allow some frames for the auto-exposure controller to stabilize
        print("Stabilizing cameras...")
        for frame in range(dispose_frames_for_stablisation):
            rs_frames = device_manager.poll_frames()
            if len(webcam_manager._enabled_webcams) > 0:
                webcam_frames = webcam_manager.poll_frames()
            time.sleep(0.1)  # Short delay to allow cameras to stabilize

        # Verify we have at least one RealSense device
        assert len(device_manager._available_devices) > 0, "No RealSense devices found"
        print(f"Found {len(device_manager._available_devices)} RealSense devices and {len(webcam_manager._enabled_webcams)} working webcams")

        """
        1: Calibration for RealSense cameras
        """
        print("\n--- Starting RealSense calibration ---")
        print("Place the chessboard where all RealSense cameras can see it clearly")
        
        # Get the intrinsics of the RealSense devices
        rs_frames = device_manager.poll_frames()
        intrinsics_devices = device_manager.get_device_intrinsics(rs_frames)

        # Set the chessboard parameters for calibration
        chessboard_params = [chessboard_height, chessboard_width, square_size]

        # Estimate the pose of the chessboard in the world coordinate using the Kabsch Method for RealSense
        calibrated_device_count = 0
        calibration_attempts = 0
        max_attempts = 100  # Prevent infinite loop
        
        print("Searching for chessboard pattern...")
        while calibrated_device_count < len(device_manager._available_devices) and calibration_attempts < max_attempts:
            calibration_attempts += 1
            rs_frames = device_manager.poll_frames()
            
            # Display the color frames from RealSense during calibration
            for device_info in device_manager._available_devices:
                device = device_info[0]
                if device in rs_frames and rs.stream.color in rs_frames[device]:
                    color_frame = rs_frames[device][rs.stream.color]
                    color_image = np.asanyarray(color_frame.get_data())
                    cv2.putText(color_image, "Position chessboard in view", (30, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow(f"RealSense {device} - Calibration", color_image)
            
            # Also show webcam feeds if available
            if len(webcam_manager._enabled_webcams) > 0:
                webcam_frames = webcam_manager.poll_frames()
                for webcam_id, frames in webcam_frames.items():
                    if 'color' in frames:
                        color_image = frames['color'].copy()
                        cv2.putText(color_image, "Calibrating RealSense...", (30, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.imshow(f"Webcam {webcam_id}", color_image)
            
            # Check for ESC key to cancel
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                raise KeyboardInterrupt("Calibration cancelled by user")
            
            # Perform pose estimation
            pose_estimator = PoseEstimation(rs_frames, intrinsics_devices, chessboard_params)
            transformation_result_kabsch = pose_estimator.perform_pose_estimation()
            object_point = pose_estimator.get_chessboard_corners_in3d()
            
            # Check calibration status for each device
            calibrated_device_count = 0
            for device_info in device_manager._available_devices:
                device = device_info[0]
                if not transformation_result_kabsch[device][0]:
                    if calibration_attempts % 10 == 0:  # Print message less frequently
                        print(f"Waiting for chessboard to be detected by RealSense device {device}...")
                else:
                    calibrated_device_count += 1
            
            # Print progress
            if calibration_attempts % 10 == 0:
                print(f"Calibration progress: {calibrated_device_count}/{len(device_manager._available_devices)} devices calibrated")

        # Clean up RealSense calibration windows
        for device_info in device_manager._available_devices:
            device = device_info[0]
            cv2.destroyWindow(f"RealSense {device} - Calibration")

        if calibration_attempts >= max_attempts:
            raise RuntimeError("Failed to calibrate RealSense devices after maximum attempts")

        print("All RealSense devices calibrated successfully!")

        # Save the transformation object for all RealSense devices
        transformation_devices = {}
        chessboard_points_cumulative_3d = np.array([-1, -1, -1]).transpose()
        for device_info in device_manager._available_devices:
            device = device_info[0]
            transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
            points3D = object_point[device][2][:, object_point[device][3]]
            points3D = transformation_devices[device].apply_transformation(points3D)
            chessboard_points_cumulative_3d = np.column_stack((chessboard_points_cumulative_3d, points3D))

        # Extract the bounds between which the object's dimensions are needed
        chessboard_points_cumulative_3d = np.delete(chessboard_points_cumulative_3d, 0, 1)
        roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)

        """
        2: Calibration for webcams
        """
        # Only calibrate webcams if there are any enabled
        webcam_intrinsics = {}
        if len(webcam_manager._enabled_webcams) > 0:
            # Calibrate webcams
            webcam_intrinsics = webcam_manager.calibrate_webcams(chessboard_height, chessboard_width)

        # Print overall calibration status
        print("\n--- Calibration Summary ---")
        print(f"RealSense cameras: {len(device_manager._available_devices)} detected, {calibrated_device_count} calibrated")
        print(f"Webcams: {len(webcam_manager._enabled_webcams)} detected, {len(webcam_intrinsics)} calibrated")
        print("Calibration completed... \nPlace the box in the field of view of the devices...")

        """
        3: Measurement and display
        """
        # Enable the emitter of the RealSense devices
        device_manager.enable_emitter(True)

        # Load the JSON settings file for high accuracy on RealSense
        try:
            device_manager.load_settings_json("./HighResHighAccuracyPreset.json")
            print("Loaded high accuracy preset for RealSense")
        except Exception as e:
            print(f"Warning: Could not load settings JSON: {e}")
            print("Continuing with default settings")

        # Get the extrinsics of the RealSense devices
        extrinsics_devices = device_manager.get_depth_to_color_extrinsics(rs_frames)

        # Combine calibration info for RealSense devices
        calibration_info_devices = defaultdict(list)
        for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
            for key, value in calibration_info.items():
                calibration_info_devices[key].append(value)

        print("\n--- Starting measurement mode ---")
        print("Press ESC to exit the program")

        # Continue acquisition until terminated with Ctrl+C by the user
        while True:
            try:
                # Get frames from RealSense devices
                rs_frames_devices = device_manager.poll_frames()
                
                # Get frames from webcams if any are enabled
                webcam_frames = {}
                if len(webcam_manager._enabled_webcams) > 0:
                    webcam_frames = webcam_manager.poll_frames()

                # Calculate the pointcloud using the depth frames from all the RealSense devices
                point_cloud = calculate_cumulative_pointcloud(rs_frames_devices, calibration_info_devices, roi_2D)

                # Get the bounding box for the pointcloud in image coordinates of the color imager
                bounding_box_points_color_image, length, width, height = calculate_boundingbox_points(point_cloud, calibration_info_devices)

                # Draw the bounding box points on the color image and visualize the results from RealSense
                visualise_measurements(rs_frames_devices, bounding_box_points_color_image, length, width, height)
                
                # Display webcam frames with measurement information if any are enabled
                for webcam_id, frames in webcam_frames.items():
                    color_frame = frames.get('color')
                    if color_frame is not None:
                        # Draw measurement text on webcam frame
                        # cv2.putText(color_frame, f"Length: {length:.2f}mm", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # cv2.putText(color_frame, f"Width: {width:.2f}mm", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # cv2.putText(color_frame, f"Height: {height:.2f}mm", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 

                         # Draw measurement text on webcam frame
                        cv2.putText(color_frame, f"Length: {length:.2f}mm", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(color_frame, f"Width: {width:.2f}mm", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(color_frame, f"Height: {height:.2f}mm", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Display webcam frame
                        cv2.imshow(f"Webcam {webcam_id}", color_frame)
                
                # Check for quit key
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    break

            except KeyboardInterrupt:
                print("The program was interrupted by the user. Closing the program...")

    finally:
        # Clean up
        device_manager.disable_streams()
        webcam_manager.disable_webcams()
        cv2.destroyAllWindows()

                            
###########################################################################################################################
##                      License: Apache 2.0. See LICENSE file in root directory.                                         ##
###########################################################################################################################
##                  Enhanced Box Dimensioner with RealSense and webcams                                                ##
###########################################################################################################################
## Workflow description:                                                                                                 ##
## 1. Place the calibration chessboard object into the field of view of all cameras.                                    ##
##    Update the chessboard parameters in the script in case a different size is chosen.                                 ##
## 2. Start the program.                                                                                                 ##
## 3. Allow calibration to occur and place the desired object ON the calibration object when the program asks for it.    ##
##    Make sure that the object to be measured is not bigger than the calibration object in length and width.            ##
## 4. The length, width and height of the bounding box of the object is then displayed in millimeters.                   ##
###########################################################################################################################

# Import RealSense, OpenCV and NumPy
import pyrealsense2 as rs
import cv2
import numpy as np
import time

# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from collections import defaultdict
from realsense_device_manager import DeviceManager
from calibration_kabsch import PoseEstimation
from helper_functions import get_boundary_corners_2D
from measurement_task import calculate_boundingbox_points, calculate_cumulative_pointcloud, visualise_measurements

# WebcamManager class to handle regular webcams
class WebcamManager:
    def __init__(self, webcam_ids=None):
        """
        Initialize webcam manager
        
        Parameters:
        -----------
        webcam_ids : list
            List of webcam IDs to use. If None, no webcams will be enabled.
        """
        self._available_webcams = []
        self._enabled_webcams = {}
        
        # Try to detect webcams if no IDs provided
        if webcam_ids is None or len(webcam_ids) == 0:
            webcam_ids = []
            print("Searching for available webcams...")
            for i in range(10):  # Try the first 10 possible camera indices
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            webcam_ids.append(i)
                            print(f"Found working webcam at index {i}")
                        else:
                            print(f"Webcam at index {i} opened but couldn't read frame")
                    cap.release()
                except Exception as e:
                    print(f"Error checking webcam at index {i}: {e}")
        
        # Store available webcams
        for webcam_id in webcam_ids:
            self._available_webcams.append(webcam_id)
            
        print(f"{len(self._available_webcams)} webcams have been found: {self._available_webcams}")
    
    def enable_all_webcams(self, resolution_width=1280, resolution_height=720):
        """
        Enable all detected webcams
        """
        for webcam_id in self._available_webcams:
            success = self.enable_webcam(webcam_id, resolution_width, resolution_height)
            if not success:
                print(f"Failed to enable webcam {webcam_id}")
    
    def enable_webcam(self, webcam_id, resolution_width=1280, resolution_height=720):
        """
        Enable a specific webcam
        
        Returns:
        --------
        bool: Whether the webcam was successfully enabled
        """
        try:
            # Try DirectShow backend on Windows for better webcam support
            cap = cv2.VideoCapture(webcam_id, cv2.CAP_DSHOW)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                print(f"Could not open webcam {webcam_id}")
                return False
                
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
            
            # Verify we can get a frame
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print(f"Could not read frame from webcam {webcam_id}")
                cap.release()
                return False
            
            # Store the enabled webcam
            self._enabled_webcams[str(webcam_id)] = cap
            
            print(f"Webcam {webcam_id} enabled successfully (resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
            return True
            
        except Exception as e:
            print(f"Error enabling webcam {webcam_id}: {e}")
            return False
    
    def poll_frames(self):
        """
        Poll for frames from all enabled webcams
        
        Returns:
        -----------
        frames : dict
            Dictionary with webcam ID as key and frame as value
        """
        frames = {}
        bad_webcams = []
        
        for webcam_id, cap in self._enabled_webcams.items():
            try:
                if not cap.isOpened():
                    print(f"Webcam {webcam_id} is not open")
                    bad_webcams.append(webcam_id)
                    continue
                    
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    frames[webcam_id] = {'color': frame}
                else:
                    print(f"Failed to get valid frame from webcam {webcam_id}")
                    bad_webcams.append(webcam_id)
            except Exception as e:
                print(f"Error reading from webcam {webcam_id}: {e}")
                bad_webcams.append(webcam_id)
        
        # Remove problematic webcams
        for webcam_id in bad_webcams:
            try:
                self._enabled_webcams[webcam_id].release()
                del self._enabled_webcams[webcam_id]
                print(f"Disabled problematic webcam {webcam_id}")
            except:
                pass
                
        return frames
    
    def calibrate_webcams(self, chessboard_height, chessboard_width):
        """
        Calibrate webcams using OpenCV's camera calibration
        
        Returns:
        -----------
        intrinsics : dict
            Dictionary with webcam ID as key and intrinsics as value
        """
        intrinsics = {}
        
        if not self._enabled_webcams:
            print("No webcams to calibrate")
            return intrinsics
            
        print("\n--- Starting webcam calibration ---")
        print("Please make sure the chessboard is visible to all webcams")
        print("Press 'c' to capture a frame for calibration")
        print("Press 'ESC' to finish calibration\n")
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((chessboard_height * chessboard_width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_width, 0:chessboard_height].T.reshape(-1, 2)
        
        # Dictionary to store calibration data for each webcam
        calibration_data = {}
        for webcam_id in self._enabled_webcams.keys():
            calibration_data[webcam_id] = {
                'objpoints': [],  # 3d points in real world space
                'imgpoints': []   # 2d points in image plane
            }
        
        # Calibration capture loop
        max_frames = 10  # Maximum number of frames to capture for calibration
        captured_frames = 0
        last_capture_time = time.time()
        
        while captured_frames < max_frames:
            # Get current frames from all webcams
            all_frames = self.poll_frames()
            
            # Display frames and check for chessboard
            for webcam_id, frames in all_frames.items():
                if 'color' not in frames:
                    continue
                    
                frame = frames['color'].copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, (chessboard_width, chessboard_height), None)
                
                # Draw the detected corners
                if ret:
                    # Draw corners
                    cv2.drawChessboardCorners(frame, (chessboard_width, chessboard_height), corners, ret)
                    cv2.putText(frame, "Chessboard detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No chessboard detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show instruction
                cv2.putText(frame, f"Frames: {captured_frames}/{max_frames}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, "Press 'c' to capture, 'ESC' to finish", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Show the frame
                cv2.imshow(f"Calibration - Webcam {webcam_id}", frame)
            
            # Check for key press
            key = cv2.waitKey(1)
            
            # 'c' key to capture a frame
            if key == ord('c') and (time.time() - last_capture_time) > 1.0:  # Ensure at least 1 second between captures
                print(f"Capturing calibration frame {captured_frames + 1}/{max_frames}")
                last_capture_time = time.time()
                
                # Process each webcam
                successful_captures = 0
                for webcam_id, frames in all_frames.items():
                    if 'color' not in frames:
                        continue
                        
                    frame = frames['color']
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Find the chessboard corners
                    ret, corners = cv2.findChessboardCorners(gray, (chessboard_width, chessboard_height), None)
                    
                    # If found, add object points, image points
                    if ret:
                        # Refine corner positions
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        
                        calibration_data[webcam_id]['objpoints'].append(objp)
                        calibration_data[webcam_id]['imgpoints'].append(corners_refined)
                        successful_captures += 1
                        
                        print(f"  Chessboard found in webcam {webcam_id}")
                    else:
                        print(f"  No chessboard found in webcam {webcam_id}")
                
                if successful_captures > 0:
                    captured_frames += 1
                else:
                    print("  No chessboards were detected in any webcam, please adjust the chessboard position")
            
            # ESC key to exit
            elif key == 27:
                print("Calibration process stopped by user")
                break
        
        # Perform the actual calibration for each webcam
        print("\nCalculating camera parameters...")
        for webcam_id, data in calibration_data.items():
            if len(data['objpoints']) >= 3:  # Need at least 3 good captures for calibration
                try:
                    # Get frame dimensions from the webcam
                    ret, frame = self._enabled_webcams[webcam_id].read()
                    if not ret:
                        print(f"  Cannot read from webcam {webcam_id} for calibration")
                        continue
                        
                    h, w = frame.shape[:2]
                    
                    # Calibrate the camera
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                        data['objpoints'], data['imgpoints'], (w, h), None, None
                    )
                    
                    if ret:
                        # Calculate re-projection error
                        total_error = 0
                        for i in range(len(data['objpoints'])):
                            imgpoints2, _ = cv2.projectPoints(data['objpoints'][i], rvecs[i], tvecs[i], mtx, dist)
                            error = cv2.norm(data['imgpoints'][i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                            total_error += error
                        
                        error = total_error / len(data['objpoints']) if len(data['objpoints']) > 0 else float('inf')
                        
                        # Store intrinsics
                        intrinsics[webcam_id] = {
                            'camera_matrix': mtx,
                            'dist_coeffs': dist,
                            'rvecs': rvecs,
                            'tvecs': tvecs,
                            'error': error
                        }
                        
                        print(f"  Webcam {webcam_id} calibrated successfully (error: {error:.6f})")
                    else:
                        print(f"  Failed to calibrate webcam {webcam_id}")
                except Exception as e:
                    print(f"  Error calibrating webcam {webcam_id}: {e}")
            else:
                print(f"  Not enough good frames for webcam {webcam_id} (need at least 3, got {len(data['objpoints'])})")
        
        # Clean up windows
        for webcam_id in self._enabled_webcams.keys():
            cv2.destroyWindow(f"Calibration - Webcam {webcam_id}")
        
        print("--- Webcam calibration complete ---\n")
        return intrinsics
    
    def disable_webcams(self):
        """
        Release all webcams
        """
        for webcam_id, cap in self._enabled_webcams.items():
            cap.release()
        
        self._enabled_webcams = {}
        print("All webcams disabled")

def run_demo():
    # Define some constants
    resolution_width = 1280  # pixels
    resolution_height = 720  # pixels
    frame_rate = 15  # fps

    dispose_frames_for_stablisation = 30  # frames

    chessboard_width = 4  # squares
    chessboard_height = 7  # squares
    square_size = 0.033  # meters

    # Define webcam IDs - update these with your webcam device IDs
    # In Windows, these are typically 0, 1, 2, etc.
    # The RealSense camera is usually handled separately by the RealSense SDK
    webcam_ids = []  # Start with empty list and detect automatically

    try:
        # Initialize RealSense device manager
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
        rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

        device_manager = DeviceManager(rs.context(), rs_config)
        device_manager.enable_all_devices()

        # Initialize webcam manager
        webcam_manager = WebcamManager(webcam_ids)
        webcam_manager.enable_all_webcams(resolution_width, resolution_height)

        # Allow some frames for the auto-exposure controller to stabilize
        for frame in range(dispose_frames_for_stablisation):
            rs_frames = device_manager.poll_frames()
            webcam_frames = webcam_manager.poll_frames()
            time.sleep(0.1)  # Short delay to allow cameras to stabilize

        # Verify we have at least one RealSense device
        assert len(device_manager._available_devices) > 0, "No RealSense devices found"
        print(f"Found {len(device_manager._available_devices)} RealSense devices and {len(webcam_manager._enabled_webcams)} webcams")

        """
        1: Calibration for RealSense cameras
        """
        # Get the intrinsics of the RealSense devices
        rs_frames = device_manager.poll_frames()
        intrinsics_devices = device_manager.get_device_intrinsics(rs_frames)

        # Set the chessboard parameters for calibration
        chessboard_params = [chessboard_height, chessboard_width, square_size]

        # Estimate the pose of the chessboard in the world coordinate using the Kabsch Method for RealSense
        calibrated_device_count = 0
        while calibrated_device_count < len(device_manager._available_devices):
            rs_frames = device_manager.poll_frames()
            pose_estimator = PoseEstimation(rs_frames, intrinsics_devices, chessboard_params)
            transformation_result_kabsch = pose_estimator.perform_pose_estimation()
            object_point = pose_estimator.get_chessboard_corners_in3d()
            calibrated_device_count = 0
            for device_info in device_manager._available_devices:
                device = device_info[0]
                if not transformation_result_kabsch[device][0]:
                    print("Place the chessboard on the plane where the object needs to be detected..")
                else:
                    calibrated_device_count += 1

        # Save the transformation object for all RealSense devices
        transformation_devices = {}
        chessboard_points_cumulative_3d = np.array([-1, -1, -1]).transpose()
        for device_info in device_manager._available_devices:
            device = device_info[0]
            transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
            points3D = object_point[device][2][:, object_point[device][3]]
            points3D = transformation_devices[device].apply_transformation(points3D)
            chessboard_points_cumulative_3d = np.column_stack((chessboard_points_cumulative_3d, points3D))

        # Extract the bounds between which the object's dimensions are needed
        chessboard_points_cumulative_3d = np.delete(chessboard_points_cumulative_3d, 0, 1)
        roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)

        """
        2: Calibration for webcams
        """
        # Calibrate webcams
        webcam_intrinsics = webcam_manager.calibrate_webcams(chessboard_height, chessboard_width)

        print("Calibration completed... \nPlace the box in the field of view of the devices...")

        """
        3: Measurement and display
        """
        # Enable the emitter of the RealSense devices
        device_manager.enable_emitter(True)

        # Load the JSON settings file for high accuracy on RealSense
        try:
            device_manager.load_settings_json("./HighResHighAccuracyPreset.json")
        except Exception as e:
            print(f"Warning: Could not load settings JSON: {e}")

        # Get the extrinsics of the RealSense devices
        extrinsics_devices = device_manager.get_depth_to_color_extrinsics(rs_frames)

        # Combine calibration info for RealSense devices
        calibration_info_devices = defaultdict(list)
        for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
            for key, value in calibration_info.items():
                calibration_info_devices[key].append(value)

        # Continue acquisition until terminated with Ctrl+C by the user
        while True:
            # Get frames from RealSense devices
            rs_frames_devices = device_manager.poll_frames()
            
            # Get frames from webcams
            webcam_frames = webcam_manager.poll_frames()

            # Calculate the pointcloud using the depth frames from all the RealSense devices
            point_cloud = calculate_cumulative_pointcloud(rs_frames_devices, calibration_info_devices, roi_2D)

            # Get the bounding box for the pointcloud in image coordinates of the color imager
            bounding_box_points_color_image, length, width, height = calculate_boundingbox_points(point_cloud, calibration_info_devices)

            # Draw the bounding box points on the color image and visualize the results from RealSense
            visualise_measurements(rs_frames_devices, bounding_box_points_color_image, length, width, height)
            
            # Display webcam frames with measurement information
            for webcam_id, frames in webcam_frames.items():
                color_frame = frames.get('color')
                if color_frame is not None:
                    # Draw measurement text on webcam frame
                    cv2.putText(color_frame, f"Length: {length:.2f}mm", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(color_frame, f"Width: {width:.2f}mm", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(color_frame, f"Height: {height:.2f}mm", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display webcam frame
                    cv2.imshow(f"Webcam {webcam_id}", color_frame)
            
            # Check for quit key
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break

    except KeyboardInterrupt:
        print("The program was interrupted by the user. Closing the program...")

    finally:
        # Clean up
        device_manager.disable_streams()
        webcam_manager.disable_webcams()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
