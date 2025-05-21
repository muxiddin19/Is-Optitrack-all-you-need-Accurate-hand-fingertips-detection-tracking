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
        if webcam_ids is None:
            webcam_ids = []
            for i in range(10):  # Try the first 10 possible camera indices
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        webcam_ids.append(i)
                cap.release()
        
        # Store available webcams
        for webcam_id in webcam_ids:
            self._available_webcams.append(webcam_id)
            
        print(f"{len(self._available_webcams)} webcams have been found")
    
    def enable_all_webcams(self, resolution_width=1280, resolution_height=720):
        """
        Enable all detected webcams
        """
        for webcam_id in self._available_webcams:
            self.enable_webcam(webcam_id, resolution_width, resolution_height)
    
    def enable_webcam(self, webcam_id, resolution_width=1280, resolution_height=720):
        """
        Enable a specific webcam
        """
        cap = cv2.VideoCapture(webcam_id)
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
        
        # Store the enabled webcam
        self._enabled_webcams[str(webcam_id)] = cap
        
        print(f"Webcam {webcam_id} enabled")
    
    def poll_frames(self):
        """
        Poll for frames from all enabled webcams
        
        Returns:
        -----------
        frames : dict
            Dictionary with webcam ID as key and frame as value
        """
        frames = {}
        for webcam_id, cap in self._enabled_webcams.items():
            ret, frame = cap.read()
            if ret:
                frames[webcam_id] = {'color': frame}
        
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
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((chessboard_height * chessboard_width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_width, 0:chessboard_height].T.reshape(-1, 2)
        
        for webcam_id, cap in self._enabled_webcams.items():
            # Arrays to store object points and image points
            objpoints = []  # 3d points in real world space
            imgpoints = []  # 2d points in image plane
            
            # Capture several frames to improve calibration
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, (chessboard_width, chessboard_height), None)
                
                # If found, add object points, image points
                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    
                    # Draw and display the corners
                    cv2.drawChessboardCorners(frame, (chessboard_width, chessboard_height), corners, ret)
                    cv2.imshow(f'Calibration - Webcam {webcam_id}', frame)
                    cv2.waitKey(500)
            
            # If we got enough points, calibrate
            if len(objpoints) > 0:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                
                # Store intrinsics
                if ret:
                    intrinsics[webcam_id] = {
                        'camera_matrix': mtx,
                        'dist_coeffs': dist,
                        'rvecs': rvecs,
                        'tvecs': tvecs
                    }
                    print(f"Webcam {webcam_id} calibrated successfully")
                else:
                    print(f"Failed to calibrate webcam {webcam_id}")
            
            cv2.destroyWindow(f'Calibration - Webcam {webcam_id}')
        
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
    webcam_ids = [0, 1, 2]  # Example IDs, update as needed

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