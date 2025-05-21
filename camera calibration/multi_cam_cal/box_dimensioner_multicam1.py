###########################################################################################################################
##                      License: Apache 2.0. See LICENSE file in root directory.                                         ##
###########################################################################################################################
##                  Enhanced Box Dimensioner with multiple RealSense cameras                                            ##
###########################################################################################################################
## Workflow description:                                                                                                 ##
## 1. Place the calibration chessboard object into the field of view of all the realsense cameras.                       ##
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
import os

# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from collections import defaultdict
from realsense_device_manager import DeviceManager
from calibration_kabsch import PoseEstimation
from helper_functions import get_boundary_corners_2D
from measurement_task import calculate_boundingbox_points, calculate_cumulative_pointcloud, visualise_measurements

def run_demo():

    # Define some constants
    resolution_width = 640  # Lower resolution for better performance
    resolution_height = 480  # Lower resolution for better performance
    frame_rate = 15  # fps

    dispose_frames_for_stablisation = 30  # frames

    chessboard_width = 4  # squares
    chessboard_height = 7  # squares
    square_size = 0.033  # meters

    try:
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
        rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

        # Use the device manager class to enable the devices and get the frames
        device_manager = DeviceManager(rs.context(), rs_config)
        device_manager.enable_all_devices()

        # Allow some frames for the auto-exposure controller to stabilize
        print("Stabilizing cameras...")
        for frame in range(dispose_frames_for_stablisation):
            frames = device_manager.poll_frames()
            if frame % 10 == 0:  # Print status every 10 frames
                print(f"Stabilizing: {frame}/{dispose_frames_for_stablisation}")
            time.sleep(0.1)  # Short delay to allow cameras to stabilize

        # Verify we have at least one RealSense device
        num_devices = len(device_manager._available_devices)
        assert(num_devices > 0)
        print(f"\nFound {num_devices} RealSense devices")
        
        # Print device information
        for i, device_info in enumerate(device_manager._available_devices):
            device = device_info[0]
            print(f"Device {i+1}: Serial: {device}")
            
        """
        1: Calibration
        Calibrate all the available devices to the world co-ordinates.
        For this purpose, a chessboard printout for use with opencv based calibration process is needed.
        """
        print("\n--- Starting RealSense calibration ---")
        print("Place the chessboard where all RealSense cameras can see it clearly")
        
        # Get the intrinsics of the realsense device
        frames = device_manager.poll_frames()
        intrinsics_devices = device_manager.get_device_intrinsics(frames)

        # Set the chessboard parameters for calibration
        chessboard_params = [chessboard_height, chessboard_width, square_size]

        # Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
        calibrated_device_count = 0
        calibration_attempts = 0
        max_attempts = 300  # Prevent infinite loop
        
        print("Searching for chessboard pattern...")
        while calibrated_device_count < num_devices and calibration_attempts < max_attempts:
            calibration_attempts += 1
            frames = device_manager.poll_frames()
            
            # Display the color frames from RealSense during calibration
            for device_info in device_manager._available_devices:
                device = device_info[0]
                if device in frames and rs.stream.color in frames[device]:
                    color_frame = frames[device][rs.stream.color]
                    color_image = np.asanyarray(color_frame.get_data())
                    cv2.putText(color_image, "Position chessboard in view", (30, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow(f"RealSense {device} - Calibration", color_image)
            
            # Check for ESC key to cancel
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                raise KeyboardInterrupt("Calibration cancelled by user")
            
            # Perform pose estimation
            pose_estimator = PoseEstimation(frames, intrinsics_devices, chessboard_params)
            transformation_result_kabsch = pose_estimator.perform_pose_estimation()
            object_point = pose_estimator.get_chessboard_corners_in3d()
            
            # Check calibration status for each device
            calibrated_device_count = 0
            for device_info in device_manager._available_devices:
                device = device_info[0]
                if not transformation_result_kabsch[device][0]:
                    if calibration_attempts % 10 == 0:  # Print message less frequently
                        print(f"Waiting for chessboard to be detected by device {device}...")
                else:
                    calibrated_device_count += 1
            
            # Print progress periodically
            if calibration_attempts % 10 == 0:
                print(f"Calibration progress: {calibrated_device_count}/{num_devices} devices calibrated")

        # Clean up RealSense calibration windows
        for device_info in device_manager._available_devices:
            device = device_info[0]
            cv2.destroyWindow(f"RealSense {device} - Calibration")

        if calibration_attempts >= max_attempts:
            raise RuntimeError("Failed to calibrate RealSense devices after maximum attempts")

        print("All RealSense devices calibrated successfully!")

        # Save the transformation object for all devices
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

        print("Calibration completed... \nPlace the box in the field of view of the devices...")

        """
        2: Measurement and display
        Measure the dimension of the object using depth maps from multiple RealSense devices
        The information from Phase 1 will be used here
        """
        # Enable the emitter of the devices
        device_manager.enable_emitter(True)

        # Try to enable advanced mode and load the JSON settings file
        try:
            print("Attempting to enable Advanced Mode for devices...")
            for device_info in device_manager._available_devices:
                device_id = device_info[0]
                device = device_info[1].pipeline_profile.get_device()
                
                # Check if device supports advanced mode
                if device.supports(rs.camera_info.product_line) and device.get_info(rs.camera_info.product_line) == "D400":
                    print(f"Enabling Advanced Mode for device {device_id}")
                    advanced_mode = rs.rs400_advanced_mode(device)
                    if not advanced_mode.is_enabled():
                        advanced_mode.toggle_advanced_mode(True)
                        print(f"Advanced mode enabled for device {device_id}")
                        # Wait for advanced mode to apply
                        time.sleep(2)
                    else:
                        print(f"Advanced mode already enabled for device {device_id}")
                else:
                    print(f"Device {device_id} does not support advanced mode")
                    
            # Try to load the JSON settings file
            if os.path.exists("./HighResHighAccuracyPreset.json"):
                print("Loading high accuracy preset...")
                device_manager.load_settings_json("./HighResHighAccuracyPreset.json")
                print("High accuracy preset loaded successfully")
            else:
                print("HighResHighAccuracyPreset.json not found, continuing with default settings")
                
        except Exception as e:
            print(f"Warning: Could not enable advanced mode or load settings: {e}")
            print("Continuing with default settings...")

        # Get the extrinsics of the device to be used later
        frames = device_manager.poll_frames()
        extrinsics_devices = device_manager.get_depth_to_color_extrinsics(frames)

        # Get the calibration info as a dictionary
        calibration_info_devices = defaultdict(list)
        for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
            for key, value in calibration_info.items():
                calibration_info_devices[key].append(value)

        print("\n--- Starting measurement mode ---")
        print("Press ESC to exit the program")

        # Continue acquisition until terminated by user
        while True:
            try:
                # Get the frames from all the devices
                frames_devices = device_manager.poll_frames()

                # Calculate the pointcloud using the depth frames
                point_cloud = calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2D)

                # Get the bounding box for the pointcloud
                bounding_box_points_color_image, length, width, height = calculate_boundingbox_points(point_cloud, calibration_info_devices)

                # Draw the bounding box points and visualize the results
                visualise_measurements(frames_devices, bounding_box_points_color_image, length, width, height)
                
                # Check for exit key
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    print("Exiting program...")
                    break

            except Exception as e:
                print(f"Error during measurement: {e}")
                time.sleep(1)  # Pause briefly before retrying

    except KeyboardInterrupt:
        print("The program was interrupted by the user. Closing the program...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up resources...")
        device_manager.disable_streams()
        cv2.destroyAllWindows()
        print("Program ended.")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"Fatal error: {e}")
        cv2.destroyAllWindows()
