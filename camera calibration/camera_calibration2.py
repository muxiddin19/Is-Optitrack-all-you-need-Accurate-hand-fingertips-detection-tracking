import numpy as np
import cv2
import glob
import os
import pickle
import argparse

def main():
    """
    Main function to run the camera calibration process with command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Camera Calibration Tool')
    parser.add_argument('--chessboard_width', type=int, default=7, help='Number of inner corners along width')
    parser.add_argument('--chessboard_height', type=int, default=4, help='Number of inner corners along height')
    parser.add_argument('--square_size', type=float, default=3.3, help='Size of square in cm')
    parser.add_argument('--images_path', type=str, default='calibration_images/*.jpg', help='Path to calibration images')
    parser.add_argument('--output_dir', type=str, default='output3', help='Directory to save results')
    parser.add_argument('--save_undistorted', action='store_true', help='Save undistorted images')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to save detection attempts')
    parser.add_argument('--adaptive', action='store_true', help='Try adaptive thresholding for detection')
    
    args = parser.parse_args()
    
    # Set parameters based on command-line arguments
    CHESSBOARD_SIZE = (args.chessboard_width, args.chessboard_height)
    SQUARE_SIZE = args.square_size
    CALIBRATION_IMAGES_PATH = args.images_path
    OUTPUT_DIRECTORY = args.output_dir
    SAVE_UNDISTORTED = args.save_undistorted
    DEBUG_MODE = args.debug
    ADAPTIVE_THRESHOLD = args.adaptive
    
    print("Starting camera calibration with the following parameters:")
    print(f"  Chessboard Size: {CHESSBOARD_SIZE}")
    print(f"  Square Size: {SQUARE_SIZE} cm")
    print(f"  Images Path: {CALIBRATION_IMAGES_PATH}")
    print(f"  Output Directory: {OUTPUT_DIRECTORY}")
    print(f"  Debug Mode: {DEBUG_MODE}")
    print(f"  Adaptive Threshold: {ADAPTIVE_THRESHOLD}")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, successful_images = calibrate_camera(
        CHESSBOARD_SIZE, SQUARE_SIZE, CALIBRATION_IMAGES_PATH, OUTPUT_DIRECTORY, DEBUG_MODE, ADAPTIVE_THRESHOLD
    )
    
    if mtx is None:
        print("Calibration failed. Exiting.")
        return
    
    if len(successful_images) > 0:
        print(f"Camera successfully calibrated with {len(successful_images)} images! Reprojection error: {ret}")
        
        # Calculate reprojection error
        mean_error = calculate_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)
        
        # Undistort images
        if SAVE_UNDISTORTED:
            undistort_images(mtx, dist, successful_images, OUTPUT_DIRECTORY)
        
        print("Camera calibration completed successfully!")
    else:
        print("Calibration failed. No valid chessboard patterns found.")


def calibrate_camera(CHESSBOARD_SIZE, SQUARE_SIZE, CALIBRATION_IMAGES_PATH, OUTPUT_DIRECTORY, DEBUG_MODE, ADAPTIVE_THRESHOLD):
    """
    Calibrate the camera using chessboard images.
    
    Returns:
        ret: The RMS re-projection error
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
        objpoints: 3D object points
        imgpoints: 2D image points
        successful_images: List of images where chessboard was detected
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (width-1, height-1, 0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    
    # Scale object points by square size (for real-world measurements)
    objp = objp * SQUARE_SIZE
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    successful_images = []  # List of successful images
    
    # Get list of calibration images
    images = glob.glob(CALIBRATION_IMAGES_PATH)
    
    if not images:
        print(f"No calibration images found at {CALIBRATION_IMAGES_PATH}")
        return None, None, None, None, None, None, None, []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    # Create debug directory if needed
    if DEBUG_MODE:
        debug_dir = os.path.join(OUTPUT_DIRECTORY, 'debug')
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
    
    print(f"Found {len(images)} calibration images")
    
    # Process each calibration image
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to read image {fname}. Skipping.")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try to find chessboard corners with different flags
        found = False
        corners = None
        
        # First try with default flags
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        # If not found and adaptive threshold is enabled, try with adaptive thresholding
        if not ret and ADAPTIVE_THRESHOLD:
            # Try with adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            ret, corners = cv2.findChessboardCorners(
                adaptive_thresh, CHESSBOARD_SIZE, 
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if DEBUG_MODE:
                cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, 'debug', f'adaptive_{os.path.basename(fname)}'), adaptive_thresh)
        
        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            successful_images.append(fname)
            
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            img_with_corners = img.copy()
            cv2.drawChessboardCorners(img_with_corners, CHESSBOARD_SIZE, corners2, ret)
            
            # Save image with corners drawn
            output_img_path = os.path.join(OUTPUT_DIRECTORY, f'corners_{os.path.basename(fname)}')
            cv2.imwrite(output_img_path, img_with_corners)
            
            print(f"Processed image {idx+1}/{len(images)}: {os.path.basename(fname)} - Chessboard found")
        else:
            print(f"Processed image {idx+1}/{len(images)}: {os.path.basename(fname)} - Chessboard NOT found")
            
            # Save debug images
            if DEBUG_MODE:
                # Draw grid on image for debugging
                debug_img = img.copy()
                h, w = img.shape[:2]
                grid_size = 50
                
                # Draw grid
                for x in range(0, w, grid_size):
                    cv2.line(debug_img, (x, 0), (x, h), (0, 255, 0), 1)
                for y in range(0, h, grid_size):
                    cv2.line(debug_img, (0, y), (w, y), (0, 255, 0), 1)
                
                # Save debug image
                debug_path = os.path.join(OUTPUT_DIRECTORY, 'debug', f'debug_{os.path.basename(fname)}')
                cv2.imwrite(debug_path, debug_img)
                
                # Try different chessboard sizes around the specified size
                for w_offset in [-1, 0, 1]:
                    for h_offset in [-1, 0, 1]:
                        if w_offset == 0 and h_offset == 0:
                            continue  # Skip the original size which was already tried
                        
                        test_size = (CHESSBOARD_SIZE[0] + w_offset, CHESSBOARD_SIZE[1] + h_offset)
                        if test_size[0] <= 0 or test_size[1] <= 0:
                            continue
                            
                        test_ret, test_corners = cv2.findChessboardCorners(gray, test_size, None)
                        if test_ret:
                            test_img = img.copy()
                            cv2.drawChessboardCorners(test_img, test_size, test_corners, test_ret)
                            test_path = os.path.join(OUTPUT_DIRECTORY, 'debug', 
                                                    f'size_{test_size[0]}x{test_size[1]}_{os.path.basename(fname)}')
                            cv2.imwrite(test_path, test_img)
                            print(f"  NOTE: Chessboard found with alternate size {test_size}")
    
    if not objpoints:
        print("No chessboard patterns were detected in any images.")
        return None, None, None, None, None, None, None, []
    
    print("Calibrating camera...")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Save calibration results
    calibration_data = {
        'camera_matrix': mtx,
        'distortion_coefficients': dist,
        'rotation_vectors': rvecs,
        'translation_vectors': tvecs,
        'reprojection_error': ret,
        'chessboard_size': CHESSBOARD_SIZE,
        'square_size': SQUARE_SIZE,
        'successful_images': successful_images
    }
    
    with open(os.path.join(OUTPUT_DIRECTORY, 'calibration_data.pkl'), 'wb') as f:
        pickle.dump(calibration_data, f)
    
    # Save camera matrix and distortion coefficients as text files
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'camera_matrix.txt'), mtx)
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'distortion_coefficients.txt'), dist)
    
    print(f"Calibration complete! RMS re-projection error: {ret}")
    print(f"Results saved to {OUTPUT_DIRECTORY}")
    
    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, successful_images


def undistort_images(mtx, dist, image_paths, OUTPUT_DIRECTORY):
    """
    Undistort all calibration images using the calibration results.
    
    Args:
        mtx: Camera matrix
        dist: Distortion coefficients
        image_paths: List of successful image paths
        OUTPUT_DIRECTORY: Directory to save output
    """
    undistorted_dir = os.path.join(OUTPUT_DIRECTORY, 'undistorted')
    if not os.path.exists(undistorted_dir):
        os.makedirs(undistorted_dir)
    
    print(f"Undistorting {len(image_paths)} images...")
    
    for idx, fname in enumerate(image_paths):
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        
        # Refine camera matrix based on free scaling parameter
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # Undistort image
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # Crop the image (optional)
        x, y, w, h = roi
        if w > 0 and h > 0:
            dst = dst[y:y+h, x:x+w]
        
        # Save undistorted image
        output_img_path = os.path.join(undistorted_dir, f'undistorted_{os.path.basename(fname)}')
        cv2.imwrite(output_img_path, dst)
        
        print(f"Undistorted image {idx+1}/{len(image_paths)}: {os.path.basename(fname)}")
    
    print(f"Undistorted images saved to {undistorted_dir}")


def calculate_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    """
    Calculate the reprojection error for each calibration image.
    
    Args:
        objpoints: 3D points in real world space
        imgpoints: 2D points in image plane
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    
    Returns:
        mean_error: Mean reprojection error
    """
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        print(f"Reprojection error for image {i+1}: {error}")
    
    mean_error = total_error / len(objpoints)
    print(f"Mean reprojection error: {mean_error}")
    
    return mean_error


if __name__ == "__main__":
    main()
