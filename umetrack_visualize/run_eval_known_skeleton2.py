# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import av
import fnmatch
import pickle
import numpy as np
import torch
import lib.data_utils.fs as fs
from functools import partial
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.common.hand import NUM_HANDS, NUM_LANDMARKS_PER_HAND
from lib.tracker.tracking_result import SingleHandPose
from multiprocessing import Pool
from typing import Optional, Tuple
from lib.models.model_loader import load_pretrained_model
from lib.tracker.tracker import HandTracker, HandTrackerOpts, InputFrame, ViewData
from lib.tracker.video_pose_data import SyncedImagePoseStream
from lib.common.camera import CameraModel

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def _find_input_output_files(input_dir: str, output_dir: str, test_only: bool):
    res_input_paths = []
    res_output_paths = []
    for cur_dir, _, filenames in fs.walk(input_dir):
        if test_only and not "testing" in cur_dir:
            continue
        mp4_files = fnmatch.filter(filenames, "*.mp4")
        input_full_paths = [fs.join(cur_dir, fname) for fname in mp4_files]
        rel_paths = [f[len(input_dir):] for f in input_full_paths]
        output_full_paths = [fs.join(output_dir, f[:-4] + ".npy") for f in rel_paths]
        res_input_paths += input_full_paths
        res_output_paths += output_full_paths
    assert len(res_input_paths) == len(res_output_paths)
    logger.info(f"Found {len(res_input_paths)} files from {input_dir}")
    return res_input_paths, res_output_paths


def visualize_pose_frame_correct(input_frame, res, hand_model, tracked_keypoints, frame_idx, save_path):
    """
    Corrected visualization using the project's camera system
    """
    import cv2
    import numpy as np
    
    # Create frames directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    views = [view.image.copy() for view in input_frame.views]
    cameras = [view.camera for view in input_frame.views]
    
    # Hand landmarks connections for drawing skeleton
    hand_connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    # Colors for different hands
    hand_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Blue, Red, Cyan
    
    # Process each detected hand
    for hand_idx in res.hand_poses.keys():
        if hand_idx >= tracked_keypoints.shape[0] or frame_idx >= tracked_keypoints.shape[1]:
            continue
            
        # Get 3D keypoints for this hand in world coordinates
        hand_keypoints_world = tracked_keypoints[hand_idx, frame_idx]  # Shape: (21, 3)
        
        if np.any(np.isnan(hand_keypoints_world)):
            continue
            
        color = hand_colors[hand_idx % len(hand_colors)]
        
        # Project to each camera view using the camera's projection methods
        for view_idx, camera in enumerate(cameras):
            try:
                # print(f"World coords range: X[{hand_keypoints_world[:,0].min():.3f}, {hand_keypoints_world[:,0].max():.3f}]")
                # Transform world coordinates to camera eye coordinates
                hand_keypoints_eye = camera.world_to_eye(hand_keypoints_world)
                
                # print(f"Eye coords range: X[{hand_keypoints_eye[:,0].min():.3f}, {hand_keypoints_eye[:,0].max():.3f}], Z[{hand_keypoints_eye[:,2].min():.3f}, {hand_keypoints_eye[:,2].max():.3f}]")
              
                # Check which points are in front of the camera (positive Z in eye space)
                valid_depth = hand_keypoints_eye[:, 2] > 0
                # print(f"Valid depth points: {valid_depth.sum()}/21")
  
                # Project to 2D window coordinates using camera's projection
                hand_keypoints_2d = camera.eye_to_window(hand_keypoints_eye)
                # print(f"2D coords range: X[{hand_keypoints_2d[:,0].min():.1f}, {hand_keypoints_2d[:,0].max():.1f}], Y[{hand_keypoints_2d[:,1].min():.1f}, {hand_keypoints_2d[:,1].max():.1f}]")

                # Check which points are within image boundaries
                valid_bounds = (
                    (hand_keypoints_2d[:, 0] >= 0) & 
                    (hand_keypoints_2d[:, 0] < camera.width) &
                    (hand_keypoints_2d[:, 1] >= 0) & 
                    (hand_keypoints_2d[:, 1] < camera.height)
                )
                
                # Combine depth and boundary checks
                valid_points = valid_depth & valid_bounds
                
                # Draw landmarks that are visible
                for landmark_idx in range(21):
                    if not valid_points[landmark_idx]:
                        continue
                        
                    x, y = hand_keypoints_2d[landmark_idx]
                    
                    # Different sizes for different landmark types
                    if landmark_idx in [4, 8, 12, 16, 20]:  # Fingertips
                        radius = 6
                    elif landmark_idx == 0:  # Wrist
                        radius = 8
                    else:  # Other joints
                        radius = 4
                    
                    # Draw the landmark
                    cv2.circle(views[view_idx], (int(x), int(y)), radius, color, -1)
                    
                    # Draw landmark number for debugging
                    cv2.putText(views[view_idx], str(landmark_idx), 
                              (int(x) + 5, int(y) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
                
                # Draw skeleton connections
                # for start_idx, end_idx in hand_connections:
                #     if not (valid_points[start_idx] and valid_points[end_idx]):
                #         continue
                        
                #     start_2d = hand_keypoints_2d[start_idx]
                #     end_2d = hand_keypoints_2d[end_idx]
                    
                #     cv2.line(views[view_idx], 
                #            (int(start_2d[0]), int(start_2d[1])),
                #            (int(end_2d[0]), int(end_2d[1])), 
                #            color, 2)
                        
            except Exception as e:
                logger.warning(f"Error projecting hand {hand_idx} to view {view_idx}: {e}")
                continue
    # print(f"World coords range: X[{hand_keypoints_world[:,0].min():.3f}, {hand_keypoints_world[:,0].max():.3f}]")
    # print(f"Eye coords range: X[{hand_keypoints_eye[:,0].min():.3f}, {hand_keypoints_eye[:,0].max():.3f}], Z[{hand_keypoints_eye[:,2].min():.3f}, {hand_keypoints_eye[:,2].max():.3f}]")
    # print(f"Valid depth points: {valid_depth.sum()}/21")
    # print(f"2D coords range: X[{hand_keypoints_2d[:,0].min():.1f}, {hand_keypoints_2d[:,0].max():.1f}], Y[{hand_keypoints_2d[:,1].min():.1f}, {hand_keypoints_2d[:,1].max():.1f}]")
    # Add frame and view information
    for view_idx, view in enumerate(views):
        cv2.putText(view, f"Frame: {frame_idx} View: {view_idx}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add camera info
        camera = cameras[view_idx]
        camera_info = f"f: {camera.f[0]:.1f}, c: ({camera.c[0]:.1f},{camera.c[1]:.1f})"
        cv2.putText(view, camera_info, 
                   (10, view.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Concatenate all views horizontally
    fused = np.hstack(views)
    cv2.imwrite(save_path, fused)


def debug_camera_info(input_frame, frame_idx=0):
    """
    Debug function to print camera information
    """
    print(f"\n=== Camera Debug Info (Frame {frame_idx}) ===")
    
    for view_idx, view in enumerate(input_frame.views):
        camera = view.camera
        print(f"\nView {view_idx}:")
        print(f"  Image size: {view.image.shape}")
        print(f"  Camera type: {type(camera).__name__}")
        print(f"  Resolution: {camera.width} x {camera.height}")
        print(f"  Focal length: fx={camera.f[0]:.2f}, fy={camera.f[1]:.2f}")
        print(f"  Principal point: cx={camera.c[0]:.2f}, cy={camera.c[1]:.2f}")
        print(f"  Distortion model: {type(camera.distort).__name__}")
        
        # Print camera pose
        camera_pose = camera.camera_to_world_xf
        print(f"  Camera position: {camera_pose[:3, 3]}")
        print(f"  Camera rotation matrix shape: {camera_pose[:3, :3].shape}")


def _track_sequence(
    input_output: Tuple[str, str],
    model_path: str,
    override: bool = False,
) -> Optional[np.ndarray]:
    data_path, output_path = input_output
    if not override and fs.exists(output_path):
        logger.info(f"Skipping '{data_path}' since output path '{output_path}' already exists")
        return None

    logger.info(f"Processing {data_path}...")
    model = load_pretrained_model(model_path)
    model.eval()

    image_pose_stream = SyncedImagePoseStream(data_path)

    gt_keypoints = np.zeros([NUM_HANDS, len(image_pose_stream), NUM_LANDMARKS_PER_HAND, 3])
    tracked_keypoints = np.zeros([NUM_HANDS, len(image_pose_stream), NUM_LANDMARKS_PER_HAND, 3])
    valid_tracking = np.zeros([NUM_HANDS, len(image_pose_stream)], dtype=bool)
    tracker = HandTracker(model, HandTrackerOpts())
    
    for frame_idx, (input_frame, gt_tracking) in enumerate(image_pose_stream):
        hand_model = image_pose_stream._hand_pose_labels.hand_model
        crop_cameras = tracker.gen_crop_cameras(
            [view.camera for view in input_frame.views],
            image_pose_stream._hand_pose_labels.camera_angles,
            hand_model,
            gt_tracking,
            min_num_crops=1,
        )
        
        # Track the current frame
        res = tracker.track_frame(input_frame, hand_model, crop_cameras)
        
        # Store tracking results for each detected hand
        for hand_idx in res.hand_poses.keys():
            tracked_keypoints[hand_idx, frame_idx] = landmarks_from_hand_pose(
                hand_model, res.hand_poses[hand_idx], hand_idx
            )
            gt_keypoints[hand_idx, frame_idx] = landmarks_from_hand_pose(
                hand_model, gt_tracking[hand_idx], hand_idx
            )
            valid_tracking[hand_idx, frame_idx] = True
        
        # Debug camera info for first frame
        if frame_idx == 0:
            debug_camera_info(input_frame, frame_idx)
        
        # Use corrected visualization
        visualize_pose_frame_correct(input_frame, res, hand_model, tracked_keypoints, frame_idx,
                                   save_path=f"frames/frame_{frame_idx:04d}.jpg")
        
        # Log progress every 30 frames
        if frame_idx % 30 == 0:
            logger.info(f"Processed frame {frame_idx}/{len(image_pose_stream)}")
         
    # Calculate the mean error per frame
    diff_keypoints = (gt_keypoints - tracked_keypoints)[valid_tracking]
    per_frame_mean_error = np.linalg.norm(diff_keypoints, axis=-1).mean(axis=-1)
    
    # Save results
    if not fs.exists(fs.dirname(output_path)):
        os.makedirs(fs.dirname(output_path))
    with fs.open(output_path, "wb") as fp:
        pickle.dump(
            {
                "tracked_keypoints": tracked_keypoints,
                "gt_keypoints": gt_keypoints,
                "valid_tracking": valid_tracking,
            },
            fp,
        )
    logger.info(f"Results saved at {output_path}")
    return per_frame_mean_error


def create_video_from_frames(frames_dir="frames", output_video="frames/hand_tracking_result.mp4", fps=30):
    """
    Create a video from the generated frame images
    """
    import cv2
    import glob
    
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    
    if not frame_files:
        logger.warning("No frame files found to create video")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    logger.info(f"Creating video from {len(frame_files)} frames...")
    
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)
    
    video_writer.release()
    logger.info(f"Video saved as {output_video}")


if __name__ == '__main__':
    root = os.path.dirname(__file__)
    model_name = "pretrained_weights.torch"
    model_path = os.path.join(root, "pretrained_models", model_name)

    error_tensors = []
    input_dir = os.path.join(root, "data", "real")
    output_dir = os.path.join(root, "tmp", "eval_results_known_skeleton1", "real")
    input_paths, output_paths = _find_input_output_files(input_dir, output_dir, test_only=True)
    
    # For debugging, use single process
    pool_size = 1  
    with Pool(pool_size) as p:
        error_tensors = p.map_async(partial(_track_sequence, model_path=model_path), zip(input_paths, output_paths)).get()

    error_tensors = [t for t in error_tensors if t is not None]
    if len(error_tensors) != 0:
        logger.info(f"Final mean error: {np.concatenate(error_tensors).mean()}")
    
    # Create video from frames
    create_video_from_frames()
    
    logger.info("Hand pose tracking and visualization complete!")
