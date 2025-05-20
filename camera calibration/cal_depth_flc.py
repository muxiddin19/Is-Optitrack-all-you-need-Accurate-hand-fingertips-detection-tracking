import pyrealsense2 as rs2
import time
import sys
import os
import numpy as np
import collections  # For frame queue implementation

def clear_line():
    """Clear the current line in the terminal"""
    sys.stdout.write('\r')
    sys.stdout.write(' ' * 80)
    sys.stdout.write('\r')
    sys.stdout.flush()

def print_status(message):
    """Print a status message with timestamp"""
    clear_line()
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{current_time}] {message}")

def progress_callback(progress):
    """Display a progress bar for calibration"""
    bar_length = 30
    filled_length = int(round(bar_length * progress))
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    # Print progress bar
    sys.stdout.write(f'\rCalibration Progress: |{bar}| {progress*100:.1f}% ')
    sys.stdout.flush()

class FrameQueue:
    """Simple frame queue implementation using collections.deque"""
    def __init__(self, maxsize=30):
        self.queue = collections.deque(maxlen=maxsize)
    
    def enqueue(self, frame):
        """Add a frame to the queue"""
        self.queue.append(frame)
    
    def size(self):
        """Return the current size of the queue"""
        return len(self.queue)

def main():
    print_status("Starting RealSense Focal Length Calibration")
    print_status("Initializing RealSense pipeline...")
    
    # Initialize pipeline and config
    pipe = rs2.pipeline()
    cfg = rs2.config()
    
    try:
        # Configure left and right infrared streams (for stereo cameras)
        print_status("Configuring IR streams...")
        # These are the typical stream configurations for stereo cameras
        cfg.enable_stream(rs2.stream.infrared, 1, 640, 480, rs2.format.y8, 30)  # Left IR
        cfg.enable_stream(rs2.stream.infrared, 2, 640, 480, rs2.format.y8, 30)  # Right IR
        
        # Try to start the pipeline
        print_status("Starting pipeline and connecting to device...")
        profile = pipe.start(cfg)
        dev = profile.get_device()
        
        # Print device info
        print_status(f"Connected to device: {dev.get_info(rs2.camera_info.name)} (S/N: {dev.get_info(rs2.camera_info.serial_number)})")
        
        # Stream some frames to warm up the camera
        print_status("Warming up camera (capturing 30 frames)...")
        for i in range(30):
            frames = pipe.wait_for_frames()
            progress = (i + 1) / 30
            bar_length = 20
            filled_length = int(round(bar_length * progress))
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            sys.stdout.write(f'\rWarm-up progress: |{bar}| {progress*100:.1f}% ')
            sys.stdout.flush()
            time.sleep(0.1)
        
        print("\n")  # Add a newline after warm-up progress
        
        # Check if the device can be cast to auto_calibrated_device
        try:
            print_status("Checking if device supports auto-calibration...")
            cal = rs2.auto_calibrated_device(dev)
            print_status("✓ Device supports auto-calibration")
        except Exception as e:
            print_status(f"✗ Device doesn't support auto-calibration: {e}")
            pipe.stop()
            return
        
        # Create frame queues for left and right streams
        left_stream_queue = FrameQueue(maxsize=30)
        right_stream_queue = FrameQueue(maxsize=30)
        
        # Collect frames for calibration
        print_status("Collecting frames for calibration...")
        for i in range(30):  # Collect 30 frames for each stream
            frames = pipe.wait_for_frames()
            
            # Get left and right infrared frames
            left_ir = frames.get_infrared_frame(1)  # Left IR
            right_ir = frames.get_infrared_frame(2)  # Right IR
            
            if left_ir and right_ir:
                left_stream_queue.enqueue(left_ir)
                right_stream_queue.enqueue(right_ir)
                
                progress = (i + 1) / 30
                bar_length = 20
                filled_length = int(round(bar_length * progress))
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                sys.stdout.write(f'\rCollecting frames: |{bar}| {progress*100:.1f}% ')
                sys.stdout.flush()
            
            time.sleep(0.1)
        
        print("\n")  # Add a newline after frame collection progress
        
        # Define parameters for focal length calibration
        # Target dimensions in millimeters - using your specified value
        target_width_mm = 450   # Width of the calibration target in mm
        target_height_mm = 450  # Height of the calibration target in mm
        adjust_both_sides = True  # Adjust both left and right cameras
        
        # Variables to store calibration results
        corrected_ratio = [0.0]  # Will be filled with the corrected ratio
        target_tilt_angle = [0.0]  # Will be filled with the target tilt angle
        
        print_status(f"Starting focal length calibration with target dimensions: {target_width_mm}mm x {target_height_mm}mm")
        print_status("Place the camera in front of the calibration target")
        print_status("Ensure the target fills a significant portion of the camera's view")
        print_status("Do not move the camera during calibration")
        
        # Print countdown before starting calibration
        for i in range(3, 0, -1):
            print_status(f"Calibration will start in {i}...")
            time.sleep(1)
        
        print("")  # Add blank line before progress bar
        
        try:
            # Run focal length calibration
            res = cal.run_focal_length_calibration(
                left_stream_queue.queue,  # Left stream frames
                right_stream_queue.queue,  # Right stream frames
                target_width_mm,  # Width of the calibration target
                target_height_mm,  # Height of the calibration target
                adjust_both_sides,  # Adjust both cameras
                corrected_ratio,  # Output parameter for corrected ratio
                target_tilt_angle,  # Output parameter for tilt angle
                progress_callback  # Progress callback function
            )
            
            print("\n")  # Add a newline after calibration progress
            
            # Check results
            print_status(f"Focal length calibration completed with result: {res}")
            print_status(f"Corrected ratio: {corrected_ratio[0]}")
            print_status(f"Target tilt angle: {target_tilt_angle[0]} degrees")
            
            if isinstance(res, list) and len(res) > 0 and res[0] == 0:
                print_status("✅ Focal length calibration SUCCESSFUL")
            else:
                print_status("⚠️ Focal length calibration completed with warnings or errors")
                
        except RuntimeError as e:
            print("\n")  # Add a newline after calibration progress
            print_status(f"❌ Focal length calibration failed: {e}")
            print("\nTroubleshooting steps:")
            print("1. Verify the target dimensions are accurate")
            print("2. Ensure the calibration target is clearly visible in both cameras")
            print("3. Ensure the target is well-lit and has good contrast")
            print("4. Ensure the camera is stationary during calibration")
            print("5. Try restarting the camera or reconnecting it")
            print("6. Update firmware if available")
            print("7. Try a different USB port (preferably USB 3.0)")
        
        # Verify calibration by checking depth readings
        print_status("Verifying calibration with test frames...")
        try:
            # Enable depth stream for verification
            cfg.enable_stream(rs2.stream.depth, 640, 480, rs2.format.z16, 30)
            pipe.stop()
            profile = pipe.start(cfg)
            
            # Get depth frames
            for i in range(10):  # Skip some frames to let the camera stabilize
                frames = pipe.wait_for_frames()
                
            # Get depth frame
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            if not depth_frame:
                print_status("No depth frame received for verification")
            else:
                # Get the depth value at the center of the image
                width = depth_frame.get_width()
                height = depth_frame.get_height()
                center_x = width // 2
                center_y = height // 2
                
                # Get depth values around the center (5x5 square)
                depth_values = []
                for x in range(center_x - 2, center_x + 3):
                    for y in range(center_y - 2, center_y + 3):
                        if 0 <= x < width and 0 <= y < height:
                            depth_values.append(depth_frame.get_distance(x, y) * 1000)  # Convert to mm
                
                # Calculate average depth
                avg_depth = np.mean(depth_values)
                
                print_status(f"Measured depth at center after calibration: {avg_depth:.1f}mm")
                
                # Get IR frames for visual check
                left_ir = frames.get_infrared_frame(1)
                right_ir = frames.get_infrared_frame(2)
                
                if left_ir and right_ir:
                    print_status("✅ Successfully receiving IR frames after calibration")
                else:
                    print_status("⚠️ Issue with IR frames after calibration")
        except Exception as e:
            print_status(f"Error during calibration verification: {e}")
            
    except Exception as e:
        print_status(f"Error during setup: {e}")
    finally:
        # Stop the pipeline
        print_status("Stopping pipeline...")
        pipe.stop()
        print_status("Calibration process complete")

if __name__ == "__main__":
    main()