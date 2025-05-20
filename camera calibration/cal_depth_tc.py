import pyrealsense2 as rs2
import time
import sys
import os
import numpy as np

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

def main():
    print_status("Starting RealSense Tare Calibration")
    print_status("Initializing RealSense pipeline...")
    
    # Initialize pipeline and config
    pipe = rs2.pipeline()
    cfg = rs2.config()
    
    try:
        # Configure depth stream
        print_status("Configuring depth stream (256x144 @ 90fps)...")
        cfg.enable_stream(rs2.stream.depth, 256, 144, rs2.format.z16, 90)
        
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

        # Define parameters for tare calibration
        timeout_ms = 60000  # 60 seconds timeout
        json_content = ""   # Empty string for default calibration settings
        
        # Ground truth distance in millimeters
        # Using actual measured distance to the target
        ground_truth = 550  # 450mm as specified in your actual setup
        
        print_status(f"Starting tare calibration with ground truth distance: {ground_truth}mm")
        print_status(f"Place the camera exactly {ground_truth}mm from a flat wall")
        print_status("Ensure the wall is perpendicular to the camera's line of sight")
        print_status("Do not move the camera during calibration")
        
        # Print countdown before starting calibration
        for i in range(3, 0, -1):
            print_status(f"Calibration will start in {i}...")
            time.sleep(1)
        
        print("")  # Add blank line before progress bar
        try:
            # Run tare calibration with the correct parameter order
            # Note: Depending on your pyrealsense2 version, the parameter order might be:
            # run_tare_calibration(self, ground_truth, json_content, timeout_ms)
            # or
            # run_tare_calibration(self, ground_truth, timeout_ms, json_content, callback)
            # Try both if one doesn't work
            
            try:
                # Try the first parameter order (older API versions)
                res = cal.run_tare_calibration(ground_truth, json_content, timeout_ms)
                print_status("Used API version with 3 parameters")
            except TypeError:
                try:
                    # Try the second parameter order (newer API versions with callback)
                    res = cal.run_tare_calibration(ground_truth, timeout_ms, json_content, progress_callback)
                    print_status("Used API version with 4 parameters including callback")
                except TypeError:
                    # Try the third possible parameter order
                    res = cal.run_tare_calibration(ground_truth, json_content, progress_callback, timeout_ms)
                    print_status("Used API version with alternate 4 parameter order")
            
            print("\n")  # Add a newline after calibration progress
            
            # Check results
            print_status(f"Tare calibration completed with result: {res}")
            
            if isinstance(res, list) and len(res) > 0 and res[0] == 0:
                print_status("✅ Tare calibration SUCCESSFUL")
            else:
                print_status("⚠️ Tare calibration completed with warnings or errors")
                
        except RuntimeError as e:
            print("\n")  # Add a newline after calibration progress
            print_status(f"❌ Tare calibration failed: {e}")
            print("\nTroubleshooting steps:")
            print("1. Verify the exact distance to wall is accurate")
            print("2. Ensure the wall is flat and perpendicular to camera")
            print("3. Ensure the camera is stationary during calibration")
            print("4. Try restarting the camera or reconnecting it")
            print("5. Update firmware if available")
            print("6. Try a different USB port (preferably USB 3.0)")
            print("7. Ensure consistent lighting (not too bright, not too dark)")
            
        # Verify calibration by checking depth readings
        print_status("Verifying calibration with depth readings...")
        try:
            # Get depth frames
            for i in range(10):  # Skip some frames to let the camera stabilize
                frames = pipe.wait_for_frames()
                
            # Get depth frame
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            if not depth_frame:
                print_status("No depth frame received")
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
                
                print_status(f"Measured depth at center: {avg_depth:.1f}mm")
                print_status(f"Expected depth (ground truth): {ground_truth}mm")
                print_status(f"Depth error: {abs(avg_depth - ground_truth):.1f}mm")
                
                if abs(avg_depth - ground_truth) < 10:  # Less than 1cm error
                    print_status("✅ Depth verification passed!")
                else:
                    print_status("⚠️ Depth verification shows significant error")
        except Exception as e:
            print_status(f"Error during depth verification: {e}")
            
    except Exception as e:
        print_status(f"Error during setup: {e}")
    finally:
        # Stop the pipeline
        print_status("Stopping pipeline...")
        pipe.stop()
        print_status("Calibration process complete")

if __name__ == "__main__":
    main()