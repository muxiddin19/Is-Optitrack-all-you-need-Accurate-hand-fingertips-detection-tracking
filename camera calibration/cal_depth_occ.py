import pyrealsense2 as rs2
import time
import sys
import os

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
    print_status("Starting RealSense Depth Camera Calibration")
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

        # Define parameters
        timeout_ms = 60000  # 60 seconds timeout
        json_content = ""   # Empty string for default calibration settings

        print_status("Starting on-chip calibration...")
        print_status("Keep the camera pointed at a flat surface 0.5-2m away")
        print_status("Do not move the camera during calibration")
        
        # Print countdown before starting calibration
        for i in range(3, 0, -1):
            print_status(f"Calibration will start in {i}...")
            time.sleep(1)
        
        print("")  # Add blank line before progress bar
        try:
            # Run on-chip calibration with the correct parameter order
            res, health = cal.run_on_chip_calibration(json_content, progress_callback, timeout_ms)
            print("\n")  # Add a newline after calibration progress
            
            # Check results
            print_status(f"Calibration completed with result code: {res}")
            print_status(f"Health values: {health}")
            
            if isinstance(res, list) and len(res) > 0 and res[0] == 0:
                print_status("✅ Calibration SUCCESSFUL")
            else:
                print_status("⚠️ Calibration completed with warnings or errors")
                
            print_status(f"Overall health score: {(health[0] + health[1])/2:.2f}")
        except RuntimeError as e:
            print("\n")  # Add a newline after calibration progress
            print_status(f"❌ Calibration failed: {e}")
            print("\nTroubleshooting steps:")
            print("1. Make sure the camera has a clear view of a flat surface")
            print("2. Ensure the camera is stationary during calibration")
            print("3. Try restarting the camera or reconnecting it")
            print("4. Update firmware if available")
            print("5. Try a different USB port (preferably USB 3.0)")
            print("6. Ensure consistent lighting (not too bright, not too dark)")
            print("7. Check power supply to the camera")
    except Exception as e:
        print_status(f"Error during setup: {e}")
    finally:
        # Stop the pipeline
        print_status("Stopping pipeline...")
        pipe.stop()
        print_status("Calibration process complete")

if __name__ == "__main__":
    main()