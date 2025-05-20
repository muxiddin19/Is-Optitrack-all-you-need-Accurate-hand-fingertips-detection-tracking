import pyrealsense2 as rs2
import time

# Initialize pipeline and config
pipe = rs2.pipeline()
cfg = rs2.config()
cfg.enable_stream(rs2.stream.depth, 256, 144, rs2.format.z16, 90)

# Start pipeline and get device
print("Starting pipeline...")
profile = pipe.start(cfg)
dev = profile.get_device()

# Stream some frames to warm up the camera
print("Warming up camera...")
for i in range(30):
    frames = pipe.wait_for_frames()
    time.sleep(0.1)
    if i % 10 == 0:
        print(f"{i} frames captured")

# Check if the device can be cast to auto_calibrated_device
try:
    cal = rs2.auto_calibrated_device(dev)
    print("Device supports auto-calibration")
except Exception as e:
    print(f"Device doesn't support auto-calibration: {e}")
    pipe.stop()
    exit(1)

# Define callback function to track progress
def cb(progress):
    print('.', end='', flush=True)

# Define parameters
timeout_ms = 60000  # 60 seconds timeout
json_content = ""   # Empty string for default calibration settings

print("Starting calibration...")
try:
    # Run on-chip calibration with the correct parameter order
    res, health = cal.run_on_chip_calibration(json_content, cb, timeout_ms)
    print(f"\nCalibration completed with result: {res}")
    print(f"Health: {health}")
except RuntimeError as e:
    print(f"\nCalibration failed: {e}")
    print("Try the following:")
    print("1. Make sure the camera has a clear view of a flat surface")
    print("2. Ensure the camera is stationary during calibration")
    print("3. Try restarting the camera or reconnecting it")
    print("4. Update firmware if available")
finally:
    # Stop the pipeline
    pipe.stop()