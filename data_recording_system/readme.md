# Optimal settings for training data
python data_capture_system.py --camera d415 --output-dir my_training_data

# Custom range and FPS
python data_capture_system.py --camera d415 --min-range 0.8 --max-range 1.5 --fps 60

# Uses your calibrated alignment
python data_capture_system.py --camera d405 --calibration calibration_results/calibration_results.json

# All cameras with your optimized range
python multi_camera_video_capture.py --min-range 0.4 --max-range 0.58

# Custom output directory
python multi_camera_video_capture.py --output-dir my_close_range_data --min-range 0.4 --max-range 0.58

# Higher FPS for detailed capture
python multi_camera_video_capture.py --fps 60 --min-range 0.4 --max-range 0.58

# Just D415 + D405 (depth cameras)
python multi_camera_video_capture.py --cameras d415 d405

# D415 + platform cameras (RGB focus)
python multi_camera_video_capture.py --cameras d415 platform1 platform2

python multi_camera_video_capture.py --min-range 0.4 --max-range 0.58

python simple_multi_cam.py --min-range 0.4 --max-range 0.58 --output-dir test_capture

python synchronized_d415_capture3.py --min-range 0.4 --max-range 0.58 --output-dir complete_capture
