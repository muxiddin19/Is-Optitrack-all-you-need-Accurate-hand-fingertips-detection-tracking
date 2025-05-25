# Optimal settings for training data
python data_capture_system.py --camera d415 --output-dir my_training_data

# Custom range and FPS
python data_capture_system.py --camera d415 --min-range 0.8 --max-range 1.5 --fps 60

# Uses your calibrated alignment
python data_capture_system.py --camera d405 --calibration calibration_results/calibration_results.json
