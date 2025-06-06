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

python synchronized_d415_capture.py --min-range 0.4 --max-range 0.58 --output-dir complete_capture

(realsense) E:\vscode\calibration>python synchronized_d415_capture.py --min-range 0.4 --max-range 0.58 --output-dir data/user01/2
![000000_synchronized_20250526_094327](https://github.com/user-attachments/assets/8e876481-6170-44d0-b61e-090dc8650120)


(realsense) E:\vscode\calibration>python synchronized_d415_capture.py --min-range 0.4 --max-range 0.58 --output-dir data/user01/1


![000000_synchronized_20250526_094915](https://github.com/user-attachments/assets/daacef52-85e6-49cc-9c61-53212bb1fd92)

(realsense) E:\vscode\calibration>python synchronized_d415_capture.py --min-range 0.4 --max-range 0.58 --output-dir data/user01/0

![000000_synchronized_20250526_095608](https://github.com/user-attachments/assets/b8b1b7ce-2498-46f1-96f9-186f12fbe315)

![image](https://github.com/user-attachments/assets/596d8c98-cacc-4b14-afcf-33e805fe6bc3)
![image](https://github.com/user-attachments/assets/57a83b9f-424c-4448-ad2e-131396bd0522)

