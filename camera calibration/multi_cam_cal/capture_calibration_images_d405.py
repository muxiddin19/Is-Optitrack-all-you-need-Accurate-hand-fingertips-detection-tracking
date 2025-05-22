import pyrealsense2 as rs
import numpy as np
import cv2
import os

def capture_d405_infrared():
    """Capture D405 infrared images for calibration"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # D405 serial number (your device)
    config.enable_device("230322270171")
    
    # Enable infrared stream (acts like grayscale camera)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    
    try:
        pipeline.start(config)
        
        os.makedirs("d405_ir_calibration", exist_ok=True)
        count = 0
        
        print("D405 Infrared Capture - SPACE: capture, ESC: exit")
        
        while True:
            frames = pipeline.wait_for_frames()
            ir_frame = frames.get_infrared_frame(1)  # Get first infrared stream
            
            if ir_frame:
                # Convert to numpy array
                ir_image = np.asanyarray(ir_frame.get_data())
                
                # Convert to 3-channel for compatibility
                ir_image_3ch = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2RGB)
                
                cv2.imshow('D405 Infrared', ir_image_3ch)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    filename = f"d405_ir_calibration/ir_calib_{count:03d}.png"
                    cv2.imwrite(filename, ir_image_3ch)
                    print(f"Saved: {filename}")
                    count += 1
                elif key == 27:  # ESC
                    break
                    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Captured {count} infrared images")

# Run the function
capture_d405_infrared()