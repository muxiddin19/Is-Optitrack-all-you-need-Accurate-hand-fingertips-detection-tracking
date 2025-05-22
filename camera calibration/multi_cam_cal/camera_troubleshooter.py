#!/usr/bin/env python3
"""
Camera troubleshooter to diagnose and fix common camera issues
"""

import cv2
import pyrealsense2 as rs
import numpy as np
import sys
import time

class CameraTroubleshooter:
    """Diagnose and fix common camera issues"""
    
    def __init__(self):
        self.test_results = {}
    
    def test_opencv_version(self):
        """Test OpenCV version and build info"""
        print("=== OpenCV Information ===")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Build info:")
        build_info = cv2.getBuildInformation()
        
        # Extract relevant build info
        for line in build_info.split('\n'):
            if any(keyword in line.lower() for keyword in ['video', 'ffmpeg', 'gstreamer', 'msmf', 'dshow']):
                print(f"  {line.strip()}")
        
        self.test_results['opencv'] = True
        return True
    
    def test_realsense_sdk(self):
        """Test RealSense SDK"""
        print("\n=== RealSense SDK Test ===")
        try:
            print(f"pyrealsense2 version: {rs.__version__}")
            
            # Test context creation
            ctx = rs.context()
            devices = ctx.query_devices()
            
            print(f"Found {len(devices)} RealSense device(s)")
            
            for i, device in enumerate(devices):
                name = device.get_info(rs.camera_info.name)
                serial = device.get_info(rs.camera_info.serial_number)
                firmware = device.get_info(rs.camera_info.firmware_version)
                print(f"  Device {i}: {name}")
                print(f"    Serial: {serial}")
                print(f"    Firmware: {firmware}")
            
            self.test_results['realsense_sdk'] = True
            return len(devices) > 0
            
        except Exception as e:
            print(f"❌ RealSense SDK error: {e}")
            self.test_results['realsense_sdk'] = False
            return False
    
    def test_webcam_backends(self):
        """Test different webcam backends"""
        print("\n=== Webcam Backend Test ===")
        
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_V4L2, "Video4Linux2"),
            (cv2.CAP_ANY, "Any Available")
        ]
        
        working_backends = []
        
        for backend_id, backend_name in backends:
            print(f"\nTesting {backend_name}...")
            
            for camera_id in range(3):  # Test first 3 camera IDs
                try:
                    cap = cv2.VideoCapture(camera_id, backend_id)
                    
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            height, width = frame.shape[:2]
                            print(f"  ✅ Camera {camera_id}: {width}x{height}")
                            working_backends.append((backend_id, backend_name, camera_id))
                        else:
                            print(f"  ❌ Camera {camera_id}: Can't read frame")
                    else:
                        print(f"  ❌ Camera {camera_id}: Can't open")
                    
                    cap.release()
                    
                except Exception as e:
                    print(f"  ❌ Camera {camera_id}: Exception - {e}")
        
        self.test_results['webcam_backends'] = working_backends
        return len(working_backends) > 0
    
    def test_realsense_streaming(self):
        """Test RealSense camera streaming"""
        print("\n=== RealSense Streaming Test ===")
        
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                print("❌ No RealSense devices found")
                return False
            
            working_devices = []
            
            for device in devices:
                serial = device.get_info(rs.camera_info.serial_number)
                name = device.get_info(rs.camera_info.name)
                
                print(f"\nTesting {name} ({serial})...")
                
                try:
                    # Create pipeline
                    pipeline = rs.pipeline()
                    config = rs.config()
                    
                    # Enable device
                    config.enable_device(serial)
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    
                    # Start streaming
                    profile = pipeline.start(config)
                    
                    # Get a few frames to test stability
                    for i in range(5):
                        frames = pipeline.wait_for_frames(timeout_ms=5000)
                        color_frame = frames.get_color_frame()
                        depth_frame = frames.get_depth_frame()
                        
                        if color_frame and depth_frame:
                            if i == 0:  # Print info for first frame
                                color_image = np.asanyarray(color_frame.get_data())
                                depth_image = np.asanyarray(depth_frame.get_data())
                                print(f"  ✅ Color: {color_image.shape}")
                                print(f"  ✅ Depth: {depth_image.shape}, range: {depth_image.min()}-{depth_image.max()}")
                        else:
                            print(f"  ❌ Frame {i}: Missing color or depth")
                    
                    pipeline.stop()
                    working_devices.append(serial)
                    print(f"  ✅ {name} streaming successful")
                    
                except Exception as e:
                    print(f"  ❌ {name} streaming failed: {e}")
                    try:
                        pipeline.stop()
                    except:
                        pass
            
            self.test_results['realsense_streaming'] = working_devices
            return len(working_devices) > 0
            
        except Exception as e:
            print(f"❌ RealSense streaming test failed: {e}")
            return False
    
    def test_camera_properties(self, camera_id=0, backend=cv2.CAP_DSHOW):
        """Test camera properties and capabilities"""
        print(f"\n=== Camera Properties Test (ID: {camera_id}) ===")
        
        try:
            cap = cv2.VideoCapture(camera_id, backend)
            
            if not cap.isOpened():
                print(f"❌ Cannot open camera {camera_id}")
                return False
            
            # Test basic properties
            properties = [
                (cv2.CAP_PROP_FRAME_WIDTH, "Width"),
                (cv2.CAP_PROP_FRAME_HEIGHT, "Height"),
                (cv2.CAP_PROP_FPS, "FPS"),
                (cv2.CAP_PROP_FOURCC, "FOURCC"),
                (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
                (cv2.CAP_PROP_CONTRAST, "Contrast"),
                (cv2.CAP_PROP_SATURATION, "Saturation"),
                (cv2.CAP_PROP_AUTOFOCUS, "Autofocus"),
                (cv2.CAP_PROP_FOCUS, "Focus"),
            ]
            
            print("Current properties:")
            for prop_id, prop_name in properties:
                try:
                    value = cap.get(prop_id)
                    print(f"  {prop_name}: {value}")
                except:
                    print(f"  {prop_name}: Not supported")
            
            # Test resolution changes
            print("\nTesting resolution changes:")
            resolutions = [(640, 480), (1280, 720), (1920, 1080)]
            
            for width, height in resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                ret, frame = cap.read()
                if ret:
                    print(f"  {width}x{height} -> {actual_width}x{actual_height} ✅")
                else:
                    print(f"  {width}x{height} -> Failed to capture ❌")
            
            cap.release()
            return True
            
        except Exception as e:
            print(f"❌ Camera properties test failed: {e}")
            return False
    
    def run_performance_test(self):
        """Run camera performance test"""
        print("\n=== Performance Test ===")
        
        # Test webcam performance if available
        if self.test_results.get('webcam_backends'):
            backend_id, backend_name, camera_id = self.test_results['webcam_backends'][0]
            print(f"Testing webcam performance ({backend_name})...")
            
            try:
                cap = cv2.VideoCapture(camera_id, backend_id)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                frame_count = 0
                start_time = time.time()
                test_duration = 5  # seconds
                
                while time.time() - start_time < test_duration:
                    ret, frame = cap.read()
                    if ret:
                        frame_count += 1
                    else:
                        break
                
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"  Webcam FPS: {fps:.2f} ({frame_count} frames in {elapsed:.2f}s)")
                
                cap.release()
                
            except Exception as e:
                print(f"  ❌ Webcam performance test failed: {e}")
        
        # Test RealSense performance if available
        if self.test_results.get('realsense_streaming'):
            serial = self.test_results['realsense_streaming'][0]
            print(f"Testing RealSense performance ({serial})...")
            
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                pipeline.start(config)
                
                frame_count = 0
                start_time = time.time()
                test_duration = 5  # seconds
                
                while time.time() - start_time < test_duration:
                    try:
                        frames = pipeline.wait_for_frames(timeout_ms=1000)
                        color_frame = frames.get_color_frame()
                        if color_frame:
                            frame_count += 1
                    except:
                        break
                
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"  RealSense FPS: {fps:.2f} ({frame_count} frames in {elapsed:.2f}s)")
                
                pipeline.stop()
                
            except Exception as e:
                print(f"  ❌ RealSense performance test failed: {e}")
    
    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        print("\n" + "="*50)
        print("RECOMMENDATIONS")
        print("="*50)
        
        if not self.test_results.get('opencv'):
            print("❌ OpenCV issues detected:")
            print("   • Reinstall OpenCV: pip install opencv-python")
        
        if not self.test_results.get('realsense_sdk'):
            print("❌ RealSense SDK issues detected:")
            print("   • Install RealSense SDK: pip install pyrealsense2")
            print("   • Check USB connection and drivers")
        
        if not self.test_results.get('webcam_backends'):
            print("❌ No working webcam backends found:")
            print("   • Check camera connections")
            print("   • Try different USB ports")
            print("   • Update camera drivers")
            print("   • Close other applications using camera")
        else:
            working = self.test_results['webcam_backends']
            print("✅ Working webcam configurations:")
            for backend_id, backend_name, camera_id in working:
                print(f"   • Camera {camera_id} with {backend_name}")
            print("   Recommendation: Use DirectShow backend to avoid MSMF issues")
        
        if not self.test_results.get('realsense_streaming'):
            print("❌ RealSense streaming issues:")
            print("   • Check USB 3.0 connection")
            print("   • Update RealSense firmware")
            print("   • Close Intel RealSense Viewer if open")
        else:
            working = self.test_results['realsense_streaming']
            print("✅ Working RealSense devices:")
            for serial in working:
                print(f"   • Device: {serial}")
    
    def run_all_tests(self):
        """Run all diagnostic tests"""
        print("🔍 Starting comprehensive camera diagnostics...\n")
        
        self.test_opencv_version()
        self.test_realsense_sdk()
        self.test_webcam_backends()
        self.test_realsense_streaming()
        self.run_performance_test()
        self.generate_recommendations()
        
        print(f"\n🏁 Diagnostic completed!")

def main():
    troubleshooter = CameraTroubleshooter()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test mode
        print("🔍 Running quick camera check...\n")
        troubleshooter.test_realsense_sdk()
        troubleshooter.test_webcam_backends()
        troubleshooter.generate_recommendations()
    else:
        # Full diagnostic
        troubleshooter.run_all_tests()

if __name__ == "__main__":
    main()