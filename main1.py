import time
import cv2
import json
import tkinter as tk
from tkinter import scrolledtext
import threading
import numpy as np
from pynput.keyboard import Controller, Key
from typing import Dict, Set, Tuple, Optional
from collections import defaultdict

from src.camera_manager import CameraManager
from src.hand_tracker import HandTracker
from src.keyboard_manager import KeyboardManager
import src.visualization_utils as viz_utils


class EnhancedVirtualKeyboardApp:
    """Enhanced virtual keyboard with multi-finger support and depth interpolation."""
    
    # Constants
    KEY_MAP = {
        "BACKSPACE": Key.backspace,
        "ENTER": Key.enter,
        "SPACE": Key.space,
        "SHIFT": Key.shift,
        "CTRL": Key.ctrl,
        "ALT": Key.alt,
        "WIN": Key.cmd,
        "ESC": Key.esc,
        "DEL": Key.delete,
        "UP": Key.up,
        "DOWN": Key.down,
        "LEFT": Key.left,
        "RIGHT": Key.right,
        "TAB": Key.tab,
        "CAPS": Key.caps_lock,
    }
    
    def __init__(self):
        # Configuration
        self.annotation_filename = 'assets/keyboard_annotations.json'
        self.thresholds_filename = 'assets/key_thresholds.json'
        self.points_per_key = 4
        self.release_threshold = 0.440
        
        # Enhanced depth processing
        self.depth_interpolation_enabled = True
        self.depth_history = defaultdict(list)  # Store depth history per finger
        self.depth_history_size = 5
        
        # Multi-finger support
        self.active_fingers = {}  # finger_id -> (hand_idx, finger_name, key_name, timestamp)
        self.finger_timeout = 450  # milliseconds
        
        # State variables
        self.key_depth_thresholds = {}
        self.last_pressed_keys = set()
        
        # Components
        self.keyboard = Controller()
        self.camera_manager = CameraManager()
        self.hand_tracker = HandTracker()
        self.keyboard_manager = KeyboardManager(
            annotation_filename=self.annotation_filename,
            points_per_key=self.points_per_key
        )
        
        # UI thread
        self.ui_thread = None
    
    def load_key_thresholds(self) -> bool:
        """Load key depth thresholds from file."""
        try:
            with open(self.thresholds_filename, 'r') as f:
                data = json.load(f)
                self.key_depth_thresholds = {key: tuple(value) for key, value in data.items()}
            print(f"Successfully loaded key thresholds from '{self.thresholds_filename}'.")
            return True
        except Exception as e:
            print(f"Error loading thresholds: {e}")
            return False
    
    def interpolate_depth(self, finger_id: str, raw_depth: float, finger_pos: Tuple[int, int]) -> float:
        """
        Interpolate depth using multiple methods when RealSense fails.
        """
        if raw_depth > 0:
            # Valid depth reading
            self.depth_history[finger_id].append(raw_depth)
            if len(self.depth_history[finger_id]) > self.depth_history_size:
                self.depth_history[finger_id].pop(0)
            return raw_depth
        
        # No valid depth - use interpolation methods
        interpolated_depth = None
        
        # Method 1: Use depth history for this finger
        if finger_id in self.depth_history and self.depth_history[finger_id]:
            interpolated_depth = np.median(self.depth_history[finger_id])
        
        # Method 2: Use neighboring pixel depths
        if interpolated_depth is None:
            interpolated_depth = self.get_neighboring_depth(finger_pos)
        
        # Method 3: Use hand plane estimation
        #if interpolated_depth is None:
            #interpolated_depth = self.estimate_hand_plane_depth(finger_pos)
        
        # Method 4: Use other finger depths as reference
        # if interpolated_depth is None:
        #     interpolated_depth = self.estimate_from_other_fingers(finger_id)
        
        return interpolated_depth if interpolated_depth and interpolated_depth > 0 else 0.458  # Default depth
    
    def get_neighboring_depth(self, finger_pos: Tuple[int, int], radius: int = 5) -> Optional[float]:
        """Get depth from neighboring pixels around finger position."""
        if not hasattr(self, 'current_depth_frame'):
            return None
            
        px, py = finger_pos
        depth_frame = self.current_depth_frame
        
        valid_depths = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
                    depth = depth_frame.get_distance(x, y)
                    if depth > 0:
                        valid_depths.append(depth)
        
        return np.median(valid_depths) if valid_depths else None
    
    # def estimate_hand_plane_depth(self, finger_pos: Tuple[int, int]) -> Optional[float]:
    #     """Estimate depth based on hand plane (simplified)."""
    #     # Simple assumption: hand is roughly planar
    #     # Use palm area depth as reference
    #     if hasattr(self, 'hand_landmarks'):
    #         try:
    #             # Get palm center depth (landmark 0)
    #             palm_landmark = self.hand_landmarks.landmark[0]
    #             palm_x = int(palm_landmark.x * self.current_depth_frame.get_width())
    #             palm_y = int(palm_landmark.y * self.current_depth_frame.get_height())
                
    #             palm_depth = self.current_depth_frame.get_distance(palm_x, palm_y)
    #             if palm_depth > 0:
    #                 # Fingers are typically slightly closer than palm
    #                 return palm_depth - 0.02  # 2cm closer
    #         except:
    #             pass
        
    #     return None
    
    # def estimate_from_other_fingers(self, current_finger: str) -> Optional[float]:
    #     """Estimate depth using other fingers' depths."""
    #     other_depths = []
        
    #     for finger_id, history in self.depth_history.items():
    #         if finger_id != current_finger and history:
    #             other_depths.append(np.median(history))
        
    #     return np.median(other_depths) if other_depths else None
    
    def is_finger_pressing_key(self, key_data: dict, finger_depth: float) -> bool:
        """Check if finger is pressing a key based on depth threshold."""
        key_name = key_data.get("key")
        if not key_name:
            return False
        
        threshold = self.key_depth_thresholds.get(key_name)
        if not threshold:
            return False
        
        min_depth, max_depth = threshold
        return min_depth <= finger_depth
    
    def cleanup_expired_fingers(self):
        """Remove expired active fingers."""
        current_time = time.time() * 1000
        expired_fingers = []
        
        for finger_id, (hand_idx, finger_name, key_name, timestamp) in self.active_fingers.items():
            if current_time - timestamp > self.finger_timeout:
                expired_fingers.append(finger_id)
        
        for finger_id in expired_fingers:
            del self.active_fingers[finger_id]
    
    def process_finger_tips(self, hand_landmarks, hand_idx: int, color_image, 
                           aligned_depth_frame, depth_frame_dims) -> Set[str]:
        """Process finger tips for a single hand and return pressed keys."""
        current_pressed_keys = set()
        current_time = time.time() * 1000
        
        # Store current depth frame for interpolation
        self.current_depth_frame = aligned_depth_frame
        self.hand_landmarks = hand_landmarks
        
        finger_tips = {
            'thumb': self.hand_tracker.get_thumb_finger_tip(hand_landmarks, color_image.shape),
            'index': self.hand_tracker.get_index_finger_tip(hand_landmarks, color_image.shape),
            'middle': self.hand_tracker.get_middle_finger_tip(hand_landmarks, color_image.shape),
            'ring': self.hand_tracker.get_ring_finger_tip(hand_landmarks, color_image.shape),
            'pinky': self.hand_tracker.get_pinky_finger_tip(hand_landmarks, color_image.shape),
        }
        
        for finger_name, (tip_coords, _) in finger_tips.items():
            if not tip_coords:
                continue
            
            px, py = tip_coords
            depth_frame_width, depth_frame_height = depth_frame_dims
            clamped_px = max(0, min(px, depth_frame_width - 1))
            clamped_py = max(0, min(py, depth_frame_height - 1))
            
            # Get raw depth
            raw_depth = aligned_depth_frame.get_distance(clamped_px, clamped_py)
            
            # Create unique finger ID
            finger_id = f"hand_{hand_idx}_{finger_name}"
            
            # Use enhanced depth interpolation
            final_depth = self.interpolate_depth(finger_id, raw_depth, (clamped_px, clamped_py))
            
            # Enhanced visualization with depth quality indicator
            color = (0, 255, 0) if raw_depth > 0 else (0, 0, 255)  # Green if direct, red if interpolated
            cv2.circle(color_image, (px, py), 8, color, -1)
            
            # Show both raw and interpolated depth
            depth_text = f"{finger_name}: {final_depth:.3f}m"
            if raw_depth == 0:
                depth_text += " (est)"
            cv2.putText(color_image, depth_text, (px + 10, py), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Check if this finger is currently active
            if finger_id in self.active_fingers:
                _, _, active_key, _ = self.active_fingers[finger_id]
                if final_depth < self.release_threshold:
                    print(f"Key {active_key} released by {finger_name}!")
                    del self.active_fingers[finger_id]
                else:
                    current_pressed_keys.add(active_key)
                continue
            
            # Check for new key presses
            finger_point = (px, py)
            for key_data in self.keyboard_manager.get_annotated_keys():
                if self.keyboard_manager.is_point_in_keycap(finger_point, key_data):
                    if self.is_finger_pressing_key(key_data, final_depth):
                        key_name = key_data['key']
                        self.active_fingers[finger_id] = (hand_idx, finger_name, key_name, current_time)
                        current_pressed_keys.add(key_name)
                        print(f"Key {key_name} pressed by {finger_name}!")
                        break
        
        return current_pressed_keys
    
    def simulate_key_presses(self, current_pressed_keys: Set[str]):
        """Simulate key presses and releases using pynput."""
        newly_pressed = current_pressed_keys - self.last_pressed_keys
        newly_released = self.last_pressed_keys - current_pressed_keys
        
        for key_str in newly_pressed:
            self._press_key(key_str)
        
        for key_str in newly_released:
            self._release_key(key_str)
        
        self.last_pressed_keys = current_pressed_keys
    
    def _press_key(self, key_str: str):
        """Press a single key."""
        try:
            if key_str in self.KEY_MAP:
                self.keyboard.press(self.KEY_MAP[key_str])
            elif len(key_str) == 1:
                self.keyboard.press(key_str.lower())
        except Exception as e:
            print(f"Could not press key '{key_str}': {e}")
    
    def _release_key(self, key_str: str):
        """Release a single key."""
        try:
            if key_str in self.KEY_MAP:
                self.keyboard.release(self.KEY_MAP[key_str])
            elif len(key_str) == 1:
                self.keyboard.release(key_str.lower())
        except Exception as e:
            print(f"Could not release key '{key_str}': {e}")
    
    def cleanup(self):
        """Clean up resources and release any pressed keys."""
        print("Application stopping...")
        
        # Release all pressed keys
        for key_str in self.last_pressed_keys:
            self._release_key(key_str)
        
        self.camera_manager.stop_stream()
        self.hand_tracker.close()
        cv2.destroyAllWindows()
        print("Application stopped.")
    
    def start_ui_thread(self):
        """Start the UI in a separate thread."""
        self.ui_thread = threading.Thread(target=self._run_ui, daemon=True)
        self.ui_thread.start()
    
    def _run_ui(self):
        """Run the tkinter UI in a separate thread."""
        try:
            root = tk.Tk()
            root.title("Enhanced Virtual Keyboard Output")
            root.geometry("600x400")

            main_frame = tk.Frame(root, padx=10, pady=10)
            main_frame.pack(fill=tk.BOTH, expand=True)

            label = tk.Label(main_frame, text="Enhanced Virtual Keyboard - All fingers supported!")
            label.pack(pady=(0, 5))

            text_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=60, height=20)
            text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            text_area.focus()

            root.mainloop()
        except Exception as e:
            print(f"Error in UI thread: {e}")
    
    def run(self):
        """Main application loop."""
        # Initialize
        if not self.load_key_thresholds():
            return
        
        # Start UI
        self.start_ui_thread()
        
        try:
            if not self.camera_manager.start_stream():
                print("Failed to start camera stream. Exiting.")
                return
            
            print("🚀 Enhanced Virtual Keyboard Started!")
            print("✅ Multi-finger support enabled")
            print("✅ Depth interpolation enabled")
            print("✅ All fingers have equal priority")
            
            while True:
                color_image, aligned_depth_frame, depth_frame_dims = self.camera_manager.get_frames()
                if color_image is None or aligned_depth_frame is None:
                    continue
                
                # Clean up expired fingers
                self.cleanup_expired_fingers()
                
                current_pressed_keys = set()
                results = self.hand_tracker.process_frame(color_image)
                
                if results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        self.hand_tracker.draw_landmarks(color_image, hand_landmarks)
                        
                        hand_pressed_keys = self.process_finger_tips(
                            hand_landmarks, hand_idx, color_image, 
                            aligned_depth_frame, depth_frame_dims
                        )
                        current_pressed_keys.update(hand_pressed_keys)
                
                # Simulate key presses
                self.simulate_key_presses(current_pressed_keys)
                
                # Enhanced visualization
                active_keys = {finger_data[2] for finger_data in self.active_fingers.values()}
                viz_utils.draw_keycap_annotations(
                    color_image, 
                    self.keyboard_manager.get_annotated_keys(), 
                    active_keys,
                    self.points_per_key
                )
                
                # Show active finger count
                cv2.putText(color_image, f"Active fingers: {len(self.active_fingers)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Enhanced Virtual Keyboard Interface', color_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.cleanup()


if __name__ == "__main__":
    app = EnhancedVirtualKeyboardApp()
    app.run()