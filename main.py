import time
import cv2
import json
import tkinter as tk
from tkinter import scrolledtext
import threading
from pynput.keyboard import Controller, Key
from typing import Dict, Set, Tuple, Optional

from src.camera_manager import CameraManager
from src.hand_tracker import HandTracker
from src.keyboard_manager import KeyboardManager
import src.visualization_utils as viz_utils


class VirtualKeyboardApp:
    """Main application class for the virtual keyboard interface."""
    
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
        self.release_threshold = 0.447
        
        # State variables
        self.key_depth_thresholds = {}
        self.active_finger = None
        self.active_key = None
        self.active_hand = None
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
    
    def has_active_finger(self) -> bool:
        """Check if there's an active finger currently pressing a key."""
        return (self.active_hand is not None and 
                self.active_finger is not None and 
                self.active_key is not None)
    
    def reset_active_finger(self):
        """Reset the active finger state."""
        self.active_finger = None
        self.active_key = None
        self.active_hand = None
    
    def set_active_finger(self, hand_idx: int, finger_name: str, key_name: str):
        """Set the active finger state."""
        self.active_hand = hand_idx
        self.active_finger = finger_name
        self.active_key = key_name
    
    def process_finger_tips(self, hand_landmarks, hand_idx: int, color_image, 
                           aligned_depth_frame, depth_frame_dims) -> Set[str]:
        """Process finger tips for a single hand and return pressed keys."""
        current_pressed_keys = set()
        
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
            depth_m = aligned_depth_frame.get_distance(clamped_px, clamped_py)
            
            viz_utils.draw_finger_tip_info(color_image, px, py, depth_m)
            
            # Check if this finger is currently active and should be released
            if (self.has_active_finger() and 
                hand_idx == self.active_hand and 
                finger_name == self.active_finger):
                if depth_m < self.release_threshold:
                    print(f"Key {self.active_key} released!")
                    self.reset_active_finger()
                continue
            
            # Skip if there's already an active finger
            if self.has_active_finger():
                continue
            
            # Check for new key presses
            finger_point = (px, py)
            for key_data in self.keyboard_manager.get_annotated_keys():
                if self.keyboard_manager.is_point_in_keycap(finger_point, key_data):
                    if self.is_finger_pressing_key(key_data, depth_m):
                        key_name = key_data['key']
                        self.set_active_finger(hand_idx, finger_name, key_name)
                        current_pressed_keys.add(key_name)
                        print(f"Key {key_name} pressed!")
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
            root.title("Virtual Keyboard Output")
            root.geometry("600x400")

            main_frame = tk.Frame(root, padx=10, pady=10)
            main_frame.pack(fill=tk.BOTH, expand=True)

            label = tk.Label(main_frame, text="Click inside this box to start typing with the virtual keyboard.")
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
            
            while True:
                color_image, aligned_depth_frame, depth_frame_dims = self.camera_manager.get_frames()
                if color_image is None or aligned_depth_frame is None:
                    continue
                
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
                
                # Visualization
                viz_utils.draw_keycap_annotations(
                    color_image, 
                    self.keyboard_manager.get_annotated_keys(), 
                    self.active_key,
                    self.points_per_key
                )
                cv2.imshow('Virtual Keyboard Interface', color_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.cleanup()


if __name__ == "__main__":
    app = VirtualKeyboardApp()
    app.run()