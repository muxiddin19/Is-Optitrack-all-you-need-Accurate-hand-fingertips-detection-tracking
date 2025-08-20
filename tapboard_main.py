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


class TapboardXApp:
    """Main application class for velocity-based virtual keyboard interface."""

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
    FINGERTIP_LANDMARKS = [4, 8, 12, 16, 20]

    def __init__(self):
        # Configuration
        self.annotation_filename = 'assets/keyboard_annotations.json'
        self.thresholds_filename = 'assets/key_thresholds.json'
        self.points_per_key = 4
        self.release_threshold = 0.290

        # Velocity-based detection parameters
        self.tap_velocity_threshold = -0.1     # m/s
        self.min_interaction_depth = 0.20
        self.max_interaction_depth = 0.35

        # State variables
        self.key_depth_thresholds = {}
        self.previous_finger_depths: Dict[Tuple[int, int], float] = {}
        self.finger_touch_states: Dict[Tuple[int, int], Optional[str]] = {}
        self.last_frame_time = time.time()
        self.typed_text = ""
        self.detected_key_events: list[str] = []
        self.current_displayed_keys: str = ""

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

    def _get_key_for_point(self, point: Tuple[int, int], depth_m: float) -> Optional[str]:
        """Finds the key under a given point at a given depth."""
        for key_data in self.keyboard_manager.get_annotated_keys():
            key_name = key_data.get("key")
            if not key_name:
                continue

            if (self.keyboard_manager.is_point_in_keycap(point, key_data) and
                    self.is_finger_pressing_key(key_data, depth_m)):
                return key_name
        return None

    def _update_typed_text(self):
        """Update typed text based on detected key events."""
        if not self.detected_key_events:
            return

        for key in self.detected_key_events:
            if key == "ENTER":
                self.typed_text += "\n"
            elif key == "BACKSPACE":
                self.typed_text = self.typed_text[:-1]
            elif key == "SPACE":
                self.typed_text += " "
            elif key not in ["SHIFT", "CTRL", "ALT", "WIN", "ESC", "DEL"]:
                # Regular character keys
                self.typed_text += key

    def _simulate_key_press(self, key_str: str):
        """Simulate a single key press using pynput."""
        try:
            if key_str in self.KEY_MAP:
                self.keyboard.press(self.KEY_MAP[key_str])
                self.keyboard.release(self.KEY_MAP[key_str])
            elif len(key_str) == 1:
                self.keyboard.press(key_str.lower())
                self.keyboard.release(key_str.lower())
        except Exception as e:
            print(f"Could not press key '{key_str}': {e}")

    def _process_frame(self, color_image, aligned_depth_frame, depth_frame_dims):
        """Process a single frame for hand tracking and key detection."""
        current_frame_time = time.time()
        delta_time = current_frame_time - self.last_frame_time
        self.last_frame_time = current_frame_time

        self.detected_key_events = []
        self.current_displayed_keys = ""
        is_touching_keyboard = False
        active_finger_ids = set()

        results = self.hand_tracker.process_frame(color_image)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.hand_tracker.draw_landmarks(color_image, hand_landmarks)

                for landmark_idx in self.FINGERTIP_LANDMARKS:
                    finger_id = (hand_idx, landmark_idx)
                    active_finger_ids.add(finger_id)

                    landmark = hand_landmarks.landmark[landmark_idx]
                    image_height, image_width, _ = color_image.shape
                    pixel_x = int(landmark.x * image_width)
                    pixel_y = int(landmark.y * image_height)

                    clamped_x = max(0, min(pixel_x, depth_frame_dims[0] - 1))
                    clamped_y = max(0, min(pixel_y, depth_frame_dims[1] - 1))

                    depth_m = aligned_depth_frame.get_distance(clamped_x, clamped_y)

                    viz_utils.draw_finger_tip_info(color_image, pixel_x, pixel_y, depth_m)

                    depth_velocity = 0
                    if finger_id in self.previous_finger_depths and delta_time > 0:
                        depth_change = depth_m - self.previous_finger_depths[finger_id]
                        depth_velocity = depth_change / delta_time

                    currently_touched_key = self._get_key_for_point((pixel_x, pixel_y), depth_m)
                    previously_touched_key = self.finger_touch_states.get(finger_id)

                    if currently_touched_key:
                        is_touching_keyboard = True
                        self.current_displayed_keys = currently_touched_key
                        if not previously_touched_key:
                            print(f"Key {currently_touched_key} - Touched by finger {finger_id} (Vel: {depth_velocity:.3f})")
                    elif previously_touched_key:  # Key was just released
                        print(f"Key {previously_touched_key} - Released by finger {finger_id} (Vel: {depth_velocity:.3f})")
                        if depth_velocity < self.tap_velocity_threshold:
                            print(f"Key {previously_touched_key} - Tapped")
                            self.detected_key_events.append(previously_touched_key)

                    self.finger_touch_states[finger_id] = currently_touched_key
                    self.previous_finger_depths[finger_id] = depth_m

        # Clean up states for fingers that are no longer detected
        lost_fingers = set(self.finger_touch_states.keys()) - active_finger_ids
        for finger_id in lost_fingers:
            del self.finger_touch_states[finger_id]
            if finger_id in self.previous_finger_depths:
                del self.previous_finger_depths[finger_id]

        # Update typed text and simulate key press
        self._update_typed_text()
        if self.detected_key_events:
            for key in self.detected_key_events:
                self._simulate_key_press(key)

        # Draw visualizations
        self._draw_visualizations(color_image, is_touching_keyboard)

        return color_image

    def _draw_visualizations(self, color_image, is_touching_keyboard):
        """Draw all visualizations on the color image."""
        # Draw keycap annotations
        viz_utils.draw_keycap_annotations(
            color_image,
            self.keyboard_manager.get_annotated_keys(),
            self.current_displayed_keys,
            self.points_per_key
        )
        # print(f"Active key: {self.current_displayed_keys}")

        # Display touching keyboard status
        status_text = "Fingertip Touching Keyboard: YES" if is_touching_keyboard else "Fingertip Touching Keyboard: NO"
        status_color = (0, 255, 0) if is_touching_keyboard else (0, 0, 255)
        cv2.putText(color_image, status_text, (10, color_image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    def cleanup(self):
        """Clean up resources."""
        print("Application stopping...")
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
            root.title("Velocity-Based Virtual Keyboard Output")
            root.geometry("600x400")

            main_frame = tk.Frame(root, padx=10, pady=10)
            main_frame.pack(fill=tk.BOTH, expand=True)

            label = tk.Label(main_frame, text="Click inside this box to start typing with the velocity-based virtual keyboard.")
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

                processed_image = self._process_frame(color_image, aligned_depth_frame, depth_frame_dims)
                cv2.imshow('Velocity-Based Virtual Keyboard Interface', processed_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()


if __name__ == "__main__":
    app = TapboardXApp()
    app.run()