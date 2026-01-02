#!/usr/bin/env python3
"""
Interactive tracking editor for manual correction of pose landmarks.

Allows frame-by-frame navigation and manual adjustment of marker positions.
Useful for correcting tracking errors that automatic algorithms can't fix.

Controls:
  Left/Right Arrow  - Previous/next frame
  Space             - Play/pause
  Ctrl+S            - Save changes and quit
  Q/Esc             - Quit (prompts to save if modified)
  R                 - Reset current frame to original
  C                 - Add delta barrier (stop propagating earlier changes)

  View (use keyboard for zoom to avoid conflicts):
    +/=             - Zoom in
    -               - Zoom out
    0               - Reset zoom and pan
    Up/Down         - Pan view vertically
    [/]             - Pan view horizontally

  Select symmetric pairs (then press ENTER to flip):
    S - Shoulders    E - Elbows      W - Wrists
    H - Hips         K - Knees       A - Ankles
    P - Pinky        I - Index       T - Thumb

  Enter             - Flip selected symmetric pair for current frame

  Mouse:
    Left click + drag       - Move marker (on marker) or pan view (not on marker)
    Shift + Left drag       - Move marker and entire downstream chain
    Right click             - Delete marker (set to NaN)
    Middle click            - Restore deleted marker (from original)

Delta Tracking:
  When you move a marker in frame N, the pixel difference from the original
  position is stored and automatically propagated to all subsequent frames.
  Multiple edits accumulate - if you edit frame 10 and frame 15, frame 20
  will have both corrections applied. Press C to add a "barrier" at the
  current frame, which stops deltas from earlier frames from propagating past
  that point. This lets you make different corrections in different sections.

Usage:
  edit_tracking.py <camera_directory>

Example:
  edit_tracking.py session/20251231/take1/camera1
"""

import argparse
import cv2
import h5py
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# MediaPipe landmark groups with shapes and colors
LANDMARK_GROUPS = {
    'face': {
        'indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'shape': 'circle',
        'color': (255, 200, 0),  # Cyan
        'name': 'Face'
    },
    'left_arm': {
        'indices': [11, 13, 15, 17, 19, 21],
        'shape': 'square',
        'color': (0, 255, 0),  # Green
        'name': 'Left Arm'
    },
    'right_arm': {
        'indices': [12, 14, 16, 18, 20, 22],
        'shape': 'square',
        'color': (0, 0, 255),  # Red
        'name': 'Right Arm'
    },
    'left_leg': {
        'indices': [23, 25, 27, 29, 31],
        'shape': 'triangle',
        'color': (0, 255, 255),  # Yellow
        'name': 'Left Leg'
    },
    'right_leg': {
        'indices': [24, 26, 28, 30, 32],
        'shape': 'triangle',
        'color': (255, 0, 255),  # Magenta
        'name': 'Right Leg'
    }
}

# Symmetric landmark pairs (left, right)
SYMMETRIC_PAIRS = [
    (1, 4),   # eye_inner
    (2, 5),   # eye
    (3, 6),   # eye_outer
    (7, 8),   # ear
    (9, 10),  # mouth
    (11, 12), # shoulder
    (13, 14), # elbow
    (15, 16), # wrist
    (17, 18), # pinky
    (19, 20), # index
    (21, 22), # thumb
    (23, 24), # hip
    (25, 26), # knee
    (27, 28), # ankle
    (29, 30), # heel
    (31, 32), # foot_index
]

# Key mappings for symmetric pairs (letter -> index in SYMMETRIC_PAIRS)
PAIR_KEYS = {
    's': 5,   # Shoulders (11, 12)
    'e': 6,   # Elbows (13, 14)
    'w': 7,   # Wrists (15, 16)
    'h': 11,  # Hips (23, 24)
    'k': 12,  # Knees (25, 26)
    'a': 13,  # Ankles (27, 28)
    'p': 8,   # Pinky (17, 18)
    'i': 9,   # Index (19, 20)
    't': 10,  # Thumb (21, 22)
}

# Hierarchical dependencies: when flipping a pair, also flip all dependent pairs
# Maps pair_index -> [dependent_pair_indices]
PAIR_DEPENDENCIES = {
    5: [6, 7, 8, 9, 10],  # Shoulder -> Elbow, Wrist, Pinky, Index, Thumb
    6: [7, 8, 9, 10],     # Elbow -> Wrist, Pinky, Index, Thumb
    7: [8, 9, 10],        # Wrist -> Pinky, Index, Thumb
    11: [12, 13, 14, 15], # Hip -> Knee, Ankle, Heel, Foot
    12: [13, 14, 15],     # Knee -> Ankle, Heel, Foot
    13: [14, 15],         # Ankle -> Heel, Foot
}

# Marker chain dependencies: maps marker_index -> [downstream_marker_indices]
# Used for Shift+drag to move entire chains
MARKER_CHAIN = {
    # Left arm
    11: [13, 15, 17, 19, 21],  # left_shoulder -> elbow, wrist, pinky, index, thumb
    13: [15, 17, 19, 21],      # left_elbow -> wrist, pinky, index, thumb
    15: [17, 19, 21],          # left_wrist -> pinky, index, thumb
    # Right arm
    12: [14, 16, 18, 20, 22],  # right_shoulder -> elbow, wrist, pinky, index, thumb
    14: [16, 18, 20, 22],      # right_elbow -> wrist, pinky, index, thumb
    16: [18, 20, 22],          # right_wrist -> pinky, index, thumb
    # Left leg
    23: [25, 27, 29, 31],      # left_hip -> knee, ankle, heel, foot
    25: [27, 29, 31],          # left_knee -> ankle, heel, foot
    27: [29, 31],              # left_ankle -> heel, foot
    # Right leg
    24: [26, 28, 30, 32],      # right_hip -> knee, ankle, heel, foot
    26: [28, 30, 32],          # right_knee -> ankle, heel, foot
    28: [30, 32],              # right_ankle -> heel, foot
}

# Skeleton connections
SKELETON_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (2, 7), (5, 8), (0, 9), (0, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# Landmark labels with L/R prefixes
LANDMARK_LABELS = [
    "N",   # 0: nose
    "LEI", "LE", "LEO",  # 1-3: left eye
    "REI", "RE", "REO",  # 4-6: right eye
    "LER", "RER",        # 7-8: ears
    "LM", "RM",          # 9-10: mouth
    "LS", "RS",          # 11-12: shoulders
    "LE", "RE",          # 13-14: elbows
    "LW", "RW",          # 15-16: wrists
    "LP", "RP",          # 17-18: pinky
    "LI", "RI",          # 19-20: index
    "LT", "RT",          # 21-22: thumb
    "LH", "RH",          # 23-24: hips
    "LK", "RK",          # 25-26: knees
    "LA", "RA",          # 27-28: ankles
    "LL", "RL",          # 29-30: heels
    "LF", "RF",          # 31-32: foot index
]


class TrackingEditor:
    def __init__(self, video_path, h5_path):
        self.video_path = Path(video_path)
        self.h5_path = Path(h5_path)

        # Load video
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load tracking data
        with h5py.File(h5_path, 'r') as f:
            tracks = f['tracks'][:]

        # Convert to (n_frames, n_nodes, 2)
        self.original_points = tracks[0].transpose(2, 1, 0).copy()
        self.points = self.original_points.copy()
        self.n_frames, self.n_nodes, _ = self.points.shape

        # State
        self.current_frame = 0
        self.playing = False
        self.modified = False
        self.modified_frames = set()

        # Delta tracking - stores pixel adjustments per frame per marker
        self.frame_deltas = {}  # {frame_idx: {marker_idx: (dx, dy)}}
        self.delta_barriers = set()  # Frames where delta propagation stops

        # Interaction state
        self.selected_marker = None
        self.dragging_marker = False
        self.dragging_chain = False  # True when Shift is held during drag
        self.drag_start_pos = None  # Initial position of dragged marker
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.selected_pair = None

        # Zoom and pan state
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # Display size
        max_display_width = 1600
        max_display_height = 900
        scale = min(max_display_width / self.width, max_display_height / self.height, 1.0)
        self.display_width = int(self.width * scale)
        self.display_height = int(self.height * scale)

        # Create tkinter window
        self.root = tk.Tk()
        self.root.title(f"Tracking Editor - {self.video_path.name}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Create canvas for image display
        self.canvas = tk.Canvas(self.root, width=self.display_width, height=self.display_height, bg='black')
        self.canvas.pack()

        # Bind events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Button-2>", self.on_middle_click)

        # Mouse wheel binding (cross-platform)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows/Mac
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down

        self.root.bind("<Key>", self.on_key_press)

        # Current displayed image
        self.photo_image = None

        print(f"Loaded: {self.video_path.name}")
        print(f"Frames: {self.n_frames}, Landmarks: {self.n_nodes}")
        print(f"Video resolution: {self.width}x{self.height}, FPS: {self.fps:.2f}")
        print(f"Display window: {self.display_width}x{self.display_height}")
        print("\nControls:")
        print("  Left/Right Arrow: Previous/Next frame")
        print("  Space: Play/Pause")
        print("  Ctrl+S: Save and quit")
        print("  Q/Esc: Quit")
        print("  R: Reset current frame")
        print("  C: Add delta barrier at current frame")
        print("\n  View:")
        print("    +/= : Zoom in")
        print("    - : Zoom out")
        print("    0 : Reset zoom and pan")
        print("    Mouse wheel: Zoom in/out")
        print("    Up/Down/[/]: Pan view")
        print("\n  Select pairs: S=Shoulders E=Elbows W=Wrists H=Hips K=Knees A=Ankles")
        print("                P=Pinky I=Index T=Thumb")
        print("  Enter: Flip selected pair")
        print("\n  Mouse:")
        print("    Left drag: Move marker (on marker) or pan view (not on marker)")
        print("    Shift+Left drag: Move marker with entire downstream chain")
        print("    Right click: Delete marker")
        print("    Middle click: Restore marker")
        print("\n  Delta Tracking: Edits propagate forward; barriers stop propagation")
        print()

    def window_to_frame_coords(self, x, y):
        """Transform window coordinates to frame coordinates through viewport."""
        # Calculate viewport bounds in frame space
        viewport_width = self.width / self.zoom
        viewport_height = self.height / self.zoom

        # Center point in frame space
        center_x = self.width / 2 - self.pan_x / self.zoom
        center_y = self.height / 2 - self.pan_y / self.zoom

        # Viewport bounds
        view_left = center_x - viewport_width / 2
        view_top = center_y - viewport_height / 2

        # Transform window coords to frame coords
        frame_x = view_left + (x / self.display_width) * viewport_width
        frame_y = view_top + (y / self.display_height) * viewport_height

        return frame_x, frame_y

    def frame_to_window_coords(self, frame_x, frame_y):
        """Transform frame coordinates to window coordinates through viewport."""
        # Calculate viewport bounds in frame space
        viewport_width = self.width / self.zoom
        viewport_height = self.height / self.zoom

        # Center point in frame space
        center_x = self.width / 2 - self.pan_x / self.zoom
        center_y = self.height / 2 - self.pan_y / self.zoom

        # Viewport bounds
        view_left = center_x - viewport_width / 2
        view_top = center_y - viewport_height / 2

        # Transform frame coords to window coords
        window_x = ((frame_x - view_left) / viewport_width) * self.display_width
        window_y = ((frame_y - view_top) / viewport_height) * self.display_height

        return window_x, window_y

    def on_mouse_down(self, event):
        """Handle mouse button press."""
        frame_x, frame_y = self.window_to_frame_coords(event.x, event.y)

        # Check if clicking on a marker
        max_dist = 15 / self.zoom if self.zoom > 1.0 else 15
        self.selected_marker = self.find_nearest_marker(frame_x, frame_y, max_dist=max_dist)

        if self.selected_marker is not None:
            self.dragging_marker = True
            self.panning = False
            # Store initial position for chain dragging
            self.drag_start_pos = self.points[self.current_frame, self.selected_marker].copy()
            # Check if Shift is held (bit 0 of event.state)
            self.dragging_chain = (event.state & 0x1) != 0
            if self.dragging_chain and self.selected_marker in MARKER_CHAIN:
                print(f"Chain drag mode: moving {LANDMARK_NAMES[self.selected_marker]} + {len(MARKER_CHAIN[self.selected_marker])} downstream markers")
        else:
            self.panning = True
            self.dragging_marker = False
            self.pan_start_x = event.x
            self.pan_start_y = event.y

    def on_mouse_drag(self, event):
        """Handle mouse drag."""
        if self.dragging_marker and self.selected_marker is not None:
            frame_x, frame_y = self.window_to_frame_coords(event.x, event.y)

            # Calculate delta from initial position
            if self.drag_start_pos is not None:
                dx = frame_x - self.drag_start_pos[0]
                dy = frame_y - self.drag_start_pos[1]

                # Move the selected marker
                self.points[self.current_frame, self.selected_marker] = [frame_x, frame_y]
                self.update_marker_delta(self.selected_marker)

                # If chain dragging with Shift, also move downstream markers
                if self.dragging_chain and self.selected_marker in MARKER_CHAIN:
                    for downstream_idx in MARKER_CHAIN[self.selected_marker]:
                        current_pos = self.points[self.current_frame, downstream_idx]
                        if not np.isnan(current_pos).any():
                            # Apply the same delta to downstream marker
                            new_pos = [current_pos[0] + dx, current_pos[1] + dy]
                            self.points[self.current_frame, downstream_idx] = new_pos
                            self.update_marker_delta(downstream_idx)

                    # Update drag start position for next iteration
                    self.drag_start_pos = [frame_x, frame_y]

                self.mark_modified()
                self.update_display()
        elif self.panning:  # Pan at all zoom levels
            # Calculate pan delta in frame coordinates
            dx_window = event.x - self.pan_start_x
            dy_window = event.y - self.pan_start_y

            # Convert window pixel movement to frame coordinate movement
            viewport_width = self.width / self.zoom
            viewport_height = self.height / self.zoom
            dx_frame = (dx_window / self.display_width) * viewport_width
            dy_frame = (dy_window / self.display_height) * viewport_height

            # Update pan (in frame coordinates)
            self.pan_x += dx_frame * self.zoom
            self.pan_y += dy_frame * self.zoom

            # Only clamp pan when zoomed in
            if self.zoom > 1.0:
                self.clamp_pan()

            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.update_display()

    def on_mouse_up(self, event):
        """Handle mouse button release."""
        self.dragging_marker = False
        self.dragging_chain = False
        self.drag_start_pos = None
        self.panning = False

    def on_right_click(self, event):
        """Handle right click - delete marker."""
        frame_x, frame_y = self.window_to_frame_coords(event.x, event.y)
        marker = self.find_nearest_marker(frame_x, frame_y, max_dist=15)
        if marker is not None:
            self.points[self.current_frame, marker] = [np.nan, np.nan]
            self.mark_modified()
            self.update_display()

    def on_middle_click(self, event):
        """Handle middle click - restore marker."""
        frame_x, frame_y = self.window_to_frame_coords(event.x, event.y)
        marker = self.find_nearest_marker(frame_x, frame_y, max_dist=15, use_original=True)
        if marker is not None:
            self.points[self.current_frame, marker] = \
                self.original_points[self.current_frame, marker].copy()

            # Clear delta for this marker in current frame
            if self.current_frame in self.frame_deltas and marker in self.frame_deltas[self.current_frame]:
                del self.frame_deltas[self.current_frame][marker]
                # If no more deltas in this frame, remove the frame entry
                if not self.frame_deltas[self.current_frame]:
                    del self.frame_deltas[self.current_frame]
                print(f"Cleared delta for marker {marker} ({LANDMARK_LABELS[marker]}) in frame {self.current_frame + 1}")
                # Recalculate to update downstream frames
                self.recalculate_all_frames()

            self.mark_modified()
            self.update_display()

    def on_mouse_wheel(self, event):
        """Handle mouse wheel - zoom."""
        # Cross-platform mouse wheel handling
        if event.num == 4 or event.delta > 0:  # Scroll up (Linux Button-4 or Windows positive delta)
            self.zoom = min(10.0, self.zoom * 1.5)
            print(f"Zoom in: {self.zoom:.1f}x")
        elif event.num == 5 or event.delta < 0:  # Scroll down (Linux Button-5 or Windows negative delta)
            self.zoom = max(0.1, self.zoom / 1.5)  # Allow zooming out much further
            print(f"Zoom out: {self.zoom:.1f}x")
        self.update_display()

    def find_nearest_marker(self, x, y, max_dist=15, use_original=False):
        """Find the nearest marker to the given position."""
        points = self.original_points if use_original else self.points
        frame_points = points[self.current_frame]

        min_dist = max_dist
        nearest = None

        for i, pt in enumerate(frame_points):
            if not np.isnan(pt).any():
                dist = np.sqrt((pt[0] - x)**2 + (pt[1] - y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = i

        return nearest

    def mark_modified(self):
        """Mark current frame as modified."""
        self.modified = True
        self.modified_frames.add(self.current_frame)

    def update_marker_delta(self, marker_idx):
        """Update the delta for a marker in current frame and propagate forward."""
        current_pos = self.points[self.current_frame, marker_idx]
        original_pos = self.original_points[self.current_frame, marker_idx]

        # Only calculate delta if both positions are valid (not NaN)
        if not (np.isnan(current_pos).any() or np.isnan(original_pos).any()):
            dx = current_pos[0] - original_pos[0]
            dy = current_pos[1] - original_pos[1]

            # Store delta for this frame
            if self.current_frame not in self.frame_deltas:
                self.frame_deltas[self.current_frame] = {}
            self.frame_deltas[self.current_frame][marker_idx] = (dx, dy)

            print(f"Delta for marker {marker_idx} ({LANDMARK_LABELS[marker_idx]}): ({dx:.1f}, {dy:.1f})")

            # Propagate to all subsequent frames
            self.propagate_deltas_forward(self.current_frame)

    def propagate_deltas_forward(self, start_frame):
        """Propagate deltas from start_frame to all subsequent frames until a barrier."""
        # Find the next barrier
        next_barrier = None
        for barrier in sorted(self.delta_barriers):
            if barrier > start_frame:
                next_barrier = barrier
                break

        end_frame = next_barrier if next_barrier else self.n_frames

        # For each frame from start+1 to end, recalculate with cumulative deltas
        for frame_idx in range(start_frame + 1, end_frame):
            self.apply_cumulative_deltas_to_frame(frame_idx)

    def apply_cumulative_deltas_to_frame(self, frame_idx):
        """Apply cumulative deltas from all previous frames to the specified frame."""
        # Find the last barrier before this frame
        last_barrier = -1
        for barrier in sorted(self.delta_barriers):
            if barrier < frame_idx:
                last_barrier = barrier
            else:
                break

        # Accumulate deltas from all frames between last_barrier and frame_idx
        cumulative_deltas = {}
        for source_frame in range(last_barrier + 1, frame_idx):
            if source_frame in self.frame_deltas:
                for marker_idx, (dx, dy) in self.frame_deltas[source_frame].items():
                    if marker_idx not in cumulative_deltas:
                        cumulative_deltas[marker_idx] = [0, 0]
                    cumulative_deltas[marker_idx][0] += dx
                    cumulative_deltas[marker_idx][1] += dy

        # Apply cumulative deltas to this frame (if not a manually edited frame)
        if cumulative_deltas and frame_idx not in self.frame_deltas:
            for marker_idx, (dx, dy) in cumulative_deltas.items():
                original_pos = self.original_points[frame_idx, marker_idx]
                if not np.isnan(original_pos).any():
                    self.points[frame_idx, marker_idx] = [
                        original_pos[0] + dx,
                        original_pos[1] + dy
                    ]
            self.modified_frames.add(frame_idx)

    def clear_deltas(self):
        """Mark current frame as a delta barrier to stop propagation."""
        self.delta_barriers.add(self.current_frame)
        print(f"Added delta barrier at frame {self.current_frame + 1}")
        print(f"Deltas from previous frames will not propagate past this point")

        # Recalculate all frames to respect the new barrier
        self.recalculate_all_frames()

    def recalculate_all_frames(self):
        """Recalculate all frames from original positions with current deltas and barriers."""
        print("Recalculating all frames...")

        # Reset frames that don't have explicit deltas
        for frame_idx in range(self.n_frames):
            if frame_idx not in self.frame_deltas:
                self.points[frame_idx] = self.original_points[frame_idx].copy()

        # Clear modified frames that aren't in frame_deltas
        self.modified_frames = set(self.frame_deltas.keys())

        # Reapply deltas with barriers
        for frame_idx in sorted(self.frame_deltas.keys()):
            # Propagate this frame's deltas forward
            self.propagate_deltas_forward(frame_idx)

        self.modified = len(self.modified_frames) > 0
        print(f"Recalculation complete")

    def draw_skeleton(self, frame, points):
        """Draw skeleton connections."""
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            start = points[start_idx]
            end = points[end_idx]

            if not (np.isnan(start).any() or np.isnan(end).any()):
                pt1 = (int(start[0]), int(start[1]))
                pt2 = (int(end[0]), int(end[1]))
                cv2.line(frame, pt1, pt2, (100, 100, 100), 2)

    def draw_landmark(self, frame, x, y, label, color, highlight=False):
        """Draw a landmark with text label."""
        x, y = int(x), int(y)

        # Draw circle background
        radius = 12 if highlight else 8
        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 2 if highlight else 1)

        # Draw text label
        font_scale = 0.4
        thickness = 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2

        # Draw black outline for text
        cv2.putText(frame, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
        # Draw white text
        cv2.putText(frame, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    def draw_ghost_landmark(self, frame, x, y, color):
        """Draw a translucent landmark showing original position."""
        x, y = int(x), int(y)

        # Create overlay for transparency
        overlay = frame.copy()

        # Draw semi-transparent circle
        radius = 6
        cv2.circle(overlay, (x, y), radius, color, -1)
        cv2.circle(overlay, (x, y), radius, (255, 255, 255), 1)

        # Draw crosshair to indicate original position
        cv2.line(overlay, (x - radius - 2, y), (x + radius + 2, y), (200, 200, 200), 1)
        cv2.line(overlay, (x, y - radius - 2), (x, y + radius + 2), (200, 200, 200), 1)

        # Blend with transparency (30% opaque)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    def draw_legend(self, frame):
        """Draw legend showing marker abbreviations."""
        h, w = frame.shape[:2]

        # Legend content - all body parts
        legend_items = [
            ("Markers:", None),
            ("N=Nose", LANDMARK_GROUPS['face']['color']),
            ("EI/E/EO=Eyes", LANDMARK_GROUPS['face']['color']),
            ("ER=Ear", LANDMARK_GROUPS['face']['color']),
            ("M=Mouth", LANDMARK_GROUPS['face']['color']),
            ("", None),
            ("S=Shoulder", LANDMARK_GROUPS['left_arm']['color']),
            ("E=Elbow", LANDMARK_GROUPS['left_arm']['color']),
            ("W=Wrist", LANDMARK_GROUPS['left_arm']['color']),
            ("P=Pinky", LANDMARK_GROUPS['left_arm']['color']),
            ("I=Index", LANDMARK_GROUPS['left_arm']['color']),
            ("T=Thumb", LANDMARK_GROUPS['left_arm']['color']),
            ("", None),
            ("H=Hip", LANDMARK_GROUPS['left_leg']['color']),
            ("K=Knee", LANDMARK_GROUPS['left_leg']['color']),
            ("A=Ankle", LANDMARK_GROUPS['left_leg']['color']),
            ("L=Heel", LANDMARK_GROUPS['left_leg']['color']),
            ("F=Foot", LANDMARK_GROUPS['left_leg']['color']),
            ("", None),
            ("L=Left side", (0, 255, 0)),
            ("R=Right side", (0, 0, 255)),
        ]

        # Draw semi-transparent background
        legend_width = 180
        legend_height = len(legend_items) * 22 + 15
        legend_x = w - legend_width - 10
        legend_y = 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (legend_x, legend_y),
                     (legend_x + legend_width, legend_y + legend_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (legend_x, legend_y),
                     (legend_x + legend_width, legend_y + legend_height),
                     (255, 255, 255), 1)

        # Draw legend items
        y = legend_y + 22
        for text, color in legend_items:
            if text:
                if color:
                    cv2.circle(frame, (legend_x + 15, y - 5), 5, color, -1)
                    cv2.putText(frame, text, (legend_x + 30, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, text, (legend_x + 10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 22

    def draw_hud(self, frame):
        """Draw HUD with frame info and controls."""
        h, w = frame.shape[:2]

        # Draw semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Frame info
        status = "PLAYING" if self.playing else "PAUSED"
        modified_marker = " *" if self.modified else ""
        cv2.putText(frame, f"Frame: {self.current_frame + 1}/{self.total_frames}  |  {status}{modified_marker}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Modified frames count
        if self.modified:
            cv2.putText(frame, f"Modified frames: {len(self.modified_frames)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Active deltas and barriers indicator
        if self.frame_deltas or self.delta_barriers:
            delta_frame_count = len(self.frame_deltas)
            barrier_count = len(self.delta_barriers)
            delta_text = f"Deltas: {delta_frame_count} frames | Barriers: {barrier_count} (Press C to add)"
            cv2.putText(frame, delta_text,
                       (w - 620, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Selected pair info
        if self.selected_pair is not None:
            left_idx, right_idx = SYMMETRIC_PAIRS[self.selected_pair]
            # Get the key for this pair
            pair_key = [k for k, v in PAIR_KEYS.items() if v == self.selected_pair]
            key_str = pair_key[0].upper() if pair_key else '?'
            pair_text = f"Selected [{key_str}]: L-{LANDMARK_NAMES[left_idx]} <-> R-{LANDMARK_NAMES[right_idx]} (Press ENTER to flip)"
            cv2.putText(frame, pair_text,
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw legend
        self.draw_legend(frame)

    def render_frame(self):
        """Render current frame with tracking overlay using viewport system."""
        # Read frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()

        if not ret:
            return None

        points = self.points[self.current_frame]

        # Calculate viewport bounds in frame space (can extend beyond frame when zoomed out)
        viewport_width = self.width / self.zoom
        viewport_height = self.height / self.zoom
        center_x = self.width / 2 - self.pan_x / self.zoom
        center_y = self.height / 2 - self.pan_y / self.zoom
        view_left = int(center_x - viewport_width / 2)
        view_top = int(center_y - viewport_height / 2)
        view_right = int(center_x + viewport_width / 2)
        view_bottom = int(center_y + viewport_height / 2)

        # Create display frame with dark gray background for areas outside the frame
        display_frame = np.full((self.display_height, self.display_width, 3), 40, dtype=np.uint8)

        # Calculate the portion of the frame that's visible in viewport
        frame_in_view_left = max(0, view_left)
        frame_in_view_top = max(0, view_top)
        frame_in_view_right = min(self.width, view_right)
        frame_in_view_bottom = min(self.height, view_bottom)

        # If any part of the frame is visible
        if frame_in_view_left < frame_in_view_right and frame_in_view_top < frame_in_view_bottom:
            # Extract the visible portion of the frame
            frame_slice = frame[frame_in_view_top:frame_in_view_bottom,
                              frame_in_view_left:frame_in_view_right].copy()

            # Calculate where this slice should be placed in the display
            display_left = int((frame_in_view_left - view_left) / viewport_width * self.display_width)
            display_top = int((frame_in_view_top - view_top) / viewport_height * self.display_height)
            display_right = int((frame_in_view_right - view_left) / viewport_width * self.display_width)
            display_bottom = int((frame_in_view_bottom - view_top) / viewport_height * self.display_height)

            # Resize the frame slice and place it
            if display_right > display_left and display_bottom > display_top:
                resized_slice = cv2.resize(frame_slice,
                                          (display_right - display_left, display_bottom - display_top),
                                          interpolation=cv2.INTER_LINEAR)
                display_frame[display_top:display_bottom, display_left:display_right] = resized_slice

        # Draw frame boundary when zoomed out to show actual video frame edges
        if view_left < 0 or view_top < 0 or view_right > self.width or view_bottom > self.height:
            # Calculate frame corners in window coordinates
            tl_x, tl_y = self.frame_to_window_coords(0, 0)
            br_x, br_y = self.frame_to_window_coords(self.width, self.height)
            cv2.rectangle(display_frame, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)),
                         (0, 255, 255), 3)  # Cyan border to mark frame boundaries

        # Draw skeleton connections in window space
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            start = points[start_idx]
            end = points[end_idx]

            if not (np.isnan(start).any() or np.isnan(end).any()):
                wx1, wy1 = self.frame_to_window_coords(start[0], start[1])
                wx2, wy2 = self.frame_to_window_coords(end[0], end[1])
                cv2.line(display_frame, (int(wx1), int(wy1)), (int(wx2), int(wy2)), (100, 100, 100), 2)

        # Draw original positions as translucent markers (show what was changed)
        original_points = self.original_points[self.current_frame]
        for idx, orig_pt in enumerate(original_points):
            if not np.isnan(orig_pt).any():
                curr_pt = points[idx]
                # Only draw original if position has changed
                if not np.isnan(curr_pt).any():
                    dist = np.linalg.norm(curr_pt - orig_pt)
                    if dist > 1.0:  # Threshold to avoid drawing for tiny changes
                        orig_wx, orig_wy = self.frame_to_window_coords(orig_pt[0], orig_pt[1])

                        # Determine color based on body part group
                        if idx in LANDMARK_GROUPS['face']['indices']:
                            color = LANDMARK_GROUPS['face']['color']
                        elif idx in LANDMARK_GROUPS['left_arm']['indices']:
                            color = LANDMARK_GROUPS['left_arm']['color']
                        elif idx in LANDMARK_GROUPS['right_arm']['indices']:
                            color = LANDMARK_GROUPS['right_arm']['color']
                        elif idx in LANDMARK_GROUPS['left_leg']['indices']:
                            color = LANDMARK_GROUPS['left_leg']['color']
                        elif idx in LANDMARK_GROUPS['right_leg']['indices']:
                            color = LANDMARK_GROUPS['right_leg']['color']
                        else:
                            color = (200, 200, 200)  # Default gray

                        # Draw translucent original marker
                        self.draw_ghost_landmark(display_frame, orig_wx, orig_wy, color)

        # Draw landmarks in window space (all markers, even off-screen)
        for idx, pt in enumerate(points):
            if not np.isnan(pt).any():
                wx, wy = self.frame_to_window_coords(pt[0], pt[1])

                # Determine color based on body part group
                if idx in LANDMARK_GROUPS['face']['indices']:
                    color = LANDMARK_GROUPS['face']['color']
                elif idx in LANDMARK_GROUPS['left_arm']['indices']:
                    color = LANDMARK_GROUPS['left_arm']['color']
                elif idx in LANDMARK_GROUPS['right_arm']['indices']:
                    color = LANDMARK_GROUPS['right_arm']['color']
                elif idx in LANDMARK_GROUPS['left_leg']['indices']:
                    color = LANDMARK_GROUPS['left_leg']['color']
                elif idx in LANDMARK_GROUPS['right_leg']['indices']:
                    color = LANDMARK_GROUPS['right_leg']['color']
                else:
                    color = (200, 200, 200)  # Default gray

                highlight = (idx == self.selected_marker)
                label = LANDMARK_LABELS[idx] if idx < len(LANDMARK_LABELS) else str(idx)

                # Draw landmark at actual position (even if off-screen when zoomed in)
                # Clipping will be handled by OpenCV
                self.draw_landmark(display_frame, wx, wy, label, color, highlight=highlight)

        # Highlight selected pair
        if self.selected_pair is not None:
            left_idx, right_idx = SYMMETRIC_PAIRS[self.selected_pair]
            for idx in [left_idx, right_idx]:
                pt = points[idx]
                if not np.isnan(pt).any():
                    wx, wy = self.frame_to_window_coords(pt[0], pt[1])
                    if 0 <= wx < self.display_width and 0 <= wy < self.display_height:
                        cv2.circle(display_frame, (int(wx), int(wy)), 12, (255, 255, 0), 2)

        # Draw HUD
        self.draw_hud(display_frame)

        return display_frame

    def next_frame(self):
        """Go to next frame."""
        self.current_frame = min(self.current_frame + 1, self.n_frames - 1)

    def prev_frame(self):
        """Go to previous frame."""
        self.current_frame = max(self.current_frame - 1, 0)

    def toggle_play(self):
        """Toggle play/pause."""
        self.playing = not self.playing

    def zoom_in(self):
        """Zoom in."""
        self.zoom *= 1.5
        print(f"Zoom: {self.zoom:.1f}x")

    def zoom_out(self):
        """Zoom out."""
        self.zoom = max(0.1, self.zoom / 1.5)  # Allow zooming out much further
        print(f"Zoom: {self.zoom:.1f}x")

    def reset_zoom(self):
        """Reset zoom and pan."""
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        print("View reset")

    def clamp_pan(self):
        """Clamp pan values to keep viewport within frame bounds."""
        if self.zoom > 1.0:
            viewport_width = self.width / self.zoom
            viewport_height = self.height / self.zoom

            # Calculate maximum pan that keeps viewport within frame
            max_pan_x = (self.width - viewport_width) / 2 * self.zoom
            max_pan_y = (self.height - viewport_height) / 2 * self.zoom

            # Clamp pan values
            self.pan_x = max(-max_pan_x, min(max_pan_x, self.pan_x))
            self.pan_y = max(-max_pan_y, min(max_pan_y, self.pan_y))

    def pan_view(self, dx, dy):
        """Pan the view at any zoom level."""
        self.pan_x += dx
        self.pan_y += dy
        # Only clamp when zoomed in
        if self.zoom > 1.0:
            self.clamp_pan()

    def reset_frame(self):
        """Reset current frame to original."""
        self.points[self.current_frame] = self.original_points[self.current_frame].copy()
        if self.current_frame in self.modified_frames:
            self.modified_frames.remove(self.current_frame)

        # Clear any deltas for this frame
        if self.current_frame in self.frame_deltas:
            del self.frame_deltas[self.current_frame]
            print(f"Reset frame {self.current_frame + 1} (cleared deltas)")
            # Recalculate to update downstream frames
            self.recalculate_all_frames()
        else:
            print(f"Reset frame {self.current_frame + 1}")

        self.modified = len(self.modified_frames) > 0

    def flip_pair(self):
        """Flip selected symmetric pair and all dependent pairs for current frame."""
        if self.selected_pair is None:
            print("No pair selected. Press S/E/W/H/K/A/P/I/T to select a pair.")
            return

        # Collect all pairs to flip (selected + dependencies)
        pairs_to_flip = [self.selected_pair]
        if self.selected_pair in PAIR_DEPENDENCIES:
            pairs_to_flip.extend(PAIR_DEPENDENCIES[self.selected_pair])

        # Get the key for the main pair
        pair_key = [k for k, v in PAIR_KEYS.items() if v == self.selected_pair]
        key_str = pair_key[0].upper() if pair_key else '?'
        print(f"Flipping [{key_str}] in frame {self.current_frame + 1} (with {len(pairs_to_flip)} pair(s)):")

        # Flip each pair
        for pair_idx in pairs_to_flip:
            left_idx, right_idx = SYMMETRIC_PAIRS[pair_idx]

            # Get positions before swap
            left_before = self.points[self.current_frame, left_idx].copy()
            right_before = self.points[self.current_frame, right_idx].copy()

            # Check if either position is NaN
            if np.isnan(left_before).any() or np.isnan(right_before).any():
                print(f"  Skipped {LANDMARK_NAMES[left_idx]}/{LANDMARK_NAMES[right_idx]} - NaN position")
                continue

            # Swap the points
            self.points[self.current_frame, left_idx] = right_before
            self.points[self.current_frame, right_idx] = left_before

            # Update deltas for both markers
            self.update_marker_delta(left_idx)
            self.update_marker_delta(right_idx)

            # Log the swap
            print(f"  {LANDMARK_NAMES[left_idx]}: ({left_before[0]:.1f}, {left_before[1]:.1f}) -> ({right_before[0]:.1f}, {right_before[1]:.1f})")
            print(f"  {LANDMARK_NAMES[right_idx]}: ({right_before[0]:.1f}, {right_before[1]:.1f}) -> ({left_before[0]:.1f}, {left_before[1]:.1f})")

        self.mark_modified()

    def save_changes(self):
        """Save changes to HDF5 file."""
        if not self.modified:
            print("No changes to save.")
            return False

        print(f"\nSaving changes...")
        print(f"Modified frames: {len(self.modified_frames)}")

        # Create backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.h5_path.parent / f"{self.h5_path.stem}.backup_{timestamp}.h5"
        print(f"Backing up to: {backup_path.name}")
        shutil.copy2(self.h5_path, backup_path)

        # Reshape points back to tracks format
        # (n_frames, n_nodes, 2) -> (1, 2, n_nodes, n_frames)
        corrected_tracks = self.points.transpose(2, 1, 0)[np.newaxis, :, :, :]

        # Read other datasets and attributes
        with h5py.File(self.h5_path, 'r') as f:
            other_datasets = {}
            for key in f.keys():
                if key != 'tracks':
                    other_datasets[key] = f[key][:]
            attrs = dict(f.attrs)

        # Write updated file
        with h5py.File(self.h5_path, 'w') as f:
            f.create_dataset('tracks', data=corrected_tracks.astype('<f8'))

            for key, data in other_datasets.items():
                f.create_dataset(key, data=data)

            for key, value in attrs.items():
                f.attrs[key] = value

        print(f"✓ Changes saved to: {self.h5_path.name}")

        # Update original to reflect saved state
        self.original_points = self.points.copy()
        self.modified = False
        self.modified_frames.clear()

        return True

    def on_key_press(self, event):
        """Handle keyboard events."""
        key = event.keysym

        # Navigation
        if key == 'Left':
            self.prev_frame()
            self.update_display()
        elif key == 'Right':
            self.next_frame()
            self.update_display()
        elif key == 'Up':
            self.pan_view(0, 50)
            self.update_display()
        elif key == 'Down':
            self.pan_view(0, -50)
            self.update_display()
        elif key == 'space':
            self.toggle_play()

        # View controls
        elif key in ['plus', 'equal']:
            self.zoom_in()
            self.update_display()
        elif key == 'minus':
            self.zoom_out()
            self.update_display()
        elif key == '0':
            self.reset_zoom()
            self.update_display()
        elif key == 'bracketleft':
            self.pan_view(50, 0)
            self.update_display()
        elif key == 'bracketright':
            self.pan_view(-50, 0)
            self.update_display()

        # Editing
        elif key.lower() == 'r':
            self.reset_frame()
            self.update_display()
        elif key.lower() == 'c':
            self.clear_deltas()
            self.update_display()
        elif key == 'Return':
            self.flip_pair()
            self.update_display()

        # Pair selection
        elif key.lower() in PAIR_KEYS:
            self.selected_pair = PAIR_KEYS[key.lower()]
            left_idx, right_idx = SYMMETRIC_PAIRS[self.selected_pair]
            print(f"Selected [{key.upper()}]: L-{LANDMARK_NAMES[left_idx]} <-> R-{LANDMARK_NAMES[right_idx]}")
            self.update_display()

        # Save (Ctrl+S) and Quit
        elif event.state & 0x4 and key.lower() == 's':  # Ctrl+S
            self.save_changes()
            self.root.quit()
        elif key in ['q', 'Escape']:
            self.on_close()

    def update_display(self):
        """Update the canvas with the current frame."""
        frame = self.render_frame()
        if frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(image=pil_image)
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

    def on_close(self):
        """Handle window close."""
        if self.modified:
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Save before closing?"
            )
            if response is True:  # Yes
                self.save_changes()
                self.root.destroy()
            elif response is False:  # No
                print("Quitting without saving.")
                self.root.destroy()
            # Cancel does nothing
        else:
            self.root.destroy()

    def play_loop(self):
        """Timer loop for playback."""
        if self.playing:
            self.next_frame()
            if self.current_frame >= self.n_frames - 1:
                self.playing = False
            self.update_display()

        # Schedule next update
        delay = int(1000 / self.fps)
        self.root.after(delay, self.play_loop)

    def run(self):
        """Run the interactive editor."""
        # Initial display
        self.update_display()

        # Start play loop timer
        self.play_loop()

        # Run tkinter main loop
        self.root.mainloop()

        # Cleanup
        self.cap.release()


def find_camera_files(camera_dir):
    """
    Find video and analysis files in a camera directory.

    Args:
        camera_dir: Path to camera directory

    Returns:
        Tuple of (video_path, analysis_path)
    """
    camera_dir = Path(camera_dir)

    if not camera_dir.is_dir():
        raise ValueError(f"Not a directory: {camera_dir}")

    # Find video file
    video_files = list(camera_dir.glob('*-view-recording.mp4'))
    if not video_files:
        raise ValueError(f"No video file (*-view-recording.mp4) found in {camera_dir}")
    if len(video_files) > 1:
        raise ValueError(f"Multiple video files found in {camera_dir}: {[f.name for f in video_files]}")

    # Find analysis file
    analysis_files = list(camera_dir.glob('*_analysis.h5'))
    if not analysis_files:
        raise ValueError(f"No analysis file (*_analysis.h5) found in {camera_dir}")
    if len(analysis_files) > 1:
        raise ValueError(f"Multiple analysis files found in {camera_dir}: {[f.name for f in analysis_files]}")

    return video_files[0], analysis_files[0]


def main():
    parser = argparse.ArgumentParser(
        description='Interactive tracking editor for manual marker correction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('camera_dir',
                       help='Path to camera directory')

    args = parser.parse_args()

    try:
        # Find video and analysis files
        video_path, analysis_path = find_camera_files(args.camera_dir)
        print(f"Found video: {video_path.name}")
        print(f"Found analysis: {analysis_path.name}")
        print()

        # Run editor
        editor = TrackingEditor(video_path, analysis_path)
        editor.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
