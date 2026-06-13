import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Finger indices
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

# Gesture logic
def is_folded(tip, pip, lm): return lm[tip].y > lm[pip].y + 0.02
def is_straight(tip, pip, lm): return lm[tip].y < lm[pip].y - 0.02
def is_half_folded(tip, pip, lm): return -0.02 < (lm[tip].y - lm[pip].y) < 0.02

def vector_2d(p1, p2): return np.array([p2.x - p1.x, p2.y - p1.y])
def unit_vector(vector): return vector / np.linalg.norm(vector) if np.linalg.norm(vector) != 0 else vector
def angle_between(v1, v2): return np.arccos(np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0)) * 180 / np.pi

def detect_gesture(lm):
    if lm is None or len(lm) < 21:
        return "no_hand"

    thumb_closed = is_folded(THUMB_TIP, THUMB_IP, lm)
    index_open = is_straight(INDEX_TIP, INDEX_PIP, lm)
    middle_open = is_straight(MIDDLE_TIP, MIDDLE_PIP, lm)
    ring_folded = is_folded(RING_TIP, RING_PIP, lm)
    pinky_folded = is_folded(PINKY_TIP, PINKY_PIP, lm)

    all_closed = thumb_closed and is_folded(INDEX_TIP, INDEX_PIP, lm) and \
                 is_folded(MIDDLE_TIP, MIDDLE_PIP, lm) and ring_folded and pinky_folded
    all_open = not thumb_closed and index_open and middle_open and not ring_folded and not pinky_folded

    # Distance-based gestures
    hand_width = np.linalg.norm([lm[5].x - lm[17].x, lm[5].y - lm[17].y])
    thumb_index_dist = np.linalg.norm([lm[4].x - lm[8].x, lm[4].y - lm[8].y])
    norm_thumb_index = thumb_index_dist / hand_width if hand_width else 0

    if index_open and middle_open and not thumb_closed and ring_folded and pinky_folded:
        angle = angle_between(vector_2d(lm[5], lm[8]), vector_2d(lm[9], lm[12]))
        if 10 < angle < 70:
            return "peace"
    if all_closed:
        return "fist"
    elif all_open:
        return "open_hand"
    elif thumb_closed and index_open and middle_open and not ring_folded and not pinky_folded:
        return "scroll_up"
    elif thumb_closed and is_half_folded(MIDDLE_TIP, MIDDLE_PIP, lm) and \
         is_half_folded(RING_TIP, RING_PIP, lm) and is_half_folded(PINKY_TIP, PINKY_PIP, lm):
        return "scroll_down"
    elif thumb_closed and is_folded(INDEX_TIP, INDEX_PIP, lm) and is_folded(MIDDLE_TIP, MIDDLE_PIP, lm) and ring_folded and pinky_folded:
        if norm_thumb_index > 1.4:
            return "zoom_out"
        elif 0.4 < norm_thumb_index <= 1.4:
            return "zoom_in"
    elif index_open and thumb_closed and is_folded(MIDDLE_TIP, MIDDLE_PIP, lm) and ring_folded and pinky_folded:
        return "move_up"
    elif is_half_folded(INDEX_TIP, INDEX_PIP, lm) and thumb_closed and is_folded(MIDDLE_TIP, MIDDLE_PIP, lm) and ring_folded and pinky_folded:
        return "move_down"
    elif index_open and middle_open and not thumb_closed and not ring_folded and not pinky_folded:
        return "palm"

    return "no_hand"

# Gesture tracking variables
prev_thumb_index_dist = None
zoom_mode_active = False
zoom_gesture_start_time = 0
last_action_time = 0

# Constants
ZOOM_TIMEOUT = 2  # seconds to deactivate zoom mode
THRESHOLD = 0.025  # min distance change to detect increase/decrease
DEBOUNCE = 0.001    # seconds between actions

print("? Gesture detection started (ESC to quit)...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("? Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        gesture = "no_hand"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks.landmark)

                # Thumb-index distance calculation
                thumb_tip = hand_landmarks.landmark[THUMB_TIP]
                index_tip = hand_landmarks.landmark[INDEX_TIP]
                current_dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

                # Fingers folded check for zoom mode
                middle_folded = is_folded(MIDDLE_TIP, MIDDLE_PIP, hand_landmarks.landmark)
                ring_folded = is_folded(RING_TIP, RING_PIP, hand_landmarks.landmark)
                pinky_folded = is_folded(PINKY_TIP, PINKY_PIP, hand_landmarks.landmark)

                if middle_folded and ring_folded and pinky_folded:
                    if not zoom_mode_active:
                        zoom_mode_active = True
                        zoom_gesture_start_time = time.time()
                        prev_thumb_index_dist = current_dist
                    else:
                        zoom_gesture_start_time = time.time()
                        if prev_thumb_index_dist is not None and time.time() - last_action_time > DEBOUNCE:
                            if current_dist - prev_thumb_index_dist > THRESHOLD:
                                print("?? increase")
                                last_action_time = time.time()
                            elif prev_thumb_index_dist - current_dist > THRESHOLD:
                                print("?? decrease")
                                last_action_time = time.time()
                        prev_thumb_index_dist = current_dist
                else:
                    if zoom_mode_active and time.time() - zoom_gesture_start_time > ZOOM_TIMEOUT:
                        zoom_mode_active = False
                        prev_thumb_index_dist = None

        if gesture != "no_hand":
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Detected Gesture: {gesture}")

        cv2.imshow("Gesture Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

except KeyboardInterrupt:
    print("? Stopped by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
