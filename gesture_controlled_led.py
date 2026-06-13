import cv2
import mediapipe as mp
import numpy as np
import time
import os
import firebase_admin
from firebase_admin import credentials, firestore 
import constants
import firebase_setup
from enum import Enum
import math
from gpiozero import PWMLED

ledBrightness =0.1
ledStatus="off"
ledLastControlTime = None
#Predefined Gestures
class GESTURES(Enum):
    palm = "palm"
    all_fingers_fold = "all_fingers_fold"
    fist = "fist"
    thumbs_up="thumbs_up"
    thumbs_down = "thumbs_down"
    point_index = "point_index"
    peace_sign = "peace_sign"
    
    
class LEDCONTROL(Enum):
    LED_OFF = "led_off"
    LED_ON = "led_on"
    INCREASE_BRIGHTNESS = "increase_brightness"
    REDUCE_BRIGHTNESS="reduce_brightness"
    
gesture_mapping =  {
    LEDCONTROL.LED_ON: GESTURES.palm,
    LEDCONTROL.LED_OFF: GESTURES.all_fingers_fold,
    LEDCONTROL.INCREASE_BRIGHTNESS: GESTURES.thumbs_up,
    LEDCONTROL.REDUCE_BRIGHTNESS: GESTURES.thumbs_down
}
    

#initialize Database
db = firestore.client()
collection = firebase_setup.db.collection(constants.COLLECTION_NAME)  
project_ref = collection.document(constants.DOCUMENT_PROJECT)
project_ref.update({'led_status': "off"})

#define LEDCONTROL
ledPin = 17  # define ledPin
led = PWMLED(ledPin)

# Suppress verbose TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Initialize OpenCV VideoCapture for USB webcam (adjust index if needed)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Set a lower resolution for potentially better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7,  # Increased confidence for more stable recognition
                        min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Finger landmark indices
THUMB_TIP = 4
THUMB_IP=3
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

THUMB_CMC = 2
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

WRIST = 0


def is_folded(tip, pip, lm): return lm[tip].y > lm[pip].y + 0.02
def is_straight(tip, pip, lm): return lm[tip].y < lm[pip].y - 0.02
def is_half_folded(tip, pip, lm): return -0.02 < (lm[tip].y - lm[pip].y) < 0.02

def vector_2d(p1, p2): return np.array([p2.x - p1.x, p2.y - p1.y])
def unit_vector(vector): return vector / np.linalg.norm(vector) if np.linalg.norm(vector) != 0 else vector
def angle_between(v1, v2): return np.arccos(np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0)) * 180 / np.pi

def attachEventListener():
    """ Attaches a Firestore snapshot listener to the  document. """
    project_ref.on_snapshot(on_gesture_change)
    
def on_gesture_change(doc_snapshot, changes, read_time):
    """ Firestore snapshot listener for threshold changes. """
    global LEDCONTROL,GESTURES,gesture_mapping
    for doc in doc_snapshot:
        print(f'Received document snapshot: {doc.to_dict()}')
        gesture_mapping[LEDCONTROL.LED_ON] = GESTURES[doc.to_dict().get('led_on_gesture')] 
        gesture_mapping[LEDCONTROL.LED_OFF] = GESTURES[doc.to_dict().get('led_off_gesture')]
       
        print(f'Updated gestures: {gesture_mapping}')  

def updateLedStatus(gesture):
    
    global ledBrightness, ledStatus, ledLastControlTime
    ledLastControlTime=time.time()
    gesture_enum = GESTURES[gesture]
    led_control = next((key for key, value in gesture_mapping.items() if value == gesture_enum), None)
    if led_control == LEDCONTROL.LED_ON and ledStatus != "on":
        led.value = ledBrightness
        ledStatus = "on"
        project_ref.update({'led_status': ledStatus})
        project_ref.update({'led_brightness': f'{int(ledBrightness * 100)}%'})

    elif led_control == LEDCONTROL.LED_OFF and ledStatus != "off":
        led.value = 0
        ledStatus = "off"
        project_ref.update({'led_status': ledStatus, 'led_brightness': '0%'})

    elif led_control == LEDCONTROL.INCREASE_BRIGHTNESS and ledStatus == "on":
        if ledBrightness < 1.0:
            ledBrightness = round(min(1.0, ledBrightness + 0.2), 1)
            led.value = ledBrightness
            project_ref.update({'led_brightness': f'{int(ledBrightness * 100)}%'})

    elif led_control == LEDCONTROL.REDUCE_BRIGHTNESS and ledStatus == "on":
        if ledBrightness > 0.2:
            ledBrightness = round(max(0.2, ledBrightness - 0.2), 1)
            led.value = ledBrightness
            project_ref.update({'led_brightness': f'{int(ledBrightness * 100)}%'})
   


def get_angle(a, b, c):
    """Calculates the angle at point b formed by points a, b, and c."""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

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
            return "peace_sign"
    if not thumb_closed and is_folded(INDEX_TIP, INDEX_PIP, lm) and is_folded(MIDDLE_TIP, MIDDLE_PIP, lm) and ring_folded and pinky_folded:
        return "fist"
    elif all_open:
        return "palm"
    elif thumb_closed and index_open and middle_open and not ring_folded and not pinky_folded:
        return "thumbs_up"
    elif thumb_closed and is_half_folded(MIDDLE_TIP, MIDDLE_PIP, lm) and \
         is_half_folded(RING_TIP, RING_PIP, lm) and is_half_folded(PINKY_TIP, PINKY_PIP, lm):
        return "thumbs_down"
    elif thumb_closed and is_folded(INDEX_TIP, INDEX_PIP, lm) and is_folded(MIDDLE_TIP, MIDDLE_PIP, lm) and ring_folded and pinky_folded:
        if norm_thumb_index > 1.4:
            return "no_hand"
        elif 0.4 < norm_thumb_index <= 1.4:
            return "no_hand"
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
THRESHOLD = 0.01  # min distance change to detect increase/decrease
DEBOUNCE = 0.002   # seconds between actions
# Main loop
def loop():
    global ledLastControlTime, zoom_mode_active, last_action_time, ledStatus

    try:
        print("? Gesture detection started (ESC to quit)...")
        while True:
            time.sleep(0.03)
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            gesture = "no_hand"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    normalized_landmarks = hand_landmarks.landmark
                    gesture = detect_gesture(normalized_landmarks)
                    # Thumb-Index distance detection for "increase"/"decrease"
                    thumb_tip = hand_landmarks.landmark[THUMB_TIP]
                    index_tip = hand_landmarks.landmark[INDEX_TIP]
                    current_dist = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

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
                            if prev_thumb_index_dist is not None and time.time() - last_action_time > DEBOUNCE :
                                if current_dist - prev_thumb_index_dist > THRESHOLD:
                                    print("?? increase")
                                    gesture="thumbs_up"
                                    last_action_time = time.time()
                                elif prev_thumb_index_dist - current_dist > THRESHOLD:
                                    print("?? decrease")
                                    gesture="thumbs_down"
                                    last_action_time = time.time()
                            prev_thumb_index_dist = current_dist
                    else:
                        if zoom_mode_active and time.time() - zoom_gesture_start_time > ZOOM_TIMEOUT:
                            zoom_mode_active = False
                            prev_thumb_index_dist = None
                    if gesture != "no_hand" and (ledLastControlTime is None or time.time() - ledLastControlTime >= 0.1)  :
                        updateLedStatus(gesture)

            print(f"Detected Gesture: {gesture}")
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Gesture Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
                break

    except KeyboardInterrupt:
        print("? Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        turnOffLed() 
        project_ref.update({'led_status': 'off'})

def turnOffLed():
    global ledStatus
    led.value = 0
    ledStatus = "off"        
        
if __name__ == '__main__':
    print("Program is starting ...") 
    attachEventListener()    
    
    try:
        loop()
    except KeyboardInterrupt:  # Handle Ctrl+C to exit
        print("? Stopped by user.") 
        turnOffLed()   
        project_ref.update({'led_status': 'off'})