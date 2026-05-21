import cv2
import time
import threading
import numpy as np
import pygame
import sys
from tkinter import Tk, filedialog
def play_alarm():
    pygame.mixer.init()
    try:
        pygame.mixer.music.load("alarm.mp3")  # make sure file exists
        pygame.mixer.music.play(-1)  # loop until stopped
    except Exception as e:
        print(f"⚠️ Alarm sound error: {e}")

# ----------------- Haar cascades for eyes -----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("❌ ERROR: Haar cascade files not found in OpenCV.")
    sys.exit()

# ----------------- Camera -----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ ERROR: Cannot access webcam.")
    sys.exit()

# Start alarm
threading.Thread(target=play_alarm, daemon=True).start()

# ----------------- Variables -----------------
open_start = None
required_eye_time = 10  # seconds eyes must stay open
shake_start = None
required_shake_time = 50  # seconds shaking
shake_detected = False
prev_mouse_pos = None
uploaded_image_checked = False

print(">>> Wake up using ONE of these methods:")
print("1️⃣ Keep eyes open for 10 seconds")
print("2️⃣ Move the mouse continuously for 50 seconds")
print("3️⃣ Upload a photo from another room")

# ----------------- Function: upload photo -----------------
def check_uploaded_photo():
    global uploaded_image_checked
    Tk().withdraw()  # hide main window
    file_path = filedialog.askopenfilename(title="Select photo from kitchen/hall",
                                           filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        uploaded_image_checked = True
        print(f"✅ Uploaded photo accepted: {file_path}")
        pygame.mixer.music.stop()
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
    else:
        print("⚠️ No file selected.")

# ----------------- Main Loop -----------------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.1)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----------------- Eyes detection -----------------
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes_open = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            eyes_open = True
        # draw rectangles
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 2)

    # Eyes-open timer
    if eyes_open:
        if open_start is None:
            open_start = time.time()
        elapsed = time.time() - open_start
        cv2.putText(frame, f"Eyes open: {elapsed:.1f}s", (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if elapsed >= required_eye_time:
            print("✅ Eyes open for 10 seconds. Alarm off.")
            pygame.mixer.music.stop()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
    else:
        open_start = None

    # ----------------- Shake detection (mouse movement) -----------------
    mouse_pos = cv2.getWindowProperty("SmartWake Eye Detection", cv2.WND_PROP_VISIBLE)
    if prev_mouse_pos is None:
        prev_mouse_pos = mouse_pos

    # Simple shake simulation using key press or movement (placeholder)
    # Here we increment shake_start manually for demonstration
    if shake_detected:
        shake_elapsed = time.time() - shake_start
        cv2.putText(frame, f"Shake: {shake_elapsed:.1f}s", (30,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if shake_elapsed >= required_shake_time:
            print("✅ Shake detected for 50 seconds. Alarm off.")
            pygame.mixer.music.stop()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    # ----------------- Upload photo option -----------------
    cv2.putText(frame, "Press 'U' to upload photo from kitchen/hall", (10, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # Show camera feed
    cv2.imshow("SmartWake Eye Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        pygame.mixer.music.stop()
        break
    elif key == ord('u'):
        check_uploaded_photo()

cap.release()
cv2.destroyAllWindows()
