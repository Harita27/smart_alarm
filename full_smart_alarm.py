import cv2
import time
import threading
import numpy as np
import pygame
import sys
import pyautogui
from tkinter import Tk, filedialog
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# ----------------- Alarm -----------------
def play_alarm():
    pygame.mixer.init()
    try:
        pygame.mixer.music.load("alarm.mp3")
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"⚠️ Alarm sound error: {e}")

# ----------------- Haar cascades for eyes -----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
if face_cascade.empty() or eye_cascade.empty():
    print("❌ ERROR: Haar cascade files not found.")
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

required_shake_time = 50  # cumulative seconds mouse must move
cumulative_shake_time = 0
shake_prev_time = time.time()
prev_mouse_pos = pyautogui.position()

uploaded_image_checked = False

print(">>> Wake up using ONE of these methods:")
print("1️⃣ Keep eyes open for 10 seconds")
print("2️⃣ Move the mouse cumulatively for 50 seconds")
print("3️⃣ Upload a photo from another room (press 'U')")

# ----------------- Load Places365 model -----------------
device = torch.device("cpu")
scene_model = models.resnet18(num_classes=365)
checkpoint = torch.load("resnet18_places365.pth", map_location=device)
# Some checkpoints are saved with key 'state_dict'
if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']
# remove 'module.' if present
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k.replace('module.', '')  # remove `module.` if present
    new_state_dict[name] = v
scene_model.load_state_dict(new_state_dict)
scene_model.eval()
scene_model.to(device)

# ----------------- Load categories -----------------
with open("categories_places365.txt") as f:
    categories_places365 = [line.strip().split(' ')[0][3:] for line in f.readlines()]

# ----------------- Function to check if image is bedroom -----------------
def is_bedroom(image_path):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = scene_model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    predicted_label = categories_places365[predicted_idx.item()]
    print(f"Predicted scene: {predicted_label}")
    return 'bedroom' in predicted_label.lower()

# ----------------- Function: upload photo -----------------
def check_uploaded_photo():
    global uploaded_image_checked
    Tk().withdraw()  # hide tkinter window
    file_path = filedialog.askopenfilename(title="Select photo from another room",
                                           filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        if not is_bedroom(file_path):
            uploaded_image_checked = True
            print(f"✅ Uploaded photo accepted: {file_path}")
            pygame.mixer.music.stop()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
        else:
            print("❌ Photo is a bedroom! Please upload another room photo.")
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
        # Draw rectangles
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
    current_mouse_pos = pyautogui.position()
    now = time.time()
    if current_mouse_pos != prev_mouse_pos:
        cumulative_shake_time += now - shake_prev_time
        cv2.putText(frame, f"Shaking: {cumulative_shake_time:.1f}s", (30,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if cumulative_shake_time >= required_shake_time:
            print("✅ Mouse shaken for 50 seconds. Alarm off.")
            pygame.mixer.music.stop()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
    shake_prev_time = now
    prev_mouse_pos = current_mouse_pos

    # ----------------- Upload photo option -----------------
    cv2.putText(frame, "Press 'U' to upload photo from another room", (10, frame.shape[0]-20),
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
