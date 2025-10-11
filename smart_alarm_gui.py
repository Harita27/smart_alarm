import cv2
import time
import threading
import numpy as np
import pygame
import pyautogui
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from torchvision import models

# ----------------- Alarm -----------------
def play_alarm():
    pygame.mixer.init()
    try:
        pygame.mixer.music.load("alarm.mp3")  # make sure this file exists
        pygame.mixer.music.play(-1)  # loop until stopped
    except Exception as e:
        print(f"⚠️ Alarm sound error: {e}")

# ----------------- Haar cascades -----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("❌ ERROR: Haar cascade files not found in OpenCV.")
    exit()

# ----------------- Scene Detection Model -----------------
scene_model = models.resnet18(num_classes=365)
checkpoint = torch.load("resnet18_places365.pth", map_location=torch.device('cpu'))
# Some checkpoints may have extra keys, adjust accordingly
if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k.replace('module.', '')  # remove `module.` if exists
    new_state_dict[name] = v
scene_model.load_state_dict(new_state_dict)
scene_model.eval()

# Load scene categories
with open("categories_places365.txt") as f:
    categories = [line.strip().split(' ')[0][3:] for line in f]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def is_bedroom(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    inp = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = scene_model(inp)
        idx = torch.argmax(out)
        label = categories[idx]
        print(f"Scene detected: {label}")
        return 'bedroom' in label.lower()

# ----------------- Variables -----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ ERROR: Cannot access webcam.")
    exit()

open_start = None
required_eye_time = 10
required_shake_time = 50
cumulative_shake_time = 0
shake_prev_time = time.time()
prev_mouse_pos = pyautogui.position()

uploaded_image_checked = False
alarm_stopped = False

# ----------------- Tkinter GUI -----------------
root = Tk()
root.title("Smart Alarm")
root.geometry("800x600")

label = Label(root)
label.pack()

status_label = Label(root, text="Alarm Active", font=("Arial", 16), fg="red")
status_label.pack()

eye_label = Label(root, text="Eyes open: 0.0 s", font=("Arial", 14))
eye_label.pack()

shake_label = Label(root, text="Mouse shake: 0.0 s", font=("Arial", 14))
shake_label.pack()

def stop_alarm():
    global alarm_stopped
    pygame.mixer.music.stop()
    alarm_stopped = True
    status_label.config(text="Alarm Stopped ✅", fg="green")
    cap.release()
    cv2.destroyAllWindows()
    root.quit()

def upload_photo():
    global uploaded_image_checked
    file_path = filedialog.askopenfilename(title="Select photo", filetypes=[("Image files","*.jpg *.png *.jpeg")])
    if file_path:
        uploaded_image_checked = True
        if is_bedroom(file_path):
            print("⚠️ Photo is from bedroom! Not accepted.")
        else:
            print("✅ Photo accepted. Alarm off.")
            stop_alarm()

upload_btn = Button(root, text="Upload Photo", command=upload_photo, font=("Arial",12))
upload_btn.pack()

# Start alarm thread
threading.Thread(target=play_alarm, daemon=True).start()

# ----------------- Update Loop -----------------
def update_frame():
    global open_start, cumulative_shake_time, shake_prev_time, prev_mouse_pos, alarm_stopped

    if alarm_stopped:
        return

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    eyes_open = False

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            eyes_open = True
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)

    # Eyes timer
    if eyes_open:
        if open_start is None:
            open_start = time.time()
        elapsed = time.time() - open_start
        eye_label.config(text=f"Eyes open: {elapsed:.1f} s")
        if elapsed >= required_eye_time:
            print("✅ Eyes open for 10s. Alarm off.")
            stop_alarm()
            return
    else:
        open_start = None
        eye_label.config(text=f"Eyes open: 0.0 s")

    # Mouse shake
    current_mouse_pos = pyautogui.position()
    now = time.time()
    if current_mouse_pos != prev_mouse_pos:
        cumulative_shake_time += now - shake_prev_time
        shake_label.config(text=f"Mouse shake: {cumulative_shake_time:.1f} s")
        if cumulative_shake_time >= required_shake_time:
            print("✅ Mouse shaken for 50s. Alarm off.")
            stop_alarm()
            return
    shake_prev_time = now
    prev_mouse_pos = current_mouse_pos

    # Convert frame to Tkinter image
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    root.after(10, update_frame)

update_frame()
root.mainloop()
