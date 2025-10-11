```markdown
# Smart Alarm

Smart Alarm is a Python application designed to help you wake up using **three interactive methods**:

1. **Eyes Open Detection:** Keep your eyes open for 10 seconds using the webcam.  
2. **Mouse Shake Detection:** Move your mouse cumulatively for 50 seconds.  
3. **Upload Photo from Another Room:** Upload a photo of a room other than your bedroom.  

Completing any of the three tasks successfully will stop the alarm.

---

## Features

- Webcam-based eye detection using OpenCV Haar cascades.
- Mouse movement detection using PyAutoGUI.
- Scene recognition using **pre-trained Places365 ResNet18 model**.
- Alarm sound using Pygame.

---

## Required Files

Due to GitHub file size limits, the following files **cannot** be uploaded:

1. `resnet18_places365.pth` – Pre-trained model weights (download separately).  
2. `categories_places365.txt` – Scene categories for the Places365 model.  
3. `alarm.mp3` – Alarm sound file.  

Place these files in the project root folder.

---

### Download Links

- **Places365 ResNet18 model:** [Download here](http://places2.csail.mit.edu/models_places365/resnet18_places365.pth)  
- **Categories file:** [Download here](https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt)  
- **Alarm sound:** Use any MP3 file named `alarm.mp3` in the project root.

---

## Project Structure

```

smartalarm/
│
├─ full_smart_alarm.py        # Main Python script
├─ alarm.mp3                  # Alarm sound file (must be downloaded)
├─ resnet18_places365.pth     # Pre-trained Places365 model (download manually)
├─ categories_places365.txt   # Scene categories (download manually)
├─ README.md                  # This file
├─ venv/                      # Python virtual environment

````

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd smartalarm
````

### 2. Create Python Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

* **Windows (PowerShell)**

```bash
& .\venv\Scripts\Activate.ps1
```

* **Windows (cmd)**

```bash
venv\Scripts\activate.bat
```

* **Linux/Mac**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install opencv-python pygame pyautogui torch torchvision pillow
```

### 5. Place large files

Download and place the following in your project root:

* `resnet18_places365.pth`
* `categories_places365.txt`
* `alarm.mp3`

---

## Run the Application

```bash
python full_smart_alarm.py
```

* Follow prompts to either keep your eyes open, shake the mouse, or upload a photo.
* Completing any task will stop the alarm.

---

## Important Notes

* **Webcam:** Make sure your webcam is functioning.
* **Mouse Shake:** Move the mouse for cumulative 50 seconds.
* **Upload Photo:** Ensure the photo is **not of your bedroom**, otherwise it will be rejected.
* **Alarm Sound:** Confirm `alarm.mp3` exists in the project root.

---

## Tips

* When using the Places365 model on CPU:

```python
checkpoint = torch.load('resnet18_places365.pth', map_location=torch.device('cpu'))
```

* If the webcam feed doesn’t show, ensure OpenCV is installed correctly and no other application is using the webcam.


