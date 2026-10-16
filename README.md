# 🦅🚁 Bird vs Drone Classifier (YOLOv8)

A real-time object-detection project that identifies and classifies **birds** and **drones** in images and video streams using YOLOv8.

The system is designed for use cases such as airspace monitoring, perimeter security, wildlife protection, drone detection, and surveillance analytics.

> **Note:** This project is intended for educational, research, and portfolio purposes. Detection performance depends on dataset quality, image conditions, object distance, lighting, camera angle, and model training quality.

---

## Features

- YOLOv8-based object detection.
- Custom model trained to detect two classes: **Bird** and **Drone**.
- Image inference.
- Video inference.
- Pretrained YOLOv8 weights for transfer learning.
- Modular project structure.
- Bounding-box visualization with class labels and confidence scores.
- Saved inference output images and videos.
- Optional Flask or Streamlit application layer.
- Suitable for extension to real-time camera or RTSP-stream inference.

---

## Tech Stack

- Python 3
- Ultralytics YOLOv8
- PyTorch
- OpenCV
- NumPy
- Streamlit (optional application layer)

---

## Model Details

| Property | Value |
|---|---|
| Model architecture | YOLOv8 |
| Framework | Ultralytics YOLOv8 with PyTorch |
| Number of classes | 2 |
| Class 0 | Bird |
| Class 1 | Drone |
| Dataset configuration | `data.yaml` |
| Base pretrained model | `yolov8n.pt` |
| Custom trained model | `bird_drone_best.pt` |

> YOLOv8 models typically use the `.pt` file extension. If your trained file is currently named `bird_drone_best.pth`, either keep that exact filename in the commands or rename it only after confirming that your loading code supports the change.

---

## Dataset Configuration

The dataset configuration is defined in `data.yaml`.

```yaml
train: data/train/images
val: data/val/images

nc: 2

names:
  - bird
  - drone
```

The expected dataset layout is:

```text
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each image should have a matching YOLO-format label file containing:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinate values must be normalized between `0` and `1`.

---

## Project Structure

```text
BirdDroneClassifier/
├── data/                       # Dataset images and YOLO-format labels
│   ├── train/
│   ├── val/
│   └── test/
├── notebooks/                  # Experiments, EDA, and training notebooks
├── runs/
│   └── detect/                 # YOLO inference outputs
├── src/                        # Optional utility modules
├── app.py                      # Flask/Streamlit application entry point
├── yolo.py                     # Model loading and inference script
├── yolotest.py                 # Model testing/evaluation script
├── data.yaml                   # YOLO dataset configuration
├── bird_drone_best.pt          # Custom-trained model weights
├── yolov8n.pt                  # YOLOv8 Nano pretrained base weights
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

> The `runs/` directory is usually generated automatically by YOLO during training or inference. It is often excluded from Git unless you specifically want to version output examples.

---

## Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/Sanjukta234/BirdDroneClassifier.git](https://github.com/Sanjukta234/BirdDroneClassifier.git)
cd BirdDroneClassifier
```

### 2. Create and Activate a Virtual Environment

Creating a virtual environment is recommended.

#### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you do not have a `requirements.txt` file yet, the main dependencies may include:

```bash
python -m pip install ultralytics opencv-python torch torchvision numpy flask
```

Install `streamlit` only if your application uses Streamlit:

```bash
python -m pip install streamlit
```

---

## Run Inference

### Image Detection

Run detection on a single image:

```bash
python yolo.py --source data/test/image.jpg
```

### Video Detection

Run detection on a video file:

```bash
python yolo.py --source data/test/video.mp4
```

### Webcam Detection

If webcam support is implemented in `yolo.py`, use:

```bash
python yolo.py --source 0
```

Here, `0` refers to the default webcam.

### Run the Test Script

```bash
python yolotest.py
```

---

## Run the Application

### Flask Application

If `app.py` is a Flask application:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

### Streamlit Application

If `app.py` is a Streamlit application, run:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

> Use only the command that matches the framework implemented in `app.py`.

---

## Output

The model generates detection results containing:

- Bounding boxes around detected objects.
- Predicted class labels: `Bird` or `Drone`.
- Confidence scores.
- Saved annotated image or video outputs.

YOLO inference results are typically saved under:

```text
runs/detect/
```

For example:

```text
runs/detect/predict/
runs/detect/predict2/
runs/detect/predict3/
```

---

## Example Detection Output

```text
Bird: 0.92 confidence
Drone: 0.87 confidence
```

The final output image or video displays labels and bounding boxes around each detected object.

---

## Future Enhancements

- Add live CCTV-camera and RTSP-stream support.
- Add real-time webcam inference.
- Deploy the model to edge devices such as NVIDIA Jetson or Raspberry Pi.
- Export and optimize the model using ONNX, TensorRT, or OpenVINO.
- Add object tracking to follow birds and drones across video frames.
- Add alert notifications when a drone is detected.
- Add sound-based drone detection as a second detection signal.
- Create a cloud deployment using Docker and an API framework.
- Add an analytics dashboard for detection history and confidence trends.
- Expand the dataset with more weather, lighting, distance, and background variations.

---

## Security and Responsible Use

- Do not use this project for unlawful surveillance or privacy-invasive activities.
- Validate the model carefully before using it for safety-critical decisions.
- Detection predictions can be incorrect, especially for small, distant, blurred, partially visible, or unusual objects.
- Human review should be included for real-world security or wildlife-management use cases.

---

## Author 

**Sanjukta Bag**

- GitHub: [Sanjukta234](https://github.com/sanjukta2019)
