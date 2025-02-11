
# Multi-Object Detection and Tracking with Kalman Filter

## Overview

This project implements multi-object detection and tracking using **OpenCV**, **TensorFlow**, and the **Kalman Filter** algorithm. The primary goal is to detect objects (specifically sports balls) in a video stream and track their movements across frames, even when they are temporarily occluded. The system utilizes **Single Shot Multibox Detector (SSD)** with the **MobileNetV3** architecture for real-time object detection and **Kalman Filter** for accurate object tracking.

---

## Key Features

- **Multi-object Detection**: Using the **SSD MobileNetV3** model pre-trained on the **COCO dataset**, the system can detect various objects in real-time.
- **Object Tracking**: Implements **Kalman Filter** to track objects across frames, even in the presence of occlusion or movement interruptions.
- **Real-time Processing**: Capable of processing video streams and tracking multiple objects simultaneously.
- **Customizable**: Easily extendable to track other object types or use different pre-trained models for detection.

---

## Requirements

To run this project, the following dependencies are required:

- Python 3.9+
- OpenCV 4.5.x or later
- NumPy 1.21.x or later
- TensorFlow (for SSD model inference)
- FFmpeg (for video processing)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multi-object-detection-tracking.git
cd multi-object-detection-tracking
```

### 2. Install Dependencies

It is recommended to create a virtual environment to manage dependencies. You can do so with the following commands:

```bash
python3 -m venv kalman_filter_env
source kalman_filter_env/bin/activate  # On Windows use `kalman_filter_env\Scripts\activate`
pip install -r requirements.txt
```

Alternatively, you can install the dependencies manually:

```bash
pip install opencv-python numpy tensorflow ffmpeg-python
```

### 3. Download Pre-trained Model

This project uses a pre-trained **SSD MobileNetV3** model. You can download the model from the following source:

- **Model Configuration File** (`ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`)
- **Frozen Inference Graph** (`frozen_inference_graph.pb`)
- **Class Names** (`coco.names`)

Place the downloaded files into the `model_SSD/ssd_mobilenet_v3_large_coco_2020_01_14/` directory.

---

## How to Run

### 1. Running with a Sample Video

To run the program on a sample video, use the following command:

```bash
python main.py
```

This will use the `videos/multiObject.avi` file as input and perform object detection and tracking.

### 2. Custom Video Input

To run the program on a different video, update the `videoPath` variable in `main.py`:

```python
videoPath = "path_to_your_video.mp4"
```

Make sure the video file exists in the specified path.

### 3. Video Output

The output video with bounding boxes and tracking paths will be saved in the project directory with a name corresponding to the input video. For example, if the input is `multiObject.avi`, the output will be saved as `multiball.avi`.

---

## How It Works

### 1. Object Detection

- The program uses a pre-trained **SSD MobileNetV3** model to detect objects in each frame.
- It specifically tracks **sports balls** (class ID 37 in the COCO dataset), but can be modified to track other objects.

### 2. Kalman Filter for Object Tracking

- **Kalman Filter** is used to predict the next location of the object based on its current location and velocity.
- If the object is occluded or temporarily out of view, the Kalman Filter continues to predict its movement based on past data.

### 3. Bounding Box & Tracking

- When a bounding box for a detected object is found, the center of the bounding box is calculated, and the Kalman Filter updates its state with the new position.
- If no bounding box is detected, the filter predicts the object's location without any correction.

### 4. Output

- The program draws bounding boxes around detected objects and circles around the predicted positions in the video feed.
- It outputs the video with bounding boxes and predictions applied.

---

## File Structure

```plaintext
multi-object-detection-tracking/
│
├── main.py                    # Main entry point for running the program
├── Detector.py                # Class responsible for object detection and tracking logic
├── kalmanfilter.py            # Kalman Filter implementation for tracking
├── model_SSD/                 # Directory containing the SSD model files
│   ├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
│   ├── frozen_inference_graph.pb
│   ├── coco.names
├── videos/                    # Sample video files for testing
├── requirements.txt           # List of required Python packages
└── README.md                  # Project documentation (this file)
```

---
---

### **I. SSD MobileNet V3**
- **SSD (Single Shot MultiBox Detector)**:
  - A single-stage object detection model that directly predicts bounding boxes and class labels in a single forward pass.
  - Unlike two-stage models like Faster R-CNN, SSD does not need a separate region proposal step, making it much faster.

- **MobileNet V3 Backbone**:
  - A lightweight CNN designed for mobile and edge devices.
  - Uses depthwise separable convolutions for efficiency.
  - Incorporates **Squeeze-and-Excitation (SE) blocks** to enhance feature extraction.

- **How It Works**:
  - Image is passed through **MobileNet V3** for feature extraction.
  - SSD applies multiple convolutional layers to detect objects at different scales.
  - Predictions are made directly on feature maps using **default anchor boxes**.

---

### **II. SSD MobileNet V3 vs Faster R-CNN or YOLO**
- **Compared to Faster R-CNN**:
  - SSD is **faster** but **less accurate**.
  - Faster R-CNN is two-stage (slower but more precise).
  - SSD is better for real-time applications where speed is important.

- **Compared to YOLO**:
  - **YOLO (You Only Look Once)** is faster than SSD but may struggle with small objects.
  - **SSD is more balanced** in terms of speed and accuracy.
  - **MobileNet V3 makes SSD more lightweight** than YOLO, especially for mobile and embedded devices.

---

### **III. Key files needed to run inference**
| File Type | Purpose |
|-----------|---------|
| **Frozen Model (.pb or SavedModel format)** | Stores trained model weights. |
| **Configuration File (.pbtxt)** | Defines model structure and parameters. |
| **Label Map (.pbtxt or .txt)** | Maps class IDs to object names (e.g., COCO labels). |

---

### **IV. How to fine-tune the model on a custom dataset**
1. **Prepare the Dataset**:
   - Collect images and annotate objects in **Pascal VOC or TFRecord format**.
   - Create a **label map** for new object classes.

2. **Modify Pipeline Configuration File**:
   - Update the dataset path.
   - Adjust **batch size, learning rate, and training steps**.
   - Set the number of object classes.

3. **Train the Model**:
   - Use TensorFlow Object Detection API.
   - Fine-tune using a **pre-trained COCO model**.

4. **Export the Trained Model**:
   - Convert to **SavedModel format** for inference.

---

### **V. Potential optimizations that can be applied for real-time inference**
1. **Quantization**: Convert model weights to lower precision (e.g., **INT8** or **FLOAT16**) for faster execution.
2. **TensorFlow Lite (TFLite)**: Convert model to **TFLite** for deployment on mobile and embedded devices.
3. **TensorRT Optimization**: Use NVIDIA **TensorRT** for GPU acceleration.
4. **Batch Inference**: Process multiple images in one pass.
5. **Reduce Input Size**: Lower resolution images for faster inference with minimal accuracy loss.

---

### **VI. COCO dataset**
- **COCO Dataset Features**:
  - 80 object classes.
  - Large-scale, diverse dataset covering real-world scenes.
  - Annotated with bounding boxes, segmentation masks, and keypoints.

- **Impact on Model**:
  - Pre-trained models on COCO **generalize well** to various detection tasks.
  - May **struggle with domain-specific objects** if fine-tuning isn’t done.
  - Provides a **strong baseline** for transfer learning.
