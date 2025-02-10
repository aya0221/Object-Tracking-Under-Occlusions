
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

## Contributing

We welcome contributions to this project. Feel free to submit pull requests or open issues if you encounter bugs or want to suggest new features.

To contribute:
1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.


---

## Acknowledgements

- OpenCV for computer vision tasks
- TensorFlow for object detection model inference
- The Kalman Filter for object tracking
- The COCO dataset for pre-trained model

---

## Conclusion

This project is a demonstration of combining object detection and tracking using deep learning and traditional algorithms like the Kalman Filter. It's designed to be modular and easy to adapt to different types of objects or video streams. With further improvements, it could be extended to real-time applications such as video surveillance or autonomous vehicles.

