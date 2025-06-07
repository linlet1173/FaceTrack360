# FaceTrack360: Multi-Face Pose and State Detection System 

## Description

This project performs real-time and static image-based face pose estimation using MediaPipe and OpenCV. It includes pose angle calculation (yaw, pitch, roll), eye state detection (open/closed), mouth state detection (open/closed), and gaze estimation (left, right, center). The system supports multi-face detection and offers both webcam-based video input and static image analysis.

## Features

### Main Features (Real-Time via main.py)

- Webcam-based live face tracking
- 3D face pose estimation (Yaw, Pitch, Roll)
- Eye state detection (based on Eye Aspect Ratio)
- Mouth state detection (based on lip distance)
- Gaze direction estimation (Left, Center, Right)
- Start/Stop video recording with Spacebar
- On-screen recording timer
- Multi-face support (up to 5 faces)

### Static Image Support (image_input.py)
- Supports any image file (.jpg, .png)
- Detects multiple faces
- Calculates pose, eye/mouth state, gaze direction
- Automatically saves annotated output image in output_images/

## Folder Structure
    term-project-cv/
    ├── main.py                  # Real-time webcam estimation
    ├── image_input.py           # Static image analysis
    ├── utils.py                 # Utility functions (3D model points, drawing axes, etc.)
    ├── requirements.txt
    ├── recordings/              # Saved video output recordings (auto-created) 
    ├── output_images/           # Saved output images (auto-created)
    ├── test.jpg                 # Example input images ( test1.jpg, test2.jpg, test3.jpg)
    └── README.md                # Project description 

## Requirements
- **Python3.8+**
- **OpenCV**
- **MediaPipe**
- **NumPy**

        pip install opencv-python mediapipe numpy

## How to Run
1. Install dependencies (recommend using a virtual environment):

       pip install opencv-python mediapipe numpy

2. For live camera tracking:

       python main.py
    - Press SPACE to start/stop recording
    - Press ESC to exit

3. For static image analysis:
    - Firstly, place the image file (e.g., test1.jpg) in the root directory
    - Set the filename in image_input.py
    - Run:

          python image_input.py


## DEMO video and Screenshot
**1. Webcam Demo & Screenshot (multi-face detection)**

Since video file size is too large, I am going to share the google drive file.
- **Webcam demo drive link**
https://drive.google.com/file/d/1EcSyBbw5WhqtxZg2s6Unz2uDeIs5MBoj/view?usp=sharing
- **Webcam screenshot**
<img width="1391" alt="mutiface detection webver" src="https://github.com/user-attachments/assets/dfbea0a5-417a-4c1b-a680-8e7852a5e679" />

**2. Static image output (multi-face detection)**

Output of test3.jpg from image_input.py showing pose and state analysis of multiple faces in a static image, with results automatically saved and labeled.


![test3_faces3](https://github.com/user-attachments/assets/7ea815f7-c841-4f6f-a4a3-a49179e2498e)

## Notes
- Works best in well-lit environments
- Uses MediaPipe FaceMesh with refined landmarks for improved iris and lip detection
- Gaze and eye/mouth state are estimated heuristically and may vary based on pose and resolution.
- Labels dynamically positioned to avoid overlap on the side of the frame
- The auto created output recordings folder cannot be uploaded due to larger file size. 

## License
This project is open-source and free to use for academic or research purposes. License: MIT

