import cv2
import mediapipe as mp
import numpy as np
import os
import time
import datetime
from utils import get_3d_model_points, draw_axes, rotation_vector_to_euler_angles

# For Eye Aspect Ratio (EAR)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

EYE_AR_THRESH = 0.25

# Gaze Estimation Landmarks
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

LEFT_EYE_LEFT = 33     # outer corner
LEFT_EYE_RIGHT = 133   # inner corner

RIGHT_EYE_LEFT = 362   # inner corner
RIGHT_EYE_RIGHT = 263  # outer corner

# Create folders if needed
os.makedirs("recordings", exist_ok=True)

# Face landmark indices for pose estimation
LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 263,
    "right_eye_corner": 33,
    "left_mouth": 287,
    "right_mouth": 57
}

MOUTH_IDS = {
    "upper_lip": 13,
    "lower_lip": 14
}

# Initialize webcam
cap = cv2.VideoCapture(0)
is_recording = False
record_start_time = None
out = None

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing setup
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def calculate_ear(landmarks, eye_indices, w, h):
    def euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    eye = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_indices]
    
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for i, face_landmarks in enumerate(results.multi_face_landmarks):
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            # Get image size and 2D points
            h, w, _ = frame.shape
            image_points = []
            for name in LANDMARK_IDS:
                lm = face_landmarks.landmark[LANDMARK_IDS[name]]
                x, y = int(lm.x * w), int(lm.y * h)
                image_points.append((x, y))
                cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)

            image_points = np.array(image_points, dtype='double')
            model_points = get_3d_model_points()

            # Camera matrix
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype='double')
            dist_coeffs = np.zeros((4, 1))

            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if success:
                draw_axes(frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                pitch, yaw, roll = rotation_vector_to_euler_angles(rotation_vector)

                # Text offset to avoid overlapping
                label_x = 20  # or use frame.shape[1] - 300 for right side
                label_y_start = 40 + i * 180  # space 150px per face
                line_height = 25

                cv2.putText(frame, f"Face {i+1} Yaw: {yaw:.2f}", (label_x, label_y_start), cv2.FONT_ITALIC, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"Pitch: {pitch:.2f}", (label_x, label_y_start + line_height), cv2.FONT_ITALIC, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"Roll: {roll:.2f}", (label_x, label_y_start + 2*line_height), cv2.FONT_ITALIC, 0.6, (255,255,255), 2)

                # Direction
                if yaw > 15:
                    direction = "Looking Left"
                elif yaw < -15:
                    direction = "Looking Right"
                elif pitch > 10:
                    direction = "Looking Down"
                elif pitch < -10:
                    direction = "Looking Up"
                elif roll > 15:
                    direction = "Tilt Right"
                elif roll < -15:
                    direction = "Tilt Left"
                else:
                    direction = "Facing Forward"

                cv2.putText(frame, direction, (label_x, label_y_start + 3*line_height), cv2.FONT_ITALIC, 0.6, (0,255,255), 2)

                # --- Mouth State Detection ---
                upper_lip = face_landmarks.landmark[MOUTH_IDS["upper_lip"]]
                lower_lip = face_landmarks.landmark[MOUTH_IDS["lower_lip"]]

                lip_distance = abs(upper_lip.y - lower_lip.y)

                if lip_distance > 0.03:
                    mouth_state = "Open"
                else:
                    mouth_state = "Close"

                cv2.putText(frame, f"Mouth: {mouth_state}", (label_x, label_y_start + 5*line_height), cv2.FONT_ITALIC, 0.6, (0,128,255), 2)

                # EAR calculation
                left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE_LANDMARKS, w, h)
                right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE_LANDMARKS, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                eye_state = "Open" if avg_ear > EYE_AR_THRESH else "Close"
                cv2.putText(frame, f"Eye State: {eye_state}", (label_x, label_y_start + 4*line_height), cv2.FONT_ITALIC, 0.6, (0,255,0), 2)

                # --- Gaze Estimation ---
                def get_gaze_ratio(iris_center, eye_left, eye_right):
                    eye_width = abs(eye_right - eye_left)
                    iris_pos = abs(iris_center - eye_left)
                    ratio = iris_pos / eye_width
                    return ratio

                # Get normalized X-coordinates
                left_iris_x = face_landmarks.landmark[LEFT_IRIS_CENTER].x
                left_eye_left_x = face_landmarks.landmark[LEFT_EYE_LEFT].x
                left_eye_right_x = face_landmarks.landmark[LEFT_EYE_RIGHT].x

                right_iris_x = face_landmarks.landmark[RIGHT_IRIS_CENTER].x
                right_eye_left_x = face_landmarks.landmark[RIGHT_EYE_LEFT].x
                right_eye_right_x = face_landmarks.landmark[RIGHT_EYE_RIGHT].x

                left_ratio = get_gaze_ratio(left_iris_x, left_eye_left_x, left_eye_right_x)
                right_ratio = get_gaze_ratio(right_iris_x, right_eye_left_x, right_eye_right_x)
                avg_ratio = (left_ratio + right_ratio) / 2.0

                # Estimate direction
                if avg_ratio <= 0.35:
                    gaze = "Looking Right"
                elif avg_ratio >= 0.65:
                    gaze = "Looking Left"
                else:
                    gaze = "Looking Center"
                
                cv2.putText(frame, f"Gaze: {gaze}", (label_x, label_y_start + 6*line_height), cv2.FONT_ITALIC, 0.6, (255,255,0), 2)

    else:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show Recording Status (top-right, after flip)
    if is_recording and out is not None:
        out.write(frame)

        elapsed_time = int(time.time() - record_start_time)
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        timer_text = f"{minutes:02}:{seconds:02}"
        recording_text = f"REC {timer_text}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (text_width, _), _ = cv2.getTextSize(recording_text, font, scale, thickness)
        x_position = frame.shape[1] - text_width - 20

        cv2.putText(frame, recording_text, (x_position, 30),
                    font, scale, (0, 0, 255), thickness)

    cv2.imshow('Face Pose Estimation', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    elif key == 32:  # SPACE to toggle recording
        if not is_recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"recordings/pose_recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame_width, frame_height))
            record_start_time = time.time()
            is_recording = True
            print(f"Started recording: {video_filename}")
        else:
            is_recording = False
            out.release()
            out = None
            print("Stopped recording.")

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
