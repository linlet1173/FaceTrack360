import cv2
import mediapipe as mp
import numpy as np
import os
from utils import get_3d_model_points, draw_axes, rotation_vector_to_euler_angles

input_filename = 'test1.jpg'

if not os.path.exists(input_filename):
    raise FileNotFoundError(f"❌ Input image not found: {input_filename}")
image = cv2.imread(input_filename)
if image is None:
    raise ValueError("❌ Could not load the image.")
h, w, _ = image.shape
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True)

LANDMARK_IDS = {
    "nose_tip": 1, "chin": 152,
    "left_eye_corner": 263, "right_eye_corner": 33,
    "left_mouth": 287, "right_mouth": 57
}
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14]
IRIS_CENTER = 468

def calculate_ear(landmarks, indices):
    a = np.linalg.norm(landmarks[indices[1]] - landmarks[indices[5]])
    b = np.linalg.norm(landmarks[indices[2]] - landmarks[indices[4]])
    c = np.linalg.norm(landmarks[indices[0]] - landmarks[indices[3]])
    return (a + b) / (2.0 * c)

results = face_mesh.process(rgb_image)
face_count = 0

if results.multi_face_landmarks:
    for i, face_landmarks in enumerate(results.multi_face_landmarks):
        face_count += 1
        landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
        image_points = [(int(landmarks[LANDMARK_IDS[name]][0]), int(landmarks[LANDMARK_IDS[name]][1]))
                        for name in LANDMARK_IDS]
        for pt in image_points:
            cv2.circle(image, pt, 3, (255, 0, 255), -1)

        model_points = get_3d_model_points()
        image_points_np = np.array(image_points, dtype='double')
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype='double')
        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(model_points, image_points_np, camera_matrix, dist_coeffs)

        if success:
            draw_axes(image, rvec, tvec, camera_matrix, dist_coeffs)
            pitch, yaw, roll = rotation_vector_to_euler_angles(rvec)

            # Calculate dynamic text position for each face
            label_x = 20  
            label_y_start = 40 + i * 180  
            line_height = 25
            
            cv2.putText(image, f"Face {i+1} Yaw: {yaw:.2f}", (label_x, label_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(image, f"Pitch: {pitch:.2f}", (label_x, label_y_start + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(image, f"Roll: {roll:.2f}", (label_x, label_y_start + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2
            eye_state = "Close" if avg_ear < 0.2 else "Open"
            cv2.putText(image, f"Eye: {eye_state}", (label_x, label_y_start + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            mouth_open = np.linalg.norm(landmarks[MOUTH[0]] - landmarks[MOUTH[1]])
            threshold = h * 0.001
            mouth_state = "Open" if mouth_open > threshold else "Close"
            cv2.putText(image, f"Mouth: {mouth_state}", (label_x, label_y_start + 4*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

            iris_x = landmarks[IRIS_CENTER][0]
            eye_left = landmarks[LEFT_EYE[0]][0]
            eye_right = landmarks[LEFT_EYE[3]][0]
            iris_ratio = (iris_x - eye_left) / (eye_right - eye_left + 1e-6)
            if iris_ratio < 0.35:
                gaze = "Right"
            elif iris_ratio > 0.65:
                gaze = "Left"
            else:
                gaze = "Center"
            cv2.putText(image, f"Gaze: {gaze}", (label_x, label_y_start + 5*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

else:
    cv2.putText(image, "No face detected", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Save output
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)
base = os.path.splitext(os.path.basename(input_filename))[0]
output_path = os.path.join(output_dir, f"{base}_faces{face_count}.jpg")
cv2.imwrite(output_path, image)

# Display image
cv2.imshow('Static Image Pose Estimation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"✅ Saved output to: {output_path}")
