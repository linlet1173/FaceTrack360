import numpy as np
import cv2
import math

def get_3d_model_points():
    return np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

def draw_axes(frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    axis = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100]])  # X, Y, Z
    origin = np.float32([[0, 0, 0]])
    p_origin, _ = cv2.projectPoints(origin, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    p_axes, _ = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    origin_point = tuple(p_origin[0].ravel().astype(int))
    for p, color in zip(p_axes, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # Z, Y, X
        axis_point = tuple(p.ravel().astype(int))
        cv2.line(frame, origin_point, axis_point, color, 2)

def rotation_vector_to_euler_angles(rvec):
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])  # pitch, yaw, roll