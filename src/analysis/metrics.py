import math
import numpy as np


def midpoint(p1, p2):
    return np.array([
        (p1.x + p2.x) / 2,
        (p1.y + p2.y) / 2
    ])


def to_np(landmark):
    return np.array([landmark.x, landmark.y])


def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.degrees(math.acos(dot))


def neck_angle(landmarks):
    """
    Angle between shoulder->head vector and vertical.
    """
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    nose = landmarks[0]

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    head = to_np(nose)

    head_vec = head - shoulder_mid
    vertical = np.array([0, -1])

    return angle_between(head_vec, vertical)


def shoulder_height_diff(landmarks):
    """
    Absolute difference between shoulder heights.
    """
    left = landmarks[11]
    right = landmarks[12]
    return abs(left.y - right.y)
