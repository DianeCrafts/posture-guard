import mediapipe as mp

class Renderer:
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def draw_pose(self, frame, pose_result):
        if pose_result.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )
        return frame
