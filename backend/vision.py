"""Pose estimation and joint-angle logic for squat tracking. Pure logic, no webcam."""

import math

import cv2  # noqa: F401  (re-exported for main.py)
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Landmarks 0-10 are face; body starts at LEFT_SHOULDER (11).
FIRST_BODY_LANDMARK = mp_pose.PoseLandmark.LEFT_SHOULDER.value

BODY_CONNECTIONS = frozenset(
    (a, b) for a, b in mp_pose.POSE_CONNECTIONS
    if a >= FIRST_BODY_LANDMARK and b >= FIRST_BODY_LANDMARK
)

LEG_JOINTS = {
    "LEFT": (
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
    ),
    "RIGHT": (
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
    ),
}

# profile_score >= this means "not in profile, don't trust this data".
PROFILE_THRESHOLD = 0.40


def angle_at_joint(a, b, c):
    """Angle in degrees at b, formed by rays b->a and b->c. 2D only — monocular z is too noisy."""
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def knee_angle_facing_camera(landmarks):
    """Angle of whichever knee is more visible (the leg facing the camera).

    Returns (side_name, angle_degrees, knee_visibility).
    """
    side = max(LEG_JOINTS, key=lambda s: landmarks[LEG_JOINTS[s][1].value].visibility)
    hip, knee, ankle = (landmarks[lm.value] for lm in LEG_JOINTS[side])
    return side, angle_at_joint(hip, knee, ankle), knee.visibility


def profile_score(landmarks):
    """Horizontal shoulder separation / torso length.

    Near 0 in a clean side profile, rises toward ~0.4+ facing the camera.
    Empirically validated — do not change the formula.
    """
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    shoulder_sep = abs(ls.x - rs.x)
    shoulder_mid = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
    hip_mid = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
    torso_length = math.hypot(shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
    if torso_length == 0:
        return 0.0
    return shoulder_sep / torso_length


class EMA:
    """Exponential moving average smoother."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value


if __name__ == "__main__":
    from collections import namedtuple

    Lm = namedtuple("Lm", "x y z visibility")
    # Right angle at b: a straight up from b, c straight right from b.
    a, b, c = Lm(0, 0, 0, 1), Lm(0, 1, 0, 1), Lm(1, 1, 0, 1)
    assert abs(angle_at_joint(a, b, c) - 90.0) < 1e-6
    print("vision.py self-check OK")
