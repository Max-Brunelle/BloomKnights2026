"""Live webcam squat coach - ties vision, scoring, IMU, and coaching together."""

import math
import statistics
import textwrap
import time

import cv2

from coach import get_coaching
from imu_listener import IMUListener
from scoring import RepCounter
from vision import (
    BODY_CONNECTIONS,
    EMA,
    FIRST_BODY_LANDMARK,
    PROFILE_THRESHOLD,
    knee_angle_facing_camera,
    mp_drawing,
    mp_pose,
    profile_score,
)

WINDOW_NAME = "Squat Coach"
EMA_ALPHA = 0.3
SET_SIZE = 5
SET_IDLE_TIMEOUT_S = 10.0
WRAP_WIDTH = 70

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (0, 0, 255)
ORANGE = (0, 140, 255)


def draw_overlay(frame, side, angle, in_profile, rep_count, last_completed_rep, coaching_text, generating):
    h, _w = frame.shape[:2]

    angle_text = f"{side} knee: {angle:.1f}" if angle is not None else "no pose detected"
    cv2.putText(frame, angle_text, (10, 30), FONT, 0.7, WHITE, 2)

    profile_text = "IN PROFILE" if in_profile else "TURN SIDE-ON"
    cv2.putText(frame, profile_text, (10, 60), FONT, 0.7, GREEN if in_profile else RED, 2)

    cv2.putText(frame, f"Rep: {rep_count}", (10, 90), FONT, 0.7, WHITE, 2)

    y = 120
    if last_completed_rep is not None:
        cv2.putText(frame, f"Last score: {last_completed_rep['score']}", (10, y), FONT, 0.7, WHITE, 2)
        y += 30
        for warning in last_completed_rep["warnings"]:
            cv2.putText(frame, warning, (10, y), FONT, 0.6, ORANGE, 2)
            y += 25

    if generating:
        cv2.putText(frame, "Generating coaching feedback...", (10, h - 20), FONT, 0.7, WHITE, 2)
    elif coaching_text:
        lines = textwrap.wrap(coaching_text, width=WRAP_WIDTH)
        start_y = h - 20 - (len(lines) - 1) * 25
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, start_y + i * 25), FONT, 0.6, WHITE, 2)


def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    pose = mp_pose.Pose()
    imu = IMUListener()
    imu.start()

    ema = EMA(alpha=EMA_ALPHA)
    rep_counter = RepCounter()

    completed_reps = []
    imu_samples = []
    side = None
    smoothed_angle = None
    in_profile = False
    last_completed_rep = None
    coaching_text = ""
    last_rep_timestamp = time.time()
    frames_processed = 0
    session_reps_total = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_processed += 1
            now = time.time()

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                side, angle, _visibility = knee_angle_facing_camera(landmarks)
                in_profile = profile_score(landmarks) < PROFILE_THRESHOLD
                smoothed_angle = ema.update(angle)

                accel_mag = None
                if imu.is_connected():
                    packet = imu.get_latest()
                    accel_mag = math.sqrt(packet["ax"] ** 2 + packet["ay"] ** 2 + packet["az"] ** 2)

                rep = rep_counter.update(smoothed_angle, in_profile, now)
                if rep is not None:
                    if len(imu_samples) >= 2:
                        rep["accel_std"] = statistics.stdev(imu_samples)
                    imu_samples.clear()
                    completed_reps.append(rep)
                    last_completed_rep = rep
                    last_rep_timestamp = now
                    session_reps_total += 1
                elif rep_counter.state == "STANDING":
                    imu_samples.clear()  # idle or aborted attempt - drop stale samples
                elif accel_mag is not None:
                    imu_samples.append(accel_mag)

                for i in range(FIRST_BODY_LANDMARK):
                    landmarks[i].visibility = 0
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, BODY_CONNECTIONS)
            else:
                in_profile = False

            draw_overlay(frame, side, smoothed_angle, in_profile, len(completed_reps),
                         last_completed_rep, coaching_text, generating=False)

            idle_too_long = completed_reps and (now - last_rep_timestamp) > SET_IDLE_TIMEOUT_S
            if len(completed_reps) >= SET_SIZE or idle_too_long:
                draw_overlay(frame, side, smoothed_angle, in_profile, len(completed_reps),
                             last_completed_rep, coaching_text, generating=True)
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(1)

                coaching_text = get_coaching(completed_reps)
                completed_reps = []
                last_rep_timestamp = now

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        imu.stop()
        print(f"Frames processed: {frames_processed} | Reps completed this session: {session_reps_total}")


if __name__ == "__main__":
    main()
