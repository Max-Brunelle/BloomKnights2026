"""Live webcam squat coach - ties vision, scoring, IMU, and coaching together."""

import math
import statistics
import textwrap
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
SET_IDLE_TIMEOUT_S = 20.0
WRAP_WIDTH = 55

WHITE = (255, 255, 255)
GREEN = (60, 200, 60)
RED = (220, 80, 60)
AMBER = (230, 180, 40)
PANEL_FILL = (0, 0, 0, 153)  # ~60% opacity dark card
PAD = 12
LINE_H = 32

try:
    FONT_MAIN = ImageFont.truetype("segoeui.ttf", 24)
    FONT_SMALL = ImageFont.truetype("segoeui.ttf", 20)
except OSError:
    FONT_MAIN = FONT_SMALL = ImageFont.load_default()


def score_color(score):
    if score >= 85:
        return GREEN
    if score >= 60:
        return AMBER
    return RED


def draw_overlay(frame, side, angle, in_profile, rep_count, last_completed_rep, coaching_text, generating):
    """Composites panel-based overlay onto frame; returns a new BGR array."""
    h, w = frame.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Main stats panel, one grouped card top-left.
    angle_text = f"{side} knee: {angle:.0f}" if angle is not None else "No pose detected"
    score_text = f"Last score: {last_completed_rep['score']}" if last_completed_rep else "Last score: --"
    lines = [
        (angle_text, WHITE),
        ("IN PROFILE" if in_profile else "TURN SIDE-ON", GREEN if in_profile else RED),
        (f"Rep: {rep_count}", WHITE),
        (score_text, score_color(last_completed_rep["score"]) if last_completed_rep else WHITE),
    ]
    panel_w = PAD * 2 + max(int(draw.textlength(t, font=FONT_MAIN)) for t, _ in lines)
    panel_h = PAD * 2 + LINE_H * len(lines)
    draw.rectangle([10, 10, 10 + panel_w, 10 + panel_h], fill=PANEL_FILL)
    for i, (text, color) in enumerate(lines):
        draw.text((10 + PAD, 10 + PAD + i * LINE_H), text, font=FONT_MAIN, fill=color)

    # Warnings card below the main panel.
    warnings = last_completed_rep["warnings"] if last_completed_rep else []
    if warnings:
        wy = 10 + panel_h + 20
        wpanel_w = PAD * 2 + max(int(draw.textlength(t, font=FONT_SMALL)) for t in warnings)
        wpanel_h = PAD * 2 + (LINE_H - 6) * len(warnings)
        draw.rectangle([10, wy, 10 + wpanel_w, wy + wpanel_h], fill=PANEL_FILL)
        for i, warning in enumerate(warnings):
            draw.text((10 + PAD, wy + PAD + i * (LINE_H - 6)), warning, font=FONT_SMALL, fill=AMBER)

    # Coaching panel spanning the bottom; only once there's something to say.
    bottom_lines = (["Generating coaching feedback..."] if generating
                    else textwrap.wrap(coaching_text, width=WRAP_WIDTH) if coaching_text else [])
    if bottom_lines:
        cpanel_h = PAD * 2 + (LINE_H - 6) * len(bottom_lines)
        draw.rectangle([0, h - cpanel_h, w, h], fill=PANEL_FILL)
        for i, line in enumerate(bottom_lines):
            draw.text((PAD, h - cpanel_h + PAD + i * (LINE_H - 6)), line, font=FONT_SMALL, fill=WHITE)

    img = Image.alpha_composite(img, overlay)
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    pose = mp_pose.Pose()
    imu = IMUListener()
    imu.start()

    print("Any context for your coach today? (e.g. 'recovering from a knee "
          "injury', 'focus on depth', or press Enter to skip)")
    user_context = input("> ").strip()

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
                if frames_processed % 30 == 0:  # throttle so it doesn't flood the console
                    print(f"[DEBUG] connected={imu.is_connected()} accel_mag={accel_mag} buffered_samples={len(imu_samples)}")

                rep = rep_counter.update(smoothed_angle, in_profile, now)
                if rep is not None:
                    sample_count = len(imu_samples)
                    if sample_count >= 2:
                        rep["accel_std"] = statistics.stdev(imu_samples)
                    print(f"[DEBUG] rep {rep['rep_number']} completed: samples_this_rep={sample_count} accel_std={rep.get('accel_std', 'MISSING')}")
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

            display = draw_overlay(frame, side, smoothed_angle, in_profile, len(completed_reps),
                                   last_completed_rep, coaching_text, generating=False)

            idle_too_long = completed_reps and (now - last_rep_timestamp) > SET_IDLE_TIMEOUT_S
            if len(completed_reps) >= SET_SIZE or idle_too_long:
                generating_display = draw_overlay(frame, side, smoothed_angle, in_profile,
                                                  len(completed_reps), last_completed_rep,
                                                  coaching_text, generating=True)
                cv2.imshow(WINDOW_NAME, generating_display)
                cv2.waitKey(1)

                coaching_text = get_coaching(completed_reps, user_context)
                completed_reps = []
                rep_counter = RepCounter()  # fresh state so the next set starts at rep 1
                last_rep_timestamp = now

            cv2.imshow(WINDOW_NAME, display)
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
