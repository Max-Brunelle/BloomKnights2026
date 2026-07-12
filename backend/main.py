"""Squat Coach - Tkinter app UI wrapping the existing vision/scoring/IMU/coach pipeline."""

import math
import statistics
import time
import tkinter as tk
from collections import deque
from tkinter import scrolledtext, ttk

import cv2
from PIL import Image, ImageDraw, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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

EMA_ALPHA = 0.3
SET_SIZE = 5
SET_IDLE_TIMEOUT_S = 20.0
VIDEO_W, VIDEO_H = 640, 480
CHART_HISTORY_LEN = 150
CHART_REDRAW_EVERY_N_TICKS = 5

WIREFRAME_BLUE = (70, 140, 255)
ACCENT_HEX = "#4692ff"
BG_HEX = "#1e1f24"
CARD_BG_HEX = "#26272e"
ROW_ALT_HEX = "#2d2f37"


def build_placeholder_image(w, h):
    """Simple retro blue wireframe: front-view squat figure with a bar
    across the shoulders. Reserves the video label's size before Start
    is pressed, and shown as the idle state."""
    img = Image.new("RGB", (w, h), (10, 12, 16))
    draw = ImageDraw.Draw(img)
    lw = 6
    cx = w // 2

    head_r = 34
    head_cy = int(h * 0.17)
    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
                 outline=WIREFRAME_BLUE, width=lw)

    shoulder_y = head_cy + head_r + 26
    bar_half = 130
    draw.line([cx, head_cy + head_r, cx, shoulder_y], fill=WIREFRAME_BLUE, width=lw)
    draw.line([cx - bar_half, shoulder_y, cx + bar_half, shoulder_y], fill=WIREFRAME_BLUE, width=lw + 2)
    plate_r = 20
    draw.ellipse([cx - bar_half - plate_r, shoulder_y - plate_r, cx - bar_half + plate_r, shoulder_y + plate_r],
                 outline=WIREFRAME_BLUE, width=lw)
    draw.ellipse([cx + bar_half - plate_r, shoulder_y - plate_r, cx + bar_half + plate_r, shoulder_y + plate_r],
                 outline=WIREFRAME_BLUE, width=lw)
    draw.line([cx - bar_half, shoulder_y, cx - bar_half + 15, shoulder_y + 45], fill=WIREFRAME_BLUE, width=lw)
    draw.line([cx + bar_half, shoulder_y, cx + bar_half - 15, shoulder_y + 45], fill=WIREFRAME_BLUE, width=lw)

    hip_y = shoulder_y + 110
    draw.line([cx, shoulder_y, cx, hip_y], fill=WIREFRAME_BLUE, width=lw)
    hip_half = 32
    draw.line([cx - hip_half, hip_y, cx + hip_half, hip_y], fill=WIREFRAME_BLUE, width=lw)

    knee_y = hip_y + 80
    foot_y = knee_y + 90
    knee_half = 55
    foot_half = 60

    draw.line([cx - hip_half, hip_y, cx - knee_half, knee_y], fill=WIREFRAME_BLUE, width=lw)
    draw.line([cx - knee_half, knee_y, cx - foot_half, foot_y], fill=WIREFRAME_BLUE, width=lw)
    draw.line([cx - foot_half - 20, foot_y, cx - foot_half + 20, foot_y], fill=WIREFRAME_BLUE, width=lw)

    draw.line([cx + hip_half, hip_y, cx + knee_half, knee_y], fill=WIREFRAME_BLUE, width=lw)
    draw.line([cx + knee_half, knee_y, cx + foot_half, foot_y], fill=WIREFRAME_BLUE, width=lw)
    draw.line([cx + foot_half - 20, foot_y, cx + foot_half + 20, foot_y], fill=WIREFRAME_BLUE, width=lw)

    bracket = 30
    for bx, by, dx, dy in [(10, 10, 1, 1), (w - 10, 10, -1, 1), (10, h - 10, 1, -1), (w - 10, h - 10, -1, -1)]:
        draw.line([bx, by, bx + bracket * dx, by], fill=WIREFRAME_BLUE, width=3)
        draw.line([bx, by, bx, by + bracket * dy], fill=WIREFRAME_BLUE, width=3)

    return img


def card(parent, **kwargs):
    """A tk.Frame styled as a subtle bordered card, matching the accent color."""
    defaults = dict(bg=CARD_BG_HEX, highlightbackground=ACCENT_HEX, highlightthickness=1, bd=0)
    defaults.update(kwargs)
    return tk.Frame(parent, **defaults)


class SquatCoachApp:
    def __init__(self, root):
        self.root = root
        root.title("Squat Coach")
        root.configure(bg=BG_HEX)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.pose = mp_pose.Pose()
        self.imu = IMUListener()
        self.imu.start()

        self.ema = EMA(alpha=EMA_ALPHA)
        self.rep_counter = RepCounter()
        self.completed_reps = []
        self.imu_samples = []
        self.last_completed_rep = None
        self.last_rep_timestamp = time.time()
        self.running = False
        self.frames_processed = 0
        self.session_reps_total = 0
        self.angle_history = deque(maxlen=CHART_HISTORY_LEN)
        self.session_rep_log = []  # every rep across the whole session, for the table

        self._placeholder_imgtk = ImageTk.PhotoImage(image=build_placeholder_image(VIDEO_W, VIDEO_H))

        self._build_ui()
        self._tick()

    def _build_ui(self):
        main_frame = tk.Frame(self.root, bg=BG_HEX)
        main_frame.pack(fill="both", expand=False, padx=14, pady=14)

        # ---- Left column: video feed + live chart + rep log table ----
        left = tk.Frame(main_frame, bg=BG_HEX, width=VIDEO_W)
        left.grid(row=0, column=0, sticky="n", padx=(0, 16))
        left.grid_propagate(False)

        self.video_label = tk.Label(left, bg="black", width=VIDEO_W, height=VIDEO_H,
                                     image=self._placeholder_imgtk)
        self.video_label.image = self._placeholder_imgtk
        self.video_label.pack()

        chart_card = card(left)
        chart_card.pack(fill="x", pady=(10, 0))

        self.fig = Figure(figsize=(VIDEO_W / 100, 1.7), dpi=100)
        self.fig.patch.set_facecolor(CARD_BG_HEX)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(CARD_BG_HEX)
        self.ax.set_title("Knee angle (live)", color="white", fontsize=10, pad=6)
        self.ax.tick_params(colors="#9aa0ab", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#3a3c45")
        self.ax.axhline(160, color="#4ade80", linewidth=0.8, linestyle="--", alpha=0.5)
        self.ax.axhline(100, color="#e05a5a", linewidth=0.8, linestyle="--", alpha=0.5)
        (self.angle_line,) = self.ax.plot([], [], color=ACCENT_HEX, linewidth=2)
        self.ax.set_ylim(0, 190)
        self.fig.tight_layout(pad=1.2)

        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=chart_card)
        self.chart_canvas.get_tk_widget().pack()
        self.chart_canvas.draw()

        # ---- Rep log table ----
        tk.Label(left, text="REP LOG", bg=BG_HEX, fg="#9aa0ab",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 4))

        log_card = card(left)
        log_card.pack(fill="both")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Log.Treeview", background=CARD_BG_HEX, fieldbackground=CARD_BG_HEX,
                         foreground="white", rowheight=26, borderwidth=0, font=("Segoe UI", 10))
        style.configure("Log.Treeview.Heading", background="#33353d", foreground="#9aa0ab",
                         font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("Log.Treeview", background=[("selected", "#3a3d47")])

        columns = ("rep", "score", "depth", "accel", "warnings")
        self.rep_table = ttk.Treeview(log_card, columns=columns, show="headings", height=9,
                                       style="Log.Treeview")
        self.rep_table.heading("rep", text="Rep")
        self.rep_table.heading("score", text="Score")
        self.rep_table.heading("depth", text="Min Angle")
        self.rep_table.heading("accel", text="Accel Std")
        self.rep_table.heading("warnings", text="Warnings")
        self.rep_table.column("rep", width=40, anchor="center")
        self.rep_table.column("score", width=55, anchor="center")
        self.rep_table.column("depth", width=80, anchor="center")
        self.rep_table.column("accel", width=75, anchor="center")
        self.rep_table.column("warnings", width=280, anchor="w")

        self.rep_table.tag_configure("good", foreground="#4ade80")
        self.rep_table.tag_configure("mid", foreground="#facc15")
        self.rep_table.tag_configure("bad", foreground="#e05a5a")

        scrollbar = ttk.Scrollbar(log_card, orient="vertical", command=self.rep_table.yview)
        self.rep_table.configure(yscrollcommand=scrollbar.set)
        self.rep_table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ---- Right column: controls / stats / context / coaching ----
        side = tk.Frame(main_frame, bg=BG_HEX, width=340)
        side.grid(row=0, column=1, sticky="n")
        side.grid_propagate(False)

        tk.Label(side, text="SQUAT COACH", bg=BG_HEX, fg=ACCENT_HEX,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        self.start_stop_btn = tk.Button(side, text="Start", width=18, command=self.toggle_running,
                                         bg="#3b82f6", fg="white", font=("Segoe UI", 12, "bold"),
                                         relief="flat", activebackground="#2563eb", activeforeground="white",
                                         cursor="hand2")
        self.start_stop_btn.pack(pady=(0, 14), ipady=4)

        stats_card = card(side)
        stats_card.pack(fill="x", pady=(0, 14))
        tk.Label(stats_card, text="STATS", bg=CARD_BG_HEX, fg="#9aa0ab",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(10, 4))

        self.angle_var = tk.StringVar(value="Knee angle: --")
        self.profile_var = tk.StringVar(value="TURN SIDE-ON")
        self.rep_var = tk.StringVar(value=f"Rep: 0 / {SET_SIZE}")
        self.score_var = tk.StringVar(value="Last score: --")

        self.angle_label = tk.Label(stats_card, textvariable=self.angle_var, bg=CARD_BG_HEX, fg="white",
                                     font=("Segoe UI", 13), anchor="w")
        self.angle_label.pack(fill="x", padx=12, pady=3)
        self.profile_label = tk.Label(stats_card, textvariable=self.profile_var, bg=CARD_BG_HEX, fg="#e05a5a",
                                       font=("Segoe UI", 12, "bold"), anchor="w")
        self.profile_label.pack(fill="x", padx=12, pady=3)
        self.rep_label = tk.Label(stats_card, textvariable=self.rep_var, bg=CARD_BG_HEX, fg="white",
                                   font=("Segoe UI", 12), anchor="w")
        self.rep_label.pack(fill="x", padx=12, pady=3)
        self.score_label = tk.Label(stats_card, textvariable=self.score_var, bg=CARD_BG_HEX, fg="white",
                                     font=("Segoe UI", 13, "bold"), anchor="w")
        self.score_label.pack(fill="x", padx=12, pady=(3, 12))

        tk.Label(side, text="CONTEXT FOR YOUR COACH", bg=BG_HEX, fg="#9aa0ab",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))
        self.context_entry = tk.Entry(side, width=34, font=("Segoe UI", 11), bg="white",
                                       relief="flat", highlightthickness=1, highlightbackground=ACCENT_HEX)
        self.context_entry.pack(ipady=4, pady=(0, 14))

        tk.Label(side, text="COACH FEEDBACK", bg=BG_HEX, fg="#9aa0ab",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))
        feedback_card = card(side)
        feedback_card.pack(fill="both")
        self.coach_text = scrolledtext.ScrolledText(feedback_card, width=36, height=10, wrap="word",
                                                      font=("Segoe UI", 10), bg=CARD_BG_HEX, fg="white",
                                                      relief="flat", padx=10, pady=10, bd=0,
                                                      highlightthickness=0)
        self.coach_text.pack()
        self.coach_text.insert("end", "Feedback will appear here after your first set.")
        self.coach_text.configure(state="disabled")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def toggle_running(self):
        self.running = not self.running
        self.start_stop_btn.config(text="Stop" if self.running else "Start",
                                    bg="#ef4444" if self.running else "#3b82f6")
        if not self.running:
            self.video_label.configure(image=self._placeholder_imgtk)
            self.video_label.image = self._placeholder_imgtk

    def _set_coach_text(self, text):
        self.coach_text.configure(state="normal")
        self.coach_text.delete("1.0", "end")
        self.coach_text.insert("end", text)
        self.coach_text.configure(state="disabled")

    def _redraw_chart(self):
        if not self.angle_history:
            return
        ys = list(self.angle_history)
        xs = list(range(len(ys)))
        self.angle_line.set_data(xs, ys)
        self.ax.set_xlim(0, max(CHART_HISTORY_LEN, len(ys)))
        self.chart_canvas.draw_idle()

    def _log_rep(self, rep):
        """Append a completed rep to the on-screen table, newest at top."""
        self.session_rep_log.append(rep)
        score = rep["score"]
        tag = "good" if score >= 85 else "mid" if score >= 60 else "bad"
        accel = f"{rep['accel_std']:.2f}" if "accel_std" in rep else "--"
        warnings_str = ", ".join(rep["warnings"]) if rep["warnings"] else "—"
        self.rep_table.insert("", 0, values=(
            rep["rep_number"], score, f"{rep['min_angle']:.0f}°", accel, warnings_str
        ), tags=(tag,))

    def _tick(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frames_processed += 1
                now = time.time()
                results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    side, angle, _vis = knee_angle_facing_camera(landmarks)
                    in_profile = profile_score(landmarks) < PROFILE_THRESHOLD
                    smoothed_angle = self.ema.update(angle)
                    self.angle_history.append(smoothed_angle)

                    accel_mag = None
                    if self.imu.is_connected():
                        packet = self.imu.get_latest()
                        accel_mag = math.sqrt(packet["ax"] ** 2 + packet["ay"] ** 2 + packet["az"] ** 2)

                    rep = self.rep_counter.update(smoothed_angle, in_profile, now)
                    if rep is not None:
                        if len(self.imu_samples) >= 2:
                            rep["accel_std"] = statistics.stdev(self.imu_samples)
                        self.imu_samples.clear()
                        self.completed_reps.append(rep)
                        self.last_completed_rep = rep
                        self.last_rep_timestamp = now
                        self.session_reps_total += 1
                        self._log_rep(rep)
                    elif self.rep_counter.state == "STANDING":
                        self.imu_samples.clear()
                    elif accel_mag is not None:
                        self.imu_samples.append(accel_mag)

                    for i in range(FIRST_BODY_LANDMARK):
                        landmarks[i].visibility = 0
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, BODY_CONNECTIONS)

                    self.angle_var.set(f"{side} knee: {smoothed_angle:.0f}")
                    self.profile_var.set("IN PROFILE" if in_profile else "TURN SIDE-ON")
                    self.profile_label.config(fg="#4ade80" if in_profile else "#e05a5a")
                    self.rep_var.set(f"Rep: {len(self.completed_reps)} / {SET_SIZE}")
                    if self.last_completed_rep:
                        s = self.last_completed_rep["score"]
                        self.score_var.set(f"Last score: {s}")
                        self.score_label.config(fg="#4ade80" if s >= 85 else "#facc15" if s >= 60 else "#e05a5a")

                    idle_too_long = self.completed_reps and (now - self.last_rep_timestamp) > SET_IDLE_TIMEOUT_S
                    if len(self.completed_reps) >= SET_SIZE or idle_too_long:
                        self._set_coach_text("Generating coaching feedback...")
                        self.root.update()
                        user_context = self.context_entry.get().strip()
                        coaching_text = get_coaching(self.completed_reps, user_context)
                        self._set_coach_text(coaching_text)
                        self.completed_reps = []
                        self.rep_counter = RepCounter()
                        self.last_rep_timestamp = now

                if self.frames_processed % CHART_REDRAW_EVERY_N_TICKS == 0:
                    self._redraw_chart()

                display_frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
                img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(15, self._tick)

    def _on_close(self):
        self.running = False
        self.cap.release()
        self.pose.close()
        self.imu.stop()
        print(f"Frames processed: {self.frames_processed} | Reps completed: {self.session_reps_total}")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SquatCoachApp(root)
    root.mainloop()