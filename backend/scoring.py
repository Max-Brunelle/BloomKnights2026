"""Rep-counting state machine and scoring rubric for squats."""

STANDING_ANGLE_THRESHOLD = 160
BOTTOM_ANGLE_THRESHOLD = 100

DEPTH_FULL_CREDIT_ANGLE = 80
DEPTH_ZERO_CREDIT_ANGLE = 100
MAX_DEPTH_PENALTY = 50
DEPTH_WARNING_THRESHOLD = 15

MIN_DESCENT_S = 0.5
DESCENT_MAX_PENALTY = 25
DESCENT_WARNING_THRESHOLD = 8

MIN_ASCENT_S = 0.6
ASCENT_MAX_PENALTY = 25
ASCENT_WARNING_THRESHOLD = 8

# Precision layer: rewards going beyond the bare minimum, so reps that
# already clear full credit still differentiate from each other instead
# of all flattening to a tied 100.
DEPTH_IDEAL_ANGLE = 65      # depth this good or better earns full precision credit
DEPTH_PRECISION_MAX = 6

DESCENT_IDEAL_S = 2.0       # a nicely controlled 2-second negative
DESCENT_PRECISION_MAX = 4


class RepCounter:
    """Tracks squat state (STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING) across frames."""

    def __init__(self):
        self.state = "STANDING"
        self.rep_number = 1
        self.min_angle = None
        self.t_standing_exit = None
        self.t_bottom_enter = None
        self.t_bottom_exit = None

    def update(self, angle, in_profile, timestamp):
        """Feed one frame's knee angle. Returns a completed-rep dict, or None."""
        if not in_profile:
            return None

        if self.state == "STANDING":
            if angle <= STANDING_ANGLE_THRESHOLD:
                self.state = "DESCENDING"
                self.t_standing_exit = timestamp
                self.min_angle = angle

        elif self.state == "DESCENDING":
            if angle > STANDING_ANGLE_THRESHOLD:
                self.state = "STANDING"
            elif angle < BOTTOM_ANGLE_THRESHOLD:
                self.state = "BOTTOM"
                self.t_bottom_enter = timestamp
                self.min_angle = angle

        elif self.state == "BOTTOM":
            self.min_angle = min(self.min_angle, angle)
            if angle >= BOTTOM_ANGLE_THRESHOLD:
                self.state = "ASCENDING"
                self.t_bottom_exit = timestamp

        elif self.state == "ASCENDING":
            if angle < BOTTOM_ANGLE_THRESHOLD:
                self.state = "BOTTOM"
                self.min_angle = min(self.min_angle, angle)
            elif angle > STANDING_ANGLE_THRESHOLD:
                descent_duration_s = self.t_bottom_enter - self.t_standing_exit
                ascent_duration_s = timestamp - self.t_bottom_exit
                score, warnings = self.score_rep(self.min_angle, descent_duration_s, ascent_duration_s)
                rep = {
                    "rep_number": self.rep_number,
                    "min_angle": self.min_angle,
                    "duration_s": timestamp - self.t_standing_exit,
                    "descent_duration_s": descent_duration_s,
                    "ascent_duration_s": ascent_duration_s,
                    "score": score,
                    "warnings": warnings,
                }
                self.rep_number += 1
                self.state = "STANDING"
                self.min_angle = None
                return rep

        return None

    def score_rep(self, min_angle, descent_duration_s, ascent_duration_s):
        """Returns (score: int, warnings: list[str]) for a completed rep."""
        warnings = []

        # Depth penalty - gradual between DEPTH_FULL_CREDIT_ANGLE and DEPTH_ZERO_CREDIT_ANGLE
        if min_angle > DEPTH_FULL_CREDIT_ANGLE:
            depth_penalty = min(
                MAX_DEPTH_PENALTY,
                MAX_DEPTH_PENALTY * (min_angle - DEPTH_FULL_CREDIT_ANGLE)
                / (DEPTH_ZERO_CREDIT_ANGLE - DEPTH_FULL_CREDIT_ANGLE),
            )
        else:
            depth_penalty = 0
        if depth_penalty > DEPTH_WARNING_THRESHOLD:
            warnings.append("Didn't reach full depth")

        # Descent tempo penalty - gradual based on how far under the threshold the descent was
        descent_threshold = max(MIN_DESCENT_S, ascent_duration_s / 2)
        if descent_duration_s >= descent_threshold:
            descent_penalty = 0
        else:
            deficit_ratio = (descent_threshold - descent_duration_s) / descent_threshold
            descent_penalty = DESCENT_MAX_PENALTY * min(1.0, deficit_ratio)
        if descent_penalty > DESCENT_WARNING_THRESHOLD:
            warnings.append("Descent too fast")

        # Ascent tempo penalty
        if ascent_duration_s >= MIN_ASCENT_S:
            ascent_penalty = 0
        else:
            deficit_ratio = (MIN_ASCENT_S - ascent_duration_s) / MIN_ASCENT_S
            ascent_penalty = ASCENT_MAX_PENALTY * min(1.0, deficit_ratio)
        if ascent_penalty > ASCENT_WARNING_THRESHOLD:
            warnings.append("Stood up too fast - drive up with control")

        # Precision layer - small continuous differentiation even among
        # reps that already clear full credit above. Clamping min_angle at
        # DEPTH_FULL_CREDIT_ANGLE means a shallow rep isn't double-penalized
        # here; it just doesn't earn the extra precision credit either.
        depth_for_precision = min(min_angle, DEPTH_FULL_CREDIT_ANGLE)
        if depth_for_precision > DEPTH_IDEAL_ANGLE:
            depth_precision_penalty = DEPTH_PRECISION_MAX * (
                (depth_for_precision - DEPTH_IDEAL_ANGLE)
                / (DEPTH_FULL_CREDIT_ANGLE - DEPTH_IDEAL_ANGLE)
            )
        else:
            depth_precision_penalty = 0

        if descent_duration_s < DESCENT_IDEAL_S:
            descent_for_precision = max(descent_duration_s, MIN_DESCENT_S)
            descent_precision_penalty = DESCENT_PRECISION_MAX * (
                (DESCENT_IDEAL_S - descent_for_precision)
                / (DESCENT_IDEAL_S - MIN_DESCENT_S)
            )
        else:
            descent_precision_penalty = 0

        score = max(0, min(100, round(
            100 - depth_penalty - descent_penalty - ascent_penalty
            - depth_precision_penalty - descent_precision_penalty
        )))
        return score, warnings


if __name__ == "__main__":
    counter = RepCounter()
    angles = [175, 170, 140, 100, 75, 100, 140, 175]
    timestamps = [0, 0.3, 0.6, 1.0, 1.7, 2.3, 2.9, 3.6]
    completed = [
        r for a, t in zip(angles, timestamps)
        if (r := counter.update(a, True, t)) is not None
    ]
    assert len(completed) == 1
    assert completed[0]["score"] >= 85
    print("scoring.py self-check OK")