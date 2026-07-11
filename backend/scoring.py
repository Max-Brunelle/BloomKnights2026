"""Rep-counting state machine and scoring rubric for squats."""

STANDING_ANGLE_THRESHOLD = 160
BOTTOM_ANGLE_THRESHOLD = 100
FULL_DEPTH_ANGLE = 90
MAX_DEPTH_PENALTY = 50
TEMPO_PENALTY = 30
MIN_DESCENT_DURATION_S = 0.5
DEPTH_WARNING_THRESHOLD = 15


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
                self.state = "STANDING"  # aborted before reaching bottom
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
                self.state = "BOTTOM"  # sank back down, still the same rep
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
        warnings = []

        if min_angle > FULL_DEPTH_ANGLE:
            depth_penalty = min(
                MAX_DEPTH_PENALTY,
                MAX_DEPTH_PENALTY * (min_angle - FULL_DEPTH_ANGLE) / (BOTTOM_ANGLE_THRESHOLD - FULL_DEPTH_ANGLE),
            )
        else:
            depth_penalty = 0
        if depth_penalty > DEPTH_WARNING_THRESHOLD:
            warnings.append("Didn't reach full depth")

        tempo_penalty = 0
        if descent_duration_s < max(MIN_DESCENT_DURATION_S, ascent_duration_s / 2):
            tempo_penalty = TEMPO_PENALTY
            warnings.append("Descent too fast")

        score = max(0, min(100, 100 - depth_penalty - tempo_penalty))
        return int(round(score)), warnings


if __name__ == "__main__":
    counter = RepCounter()
    angles = [175, 170, 140, 100, 85, 100, 140, 175]
    completed = [r for i, a in enumerate(angles) if (r := counter.update(a, True, float(i))) is not None]
    assert len(completed) == 1
    assert completed[0]["score"] >= 90
    print("scoring.py self-check OK")
