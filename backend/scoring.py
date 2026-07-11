"""Rep-counting state machine and scoring rubric for squats."""

STANDING_ANGLE_THRESHOLD = 160
BOTTOM_ANGLE_THRESHOLD = 100
DEPTH_FULL_CREDIT_ANGLE = 90
DEPTH_ZERO_CREDIT_ANGLE = 100
MAX_DEPTH_PENALTY = 50
DEPTH_WARNING_THRESHOLD = 15

MIN_DESCENT_S = 0.5
TEMPO_MAX_PENALTY = 30
TEMPO_WARNING_THRESHOLD = 10


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

        # Tempo penalty - gradual based on how far under the threshold the descent was
        tempo_threshold = max(MIN_DESCENT_S, ascent_duration_s / 2)
        if descent_duration_s >= tempo_threshold:
            tempo_penalty = 0
        else:
            deficit_ratio = (tempo_threshold - descent_duration_s) / tempo_threshold
            tempo_penalty = TEMPO_MAX_PENALTY * min(1.0, deficit_ratio)
        if tempo_penalty > TEMPO_WARNING_THRESHOLD:
            warnings.append("Descent too fast")

        score = max(0, min(100, round(100 - depth_penalty - tempo_penalty)))
        return score, warnings


if __name__ == "__main__":
    counter = RepCounter()
    angles = [175, 170, 140, 100, 85, 100, 140, 175]
    completed = [r for i, a in enumerate(angles) if (r := counter.update(a, True, float(i))) is not None]
    assert len(completed) == 1
    assert completed[0]["score"] >= 90
    print("scoring.py self-check OK")