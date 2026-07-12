"""Gemini-powered squat coaching feedback, generated once per completed set."""

import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()  # picks up GEMINI_API_KEY from the environment

MODEL = "gemini-3.5-flash"

COACHING_SYSTEM_PROMPT = (
    "You are a concise squat form coach. Given structured rep data from a set, "
    "give 2-3 sentences of specific, actionable feedback. Reference specific "
    "rep numbers when calling out issues. If a note about wrist/arm stability "
    "is provided below the rep data, follow its instruction exactly - either "
    "reference the specific rep it names, or stay silent on stability if it "
    "says there wasn't enough motion to assess. Do not independently judge "
    "raw accel_std numbers yourself. Be direct and encouraging, not generic."
)

# Thresholds derived from real squat testing: calm/controlled reps landed
# ~0.4-1.2, deliberately shaky reps landed ~1.2-2.5+. Values below the noise
# floor mean there wasn't real motion to assess (e.g. a dry-run test).
ACCEL_STD_NOISE_FLOOR = 0.3
ACCEL_STD_NOTABLE = 1.2


def _stability_note(rep_data: list[dict]) -> str:
    """Pre-classifies wrist stability so the model doesn't have to judge
    raw magnitudes with no frame of reference."""
    reps_with_accel = [r for r in rep_data if "accel_std" in r]
    if not reps_with_accel:
        return ""

    shakiest = max(reps_with_accel, key=lambda r: r["accel_std"])
    val = shakiest["accel_std"]

    if val < ACCEL_STD_NOISE_FLOOR:
        return (
            "\n\nNote: wrist motion data this set was minimal/near-zero "
            "(below meaningful movement threshold) - do NOT comment on "
            "shakiness or instability, there wasn't enough real motion "
            "to assess it."
        )
    if val >= ACCEL_STD_NOTABLE:
        return (
            f"\n\nNote: rep {shakiest['rep_number']} had notably high "
            f"wrist instability this set (accel_std={val:.2f}), clearly "
            f"elevated vs typical controlled reps (~0.4-1.2). Reference "
            f"this rep specifically."
        )
    return ""  # normal range - say nothing, don't manufacture a warning


def get_coaching(rep_data: list[dict], user_context: str | None = None) -> str:
    """Blocking call - invoked once per completed set, not per frame."""
    prompt = (
        f"User-provided context: {user_context or 'none'}\n\n"
        f"Here is the data from this set:\n{json.dumps(rep_data)}"
        f"{_stability_note(rep_data)}"
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=COACHING_SYSTEM_PROMPT,
            max_output_tokens=200,
            temperature=0.7,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


if __name__ == "__main__":
    fake_reps = [
        {
            "rep_number": 1,
            "min_angle": 82.0,
            "duration_s": 3.1,
            "descent_duration_s": 1.4,
            "ascent_duration_s": 1.5,
            "score": 100,
            "warnings": [],
        },
        {
            "rep_number": 2,
            "min_angle": 108.0,
            "duration_s": 2.6,
            "descent_duration_s": 1.2,
            "ascent_duration_s": 1.3,
            "score": 55,
            "warnings": ["Didn't reach full depth"],
        },
        {
            "rep_number": 3,
            "min_angle": 88.0,
            "duration_s": 2.9,
            "descent_duration_s": 1.3,
            "ascent_duration_s": 1.4,
            "score": 100,
            "warnings": [],
        },
    ]
    print(get_coaching(fake_reps))
    print("---")
    # Second case: simulate a near-zero accel_std (dry-run / no real motion)
    # to confirm the noise-floor branch suppresses fabricated warnings.
    dry_run_reps = [dict(r, accel_std=0.0127) for r in fake_reps]
    print(get_coaching(dry_run_reps))