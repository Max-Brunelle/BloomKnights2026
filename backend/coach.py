"""Gemini-powered squat coaching feedback, generated once per completed set."""

import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()  # picks up GEMINI_API_KEY from the environment

MODEL = "gemini-2.5-flash"

COACHING_SYSTEM_PROMPT = (
    "You are a concise squat form coach. Given structured rep data from a set, "
    "give 2-3 sentences of specific, actionable feedback. Reference specific rep "
    "numbers when calling out issues. Be direct and encouraging, not generic."
)


def get_coaching(rep_data: list[dict]) -> str:
    """Blocking call - invoked once per completed set, not per frame."""
    prompt = f"Here is the data from this set:\n{json.dumps(rep_data)}"
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=COACHING_SYSTEM_PROMPT,
            max_output_tokens=200,
            temperature=0.7,
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
