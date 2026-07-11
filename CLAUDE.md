@"
# BloomKnights 2026 — Exercise Form Coach (SUBMISSION REPO)

Event-day repo.

## Project
CV-based squat form scoring (MediaPipe pose -> joint angles -> rubric)
+ ESP32 IMU wearable (rep tempo/control, optional/cuttable)
+ Gemini API coaching. Output: OpenCV window, no web frontend.

## Build/test
- Activate venv: .\venv\Scripts\Activate.ps1
- Install deps: pip install -r requirements.txt
- Run: python backend\main.py

## Layout
- backend/main.py       - webcam capture, pose overlay, integration
- backend/vision.py     - angle math, EMA smoothing, profile-score gating
- backend/scoring.py    - rep state machine + rubric scoring
- backend/imu_listener.py - UDP IMU ingest, own thread, is_connected() gate
- backend/coach.py      - Gemini API coaching call (google-genai SDK)
- firmware/wearable/    - ESP32 firmware

## Coaching API
Uses Google's unified "google-genai" SDK (NOT the deprecated
"google-generativeai" package). Model: gemini-2.5-flash. API key via
GEMINI_API_KEY environment variable (auto-picked-up by genai.Client(),
do not pass the key explicitly in code). Sponsor prize consideration:
Gemini is a BloomKnights sponsor.

## Wearable data contract (UDP JSON)
{"t":<device_millis>,"ax":..,"ay":..,"az":..,"gx":..,"gy":..,"gz":..,"batt":..}

## Conventions
- Vision must work standalone; IMU is additive, never a hard dependency
- Rule-based scoring, not ML
- Minimal code: prefer stdlib/native solutions before new dependencies
- Rep durations are wall-clock (time.time()); CSV timestamps use elapsed-
  since-start - different epochs, don't compare directly
- See STATUS.md before starting work; update it before ending a session
"@ | Out-File -FilePath AGENTS.md -Encoding utf8