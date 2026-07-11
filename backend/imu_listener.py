"""Background-thread UDP listener for a wearable IMU streaming JSON packets.

Packet schema: {"t": device_millis, "ax","ay","az","gx","gy","gz": float, "batt": int}
"t" is device-uptime ms, not synced to the PC clock - never compare it to PC-side timestamps.
"""

import collections
import json
import socket
import threading
import time

# No packet in this long -> treat the wearable as disconnected.
STALE_TIMEOUT = 2.0

REQUIRED_KEYS = {"t", "ax", "ay", "az", "gx", "gy", "gz"}


class IMUListener:
    def __init__(self, port=4210, buffer_size=500):
        self.port = port
        self._buffer = collections.deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._last_packet_time = None
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(0.5)
        try:
            while not self._stop_event.is_set():
                try:
                    data, _addr = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:
                    continue

                try:
                    packet = json.loads(data)
                except (ValueError, UnicodeDecodeError):
                    continue
                if not isinstance(packet, dict) or not REQUIRED_KEYS.issubset(packet):
                    continue

                with self._lock:
                    self._buffer.append(packet)
                    self._last_packet_time = time.monotonic()
        finally:
            sock.close()

    def get_latest(self):
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_since(self, timestamp_ms):
        with self._lock:
            return [p for p in self._buffer if p["t"] > timestamp_ms]

    def is_connected(self):
        with self._lock:
            last = self._last_packet_time
        return last is not None and (time.monotonic() - last) < STALE_TIMEOUT


if __name__ == "__main__":
    listener = IMUListener()
    listener.start()
    print(f"Listening for IMU packets on UDP :{listener.port}... Ctrl+C to stop.")
    try:
        while True:
            print(listener.is_connected(), listener.get_latest())
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
