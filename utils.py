# utils.py
import os
import time
import datetime
import threading
import numpy as np

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABNORMAL_DIR = os.path.join(BASE_DIR, "abnormal_frames")
HEATMAP_DIR = os.path.join(BASE_DIR, "heatmaps")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "abnormal_activity_log.txt")

for d in (ABNORMAL_DIR, HEATMAP_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

# ---------- Distance ----------
def calculate_distance(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

# ---------- Alarm ----------
try:
    import winsound
    def _beep(freq, dur):
        winsound.Beep(int(freq), int(dur))
except Exception:
    try:
        import pygame
        pygame.mixer.init()
        def _beep(freq, dur):
            sr = 44100
            n = int(sr * dur / 1000.0)
            t = np.linspace(0, dur / 1000.0, n, False)
            wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            stereo = np.column_stack([wave, wave])
            pygame.sndarray.make_sound(stereo).play()
    except Exception:
        def _beep(freq, dur):
            pass  # silent fallback

_alarm_lock = threading.Lock()

def play_alarm_async(freq, dur):
    """Non-blocking beep so the GUI never freezes."""
    def _run():
        with _alarm_lock:
            try:
                _beep(freq, dur)
            except Exception:
                pass
    threading.Thread(target=_run, daemon=True).start()

# ---------- Logging ----------
def log_alert(message: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{ts} - {message}\n")

# ---------- Frame saving ----------
def save_abnormal_frame(frame, activity: str):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(ABNORMAL_DIR, f"abnormal_{activity}_{ts}.jpg")
    import cv2
    cv2.imwrite(path, frame)
    return path

def save_heatmap_image(heatmap, timestamp=None):
    import cv2
    if heatmap is None:
        return None
    norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(HEATMAP_DIR, f"heatmap_{timestamp}.jpg")
    cv2.imwrite(path, colored)
    return path