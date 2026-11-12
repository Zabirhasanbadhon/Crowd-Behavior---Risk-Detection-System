# detector.py
import os
import time
import threading
import collections
import cv2
import numpy as np
from ultralytics import YOLO

from utils import (
    calculate_distance, play_alarm_async, log_alert,
    save_abnormal_frame, save_heatmap_image,
)

# ---------- Configuration ----------
COLORS = {
    "person":      {"normal": (0, 255, 0),    "abnormal": (0, 0, 255)},
    "walking":     {"normal": (0, 255, 0),    "abnormal": (0, 165, 255)},
    "standing":    {"normal": (255, 128, 0),  "abnormal": (0, 0, 255)},
    "running":     {"normal": (255, 255, 0),  "abnormal": (0, 0, 255)},
    "fighting":    {"normal": (128, 0, 128),  "abnormal": (0, 0, 255)},
    "jumping":     {"normal": (0, 255, 255),  "abnormal": (0, 0, 255)},
    "robbery":     {"normal": (255, 165, 0),  "abnormal": (0, 0, 255)},
    "sitting":     {"normal": (0, 128, 255),  "abnormal": (0, 0, 255)},
    "armed":       {"normal": (128, 128, 128),"abnormal": (0, 0, 255)},
    "lying_down":  {"normal": (0, 255, 128),  "abnormal": (0, 0, 255)},
    "violence":    {"normal": (0, 255, 128),  "abnormal": (0, 0, 255)},
    "weapon":      {"normal": (0, 255, 128),  "abnormal": (0, 0, 255)},
    "normal":      {"normal": (200, 200, 200), "abnormal": (0, 0, 255)},
}
DEFAULT_COLOR = {"normal": (200, 200, 200), "abnormal": (0, 0, 255)}

# Custom model classes that should ALWAYS trigger an alarm
ALWAYS_ABNORMAL = {"fighting", "robbery", "armed", "violence", "weapon", "lying_down"}

ACTIVITY_DENSITY_THRESHOLD = {
    "running": 0.15, "fighting": 0.10, "robbery": 0.05, "armed": 0.05,
}

ALERT_SETTINGS = {
    "fighting": {"freq": 3000, "dur": 1200},
    "robbery":  {"freq": 3500, "dur": 1500},
    "armed":    {"freq": 4000, "dur": 1500},
    "violence": {"freq": 3500, "dur": 1500},
    "weapon":   {"freq": 4000, "dur": 1500},
}

ALERT_COOLDOWN = 3.0
SUDDEN_MOVE_PIXELS = 80
GATHERING_MIN_PEOPLE = 3
GATHERING_DIST_PX = 100
CROWD_DENSITY_HIGH = 0.40
FRAME_SKIP = 2
HEATMAP_SAVE_INTERVAL = 30

class CrowdDetector:
    def __init__(self, model_path="best.pt", conf=0.35, iou=0.45):
        # 1. Load the standard Person Tracker (this counts the crowd)
        self.person_model = YOLO("yolov8n.pt") 
        
        # 2. Load your Custom Crime Model (this detects violence/weapons)
        self.crime_model = YOLO(model_path)
        
        self.conf = conf
        self.iou = iou

        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )
        self.heatmap = None
        self.frame_count = 0
        self.prev_gray = None
        self.prev_track_centers = {}
        self.standing_start = {}
        self.last_alert_time = {}
        self.last_saved_frame = {}
        self.trajectories = collections.defaultdict(list)
        self.last_heatmap_time = time.time()

    def _analyze_crowd(self, frame):
        mask = self.back_sub.apply(frame)
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        h, w = frame.shape[:2]
        density = cv2.countNonZero(mask) / float(h * w)
        return mask, density

    def _update_heatmap(self, mask):
        if self.heatmap is None:
            self.heatmap = np.zeros_like(mask, dtype=np.float32)
        self.heatmap += mask.astype(np.float32)
        self.heatmap *= 0.999

    def _maybe_alert(self, activity):
        if activity not in ALERT_SETTINGS:
            return
        now = time.time()
        if now - self.last_alert_time.get(activity, 0) < ALERT_COOLDOWN:
            return
        self.last_alert_time[activity] = now
        cfg = ALERT_SETTINGS[activity]
        play_alarm_async(cfg["freq"], cfg["dur"])
        log_alert(f"ALERT: Abnormal '{activity}' detected")

    def _maybe_save_frame(self, frame, activity):
        now = time.time()
        if now - self.last_saved_frame.get(activity, 0) < 2.0:
            return
        self.last_saved_frame[activity] = now
        save_abnormal_frame(frame, activity)

    def process_frame(self, frame):
        self.frame_count += 1
        annotated = frame.copy()

        mask, density = self._analyze_crowd(frame)
        self._update_heatmap(mask)

        if time.time() - self.last_heatmap_time >= HEATMAP_SAVE_INTERVAL:
            save_heatmap_image(self.heatmap)
            self.last_heatmap_time = time.time()

        stats = {
            "density": density,
            "humans": 0,
            "abnormal": False,
            "activities": [],
            "alerts": [],
        }

        if self.frame_count % FRAME_SKIP != 0:
            return annotated, stats

        # ==========================================
        # PASS 1: Detect and Track People (Crowd)
        # ==========================================
        person_results = self.person_model.track(
            frame, persist=True, conf=0.3, iou=0.5, 
            tracker="bytetrack.yaml", classes=[0], verbose=False
        )

        cur_centers = {}
        person_detections = []

        if person_results and person_results[0].boxes is not None and len(person_results[0].boxes) > 0:
            boxes = person_results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tid = int(ids[i]) if ids is not None else i
                cur_centers[tid] = (cx, cy)
                person_detections.append((tid, x1, y1, x2, y2, (cx, cy)))

        # ==========================================
        # PASS 2: Detect Crimes (Fighting, Weapons, etc)
        # ==========================================
        crime_results = self.crime_model.predict(
            frame, conf=self.conf, iou=self.iou, verbose=False
        )
        
        crime_detections = []
        if crime_results and crime_results[0].boxes is not None:
            for box in crime_results[0].boxes:
                cls_id = int(box.cls[0])
                label = self.crime_model.names.get(cls_id, "unknown")
                # Ignore "normal" or "NonViolence" predictions from the crime model
                if "normal" in label.lower() or "non" in label.lower():
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crime_detections.append((label, x1, y1, x2, y2))

        # ==========================================
        # Combine and Analyze
        # ==========================================
        sudden_move = False
        for tid, _, _, _, _, (cx, cy) in person_detections:
            if tid in self.prev_track_centers:
                if calculate_distance((cx, cy), self.prev_track_centers[tid]) > SUDDEN_MOVE_PIXELS:
                    sudden_move = True
                    break

        centers = [d[5] for d in person_detections]
        gathering_ids = set()
        for i in range(len(centers)):
            group = [i]
            for j in range(len(centers)):
                if i != j and calculate_distance(centers[i], centers[j]) < GATHERING_DIST_PX:
                    group.append(j)
            if len(group) >= GATHERING_MIN_PEOPLE:
                gathering_ids.update(group)

        now = time.time()
        active_ids = set()

        # Draw People
        for idx, (tid, x1, y1, x2, y2, center) in enumerate(person_detections):
            active_ids.add(tid)
            self.trajectories[tid].append(center)
            if len(self.trajectories[tid]) > 60:
                self.trajectories[tid].pop(0)

            is_gathering_member = idx in gathering_ids
            is_abn = is_gathering_member or density > CROWD_DENSITY_HIGH or sudden_move
            
            palette = COLORS.get("person", DEFAULT_COLOR)
            color = palette["abnormal"] if is_abn else palette["normal"]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            tag = "ABNORMAL" if is_abn else "Normal"
            text = f"ID{tid} Person [{tag}]"
            cv2.putText(annotated, text, (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            if is_abn:
                stats["abnormal"] = True
                stats["activities"].append(("person", True))

            # Draw trajectory
            pts = self.trajectories[tid]
            for k in range(1, len(pts)):
                cv2.line(annotated, pts[k - 1], pts[k], color, 1, cv2.LINE_AA)

        # Draw Crimes (if any)
        for label, x1, y1, x2, y2 in crime_detections:
            is_abn = True # Any crime is abnormal
            palette = COLORS.get(label, DEFAULT_COLOR)
            color = palette["abnormal"]
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            text = f"CRIME: {label.upper()}"
            cv2.putText(annotated, text, (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            stats["abnormal"] = True
            stats["alerts"].append(label)
            self._maybe_alert(label)
            self._maybe_save_frame(annotated, label)

        # Cleanup
        for tid in list(self.trajectories.keys()):
            if tid not in active_ids:
                self.trajectories[tid].pop(0)
                if not self.trajectories[tid]:
                    del self.trajectories[tid]

        self.prev_track_centers = cur_centers
        stats["humans"] = len(person_detections)

        # HUD
        cv2.putText(annotated, f"Density: {density:.2f}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Humans:  {stats['humans']}", (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        overall = "ABNORMAL" if stats["abnormal"] else "NORMAL"
        oc = (0, 0, 255) if stats["abnormal"] else (0, 255, 0)
        cv2.putText(annotated, overall, (12, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, oc, 2, cv2.LINE_AA)

        return annotated, stats

    def finalize(self):
        save_heatmap_image(self.heatmap)