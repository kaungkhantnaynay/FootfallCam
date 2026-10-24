"""Conservative local tracker using motion, overlap and appearance association.

This is a small custom tracker, not an implementation of ByteTrack.
Ambiguous associations break identity instead of inheriting a staff flag.
"""
from dataclasses import dataclass, field
from collections import deque
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import centre, iou


def appearance(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=int)
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return np.zeros(32, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [1, 2], None, [4, 8], [0, 256, 0, 256]).ravel()
    return hist / max(float(hist.sum()), 1)


@dataclass
class Track:
    id: int
    box: np.ndarray
    last_frame: int
    histogram: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hit_frames: deque = field(default_factory=deque)
    observations: list = field(default_factory=list)


class Tracker:
    def __init__(self, fps, config):
        self.fps, self.config = fps, config
        self.active = {}
        self.finished = []
        self.next_id = 1

    def update(self, frame_index, frame, detections):
        expired = [tid for tid, t in self.active.items()
                   if frame_index - t.last_frame > self.config.max_track_gap_s * self.fps]
        for tid in expired:
            self.finished.append(self.active.pop(tid))
        tracks = list(self.active.values())
        histograms = [appearance(frame, d[0]) for d in detections]
        costs = np.full((len(tracks), len(detections)), 1e6)
        for i, track in enumerate(tracks):
            predicted = track.box + track.velocity * (frame_index - track.last_frame)
            scale = max(30, np.linalg.norm(track.box[2:] - track.box[:2]))
            for j, (box, _) in enumerate(detections):
                distance = np.linalg.norm(np.array(centre(predicted)) - centre(box)) / scale
                overlap = iou(predicted, box)
                app = float(cv2.compareHist(track.histogram, histograms[j], cv2.HISTCMP_BHATTACHARYYA))
                if distance < 0.65 and (overlap > 0.05 or distance < 0.25) and app < 0.70:
                    costs[i, j] = 0.5 * (1 - overlap) + 0.3 * distance + 0.2 * app
        matched = {}
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if costs[i, j] >= 0.85:
                    continue
                # Near-tied identities are deliberately not propagated.
                alternatives = np.delete(costs[:, j], i)
                if alternatives.size and alternatives.min() - costs[i, j] < 0.04:
                    continue
                matched[j] = tracks[i]
        result = []
        for j, (box, confidence) in enumerate(detections):
            track = matched.get(j)
            if track is None:
                track = Track(self.next_id, np.asarray(box).copy(), frame_index, histograms[j])
                self.active[track.id] = track
                self.next_id += 1
            else:
                delta = (box - track.box) / max(1, frame_index - track.last_frame)
                track.velocity = 0.5 * track.velocity + 0.5 * delta
                track.box, track.last_frame = np.asarray(box).copy(), frame_index
                track.histogram = 0.8 * track.histogram + 0.2 * histograms[j]
            result.append((track, confidence))
        return result

def staff_intervals(hit_frames, fps, config, backfill=False):
    """Confirm clustered evidence; extend only within bounded time windows.

    Returns inclusive frame intervals. Repeated observations in ONE frame never
    count as multiple hits. No evidence can label an entire video indefinitely.
    """
    hits = sorted(set(hit_frames))
    intervals = []
    window = deque()
    confirmed_until = -1
    for frame in hits:
        window.append(frame)
        while window and frame - window[0] > config.confirm_window_s * fps:
            window.popleft()
        if len(window) >= config.confirm_hits or frame <= confirmed_until:
            start = frame
            if backfill and len(window) >= config.confirm_hits:
                start = max(0, window[0] - round(config.offline_backfill_s * fps))
            end = frame + round(config.evidence_ttl_s * fps)
            confirmed_until = end
            if intervals and start <= intervals[-1][1] + 1:
                intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
            else:
                intervals.append((start, end))
    return intervals
