"""Image-space geometry shared by detection, association and evaluation."""
import numpy as np


def centre(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def clip_box(box, width, height):
    return np.array([np.clip(box[0], 0, width), np.clip(box[1], 0, height),
                     np.clip(box[2], 0, width), np.clip(box[3], 0, height)], dtype=float)
