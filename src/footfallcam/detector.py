"""Person boxes from pretrained YOLO; no model training is performed."""
from pathlib import Path
import cv2
import numpy as np


class PersonDetector:
    def __init__(self, config):
        # Keep model downloads as an explicit setup step.
        if not Path(config.model).is_file():
            raise FileNotFoundError(f"Missing local weights: {config.model}. Run scripts/download_model.py first.")
        from ultralytics import YOLO
        self.model = YOLO(config.model)
        self.config = config

    def detect(self, frame):
        height, width = frame.shape[:2]
        boxes, confidences = [], []
        for rotation in self.config.detector_rotations:
            rotated = np.ascontiguousarray(np.rot90(frame, -(rotation // 90)))
            result = self.model.predict(rotated, classes=[0], conf=self.config.person_confidence,
                                        imgsz=self.config.image_size, device=self.config.device,
                                        verbose=False)[0]
            for raw, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                box = inverse_rotated_box(raw, rotation, width, height)
                if min(box[2] - box[0], box[3] - box[1]) >= self.config.min_person_size:
                    boxes.append(box)
                    confidences.append(float(confidence))
        if not boxes:
            return []
        xywh = [[float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])] for b in boxes]
        keep = cv2.dnn.NMSBoxes(xywh, confidences, self.config.person_confidence, 0.45)
        return [(boxes[int(i)], confidences[int(i)]) for i in np.asarray(keep).ravel()]


def inverse_rotated_box(box, rotation, width, height):
    """Map continuous xyxy coordinates from a clockwise rotated image to original."""
    x1, y1, x2, y2 = map(float, box)
    if rotation == 0:
        return np.array([x1, y1, x2, y2])
    if rotation == 90:
        return np.array([y1, height-x2, y2, height-x1])
    if rotation == 180:
        return np.array([width-x2, height-y2, width-x1, height-y1])
    if rotation == 270:
        return np.array([width-y2, x1, width-y1, x2])
    raise ValueError("Only quarter-turn rotations are supported")
