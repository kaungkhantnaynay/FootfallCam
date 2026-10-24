"""Reference-pattern matching at multiple scales and rotations.

Scores are normalized image similarity, NOT staff probabilities.
Reference patches come from the employer's PDF, not inferred staff clothing.
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from .verification import canonical_patch, TagVerifier


@dataclass
class TagMatch:
    score: float
    box: tuple[int, int, int, int] | None
    reference: str = ""
    angle: int = 0


class TagMatcher:
    def __init__(self, reference_dir, config):
        self.config = config
        self.templates = []
        self.verifier = TagVerifier(config.verifier, config.verifier_margin) if config.verifier else None
        for path in sorted(Path(reference_dir).glob("tag-*.png")):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.std() < 2:
                raise ValueError(f"Invalid or constant reference patch: {path}")
            for width in config.tag_widths:
                height = max(4, round(image.shape[0] * width / image.shape[1]))
                small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                for angle in range(0, 360, config.rotation_step):
                    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
                    c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
                    rw, rh = int(np.ceil(width * c + height * s)), int(np.ceil(height * c + width * s))
                    matrix[0, 2] += (rw - width) / 2
                    matrix[1, 2] += (rh - height) / 2
                    rotated = cv2.warpAffine(small, matrix, (rw, rh),
                                             borderValue=int(np.median(small)))
                    if rotated.std() >= 2:
                        self.templates.append((rotated, path.stem, angle, width, height))
        if not self.templates:
            raise FileNotFoundError(f"No tag-*.png reference images in {reference_dir}")

    def match(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return TagMatch(0.0, None)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        candidates = []
        for template, reference, angle, original_width, original_height in self.templates:
            th, tw = template.shape
            if th > gray.shape[0] or tw > gray.shape[1]:
                continue
            response = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, value, _, point = cv2.minMaxLoc(response)
            if not np.isfinite(value) or value < self.config.tag_threshold:
                continue
            x, y = point
            patch = gray[y:y + th, x:x + tw]
            if patch.std() < self.config.min_tag_std:
                continue
            match = TagMatch(float(value), (x1 + x, y1 + y, x1 + x + tw, y1 + y + th), reference, angle)
            candidates.append((match, canonical_patch(patch, angle, original_width, original_height)))
        candidates.sort(key=lambda x: -x[0].score)
        if self.verifier:
            accepted = self.verifier.accepts([p for _, p in candidates])
            for (match, _), ok in zip(candidates, accepted):
                if ok:
                    return match
            return TagMatch(0.0, None)
        return candidates[0][0] if candidates else TagMatch(0.0, None)
