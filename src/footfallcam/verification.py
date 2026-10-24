"""Small visual patch verifier trained from disclosed reference augmentation.

It scores the pattern AFTER a template proposes a position. No person identity,
shirt colour, timestamp, or location is an input feature.
"""
import cv2
import numpy as np
import joblib


def patch_features(patch):
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(patch, (24, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = (small - small.mean()) / max(float(small.std()), 5.0)
    return np.clip(small, -3, 3).ravel().astype(np.float32)


def canonical_patch(patch, angle, original_width, original_height):
    h, w = patch.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1)
    matrix[0, 2] += (original_width - w) / 2
    matrix[1, 2] += (original_height - h) / 2
    return cv2.warpAffine(patch, matrix, (original_width, original_height),
                          borderMode=cv2.BORDER_REPLICATE)


class TagVerifier:
    def __init__(self, path, margin=0.0):
        # Only load this project's locally generated, trusted model file.
        self.model = joblib.load(path)
        self.margin = margin

    def accepts(self, patches):
        if not patches:
            return np.empty(0, dtype=bool)
        features = np.stack([patch_features(p) for p in patches])
        return self.model.decision_function(features).ravel() >= self.margin
