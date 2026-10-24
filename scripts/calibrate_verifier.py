"""Reproducible small-patch SVM calibration.

Sources: PDF tag crops, reviewed sample moments at 0/4/8 seconds, and the explicit
additional patch manifest. This is development data, never a holdout test.
"""
from pathlib import Path
import cv2
import json
import numpy as np
from sklearn.svm import SVC
import joblib
import hashlib
from footfallcam.verification import patch_features, canonical_patch

rng = np.random.default_rng(42)
references = [cv2.imread(str(p), 0) for p in sorted(Path('data/reference').glob('tag-*.png'))]
features, labels = [], []
for ref in references:
    for _ in range(600):
        h, w = ref.shape
        corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        perturbed = corners + rng.uniform(-0.09, 0.09, (4, 2)).astype(np.float32) * [w, h]
        warped = cv2.warpPerspective(ref, cv2.getPerspectiveTransform(corners, perturbed.astype(np.float32)), (w, h), borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < .6:
            warped = cv2.GaussianBlur(warped, (3, 3), float(rng.uniform(.2, 1.0)))
        noise = rng.normal(0, rng.uniform(0, 8), warped.shape)
        warped = np.clip(warped.astype(float) * rng.uniform(.65, 1.3) + rng.uniform(-20, 20) + noise, 0, 255).astype(np.uint8)
        features.append(patch_features(warped)); labels.append(1)
cap = cv2.VideoCapture('data/input/sample.mp4')
for sec in (0, 4, 8):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000); ok, frame = cap.read()
    if not ok: raise RuntimeError('Unable to read calibration video')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for _ in range(1600):
        width = int(rng.integers(12, 48)); height = int(rng.integers(8, 36))
        x = int(rng.integers(0, gray.shape[1]-width)); y = int(rng.integers(0, gray.shape[0]-height))
        features.append(patch_features(gray[y:y+height, x:x+width])); labels.append(-1)
    # Mine hard negatives using the same rotated reference proposals as inference.
    from footfallcam.config import Config
    from footfallcam.tags import TagMatcher
    matcher = TagMatcher('data/reference', Config(verifier='', tag_threshold=.65))
    for template, _, angle, width, height in matcher.templates:
        th, tw = template.shape
        response = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        for _ in range(2):
            _, _, _, (x, y) = cv2.minMaxLoc(response)
            patch = canonical_patch(gray[y:y+th, x:x+tw], angle, width, height)
            features.append(patch_features(patch)); labels.append(-1)
            cv2.rectangle(response, (max(0,x-tw), max(0,y-th)), (x+tw,y+th), -1, -1)
cap.release()
additional_dir=Path('data/annotations/calibration_patches')
additional=json.loads((additional_dir/'manifest.json').read_text()) if (additional_dir/'manifest.json').exists() else {'patches':[]}
if additional['patches'] and additional['source_sha256']!=hashlib.sha256(Path('data/input/sample.mp4').read_bytes()).hexdigest():
    raise ValueError('Additional calibration labels belong to another video')
for entry in additional['patches']:
    patch=cv2.imread(str(additional_dir/entry['file']),0)
    if patch is None:raise ValueError(f"Missing calibration patch {entry['file']}")
    if entry['label']==1:
        # Small real tag examples cover blur not represented by the PDF crops.
        for _ in range(500):
            h,w=patch.shape
            scale=float(rng.uniform(.75,1.25))
            resized=cv2.resize(patch,(max(4,round(w*scale)),max(4,round(h*scale))))
            if rng.random()<.5:resized=cv2.GaussianBlur(resized,(3,3),float(rng.uniform(.1,.7)))
            values=np.clip(resized.astype(float)+rng.normal(0,5,resized.shape),0,255).astype(np.uint8)
            features.append(patch_features(values));labels.append(1)
    else:
        for angle in range(0,360,15):
            features.append(patch_features(canonical_patch(patch,angle,24,16)));labels.append(-1)
svm = SVC(C=8, gamma=.005)
svm.fit(np.stack(features), np.array(labels, np.int32))
joblib.dump(svm, 'models/tag_verifier.joblib')
manifest = {'seed': 42, 'positive_count': labels.count(1), 'negative_count': labels.count(-1),
            'source_sha256':hashlib.sha256(Path('data/input/sample.mp4').read_bytes()).hexdigest(),
            'positive_source': 'Three tag crops from employer PDF, synthetic augmentation only',
            'negative_source': 'Reviewed development frames at 0, 4, 8 seconds of sample.mp4',
            'holdout_claim': 'None: this is calibration, not an independent dataset',
            'features': '24x16 normalized grayscale', 'kernel': 'RBF', 'C': 8, 'gamma': .005}
manifest['additional_calibration']=additional
manifest['positive_source']='Three PDF tag crops plus two explicitly labelled sample tag patches, augmented locally'
manifest['negative_source']='Development negatives at 0, 4, 8 seconds plus explicitly labelled sample hard negatives'
Path('models/tag_verifier_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))
