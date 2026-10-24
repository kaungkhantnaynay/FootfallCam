"""Recreate reviewed patch images from their explicit crop manifest and video."""
import json
from pathlib import Path
import cv2
from footfallcam.pipeline import sha256
from footfallcam.verification import canonical_patch

source=Path('data/input/sample.mp4')
directory=Path('data/annotations/calibration_patches')
manifest=json.loads((directory/'manifest.json').read_text())
if sha256(source)!=manifest['source_sha256']:raise ValueError('Calibration manifest belongs to another video')
cap=cv2.VideoCapture(str(source))
try:
    for row in manifest['patches']:
        cap.set(cv2.CAP_PROP_POS_FRAMES,row['frame_index']);ok,frame=cap.read()
        if not ok:raise ValueError(f"Missing source frame {row['frame_index']}")
        a,b,c,d=row['box_xyxy'];patch=frame[b:d,a:c]
        if patch.size==0:raise ValueError('Empty calibration crop')
        if 'inverse_rotation_deg' in row:
            patch=canonical_patch(patch,row['inverse_rotation_deg'],row['canonical_width'],row['canonical_height'])
        if not cv2.imwrite(str(directory/row['file']),patch):raise RuntimeError('Cannot save calibration image')
finally:cap.release()
print(f"Rebuilt {len(manifest['patches'])} labelled development patches.")
