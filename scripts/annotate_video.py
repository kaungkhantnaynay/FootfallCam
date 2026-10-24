"""Local keyboard review tool. Run manually on the laptop with a display.

Space: play/pause. A/D: previous/next frame. J/L: jump one second.
P: label staff present. N: label absent. U: mark uncertain. C: clear label.
S: save JSON. Q/Escape: save and exit. No labels are inferred automatically.
"""
import argparse
import json
from pathlib import Path
import cv2
from footfallcam.pipeline import sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',required=True);parser.add_argument('--labels',required=True)
args=parser.parse_args();source=Path(args.input);destination=Path(args.labels)
digest=sha256(source)
if destination.exists():
    document=json.loads(destination.read_text())
    if document['source_sha256']!=digest:raise ValueError('Existing labels refer to another video')
else:
    document={'source_sha256':digest,'scope':'Candidate manual review; record development/test status before reporting metrics',
              'position_gate_px':100,'frames':[]}
labels={int(row['frame_index']):row for row in document['frames']}
cap=cv2.VideoCapture(str(source))
if not cap.isOpened():raise ValueError('Cannot open video')
fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if fps<=0 or total<=0:raise ValueError('Video requires valid FPS and frame count')
i=0;playing=False

def save():
    document['frames']=[labels[k] for k in sorted(labels)]
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(document,indent=2)+'\n')
    print(f'Saved {len(labels)} reviewed frames to {destination}')

try:
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read()
        if not ok:break
        state=labels.get(i,{}).get('staff_present','UNREVIEWED')
        cv2.rectangle(frame,(0,0),(frame.shape[1],65),(25,25,25),-1)
        cv2.putText(frame,f'Frame {i}/{total-1} | {i/fps:.2f}s | label: {state}',(10,25),0,.55,(255,255,255),1)
        cv2.putText(frame,'P present | N absent | U unsure | A/D step | J/L 1s | S save | Q quit',(10,52),0,.45,(255,255,255),1)
        cv2.imshow('Original footage annotation',frame)
        key=cv2.waitKey(max(1,round(1000/fps)) if playing else 0)&0xff
        if key in (ord('q'),27):save();break
        if key==ord(' '):playing=not playing
        if key in (ord('p'),ord('n'),ord('u')):
            labels[i]={'frame_index':i,'staff_present':True if key==ord('p') else False if key==ord('n') else None,
                       'centres':[],'note':'Candidate manual review'}
        if key==ord('c'):labels.pop(i,None)
        if key==ord('s'):save()
        if key==ord('a'):i=max(0,i-1);playing=False
        elif key==ord('d'):i=min(total-1,i+1);playing=False
        elif key==ord('j'):i=max(0,i-round(fps));playing=False
        elif key==ord('l'):i=min(total-1,i+round(fps));playing=False
        elif playing:i=min(total-1,i+1)
finally:
    cap.release();cv2.destroyAllWindows()
