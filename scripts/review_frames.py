"""Export an original-frame contact sheet for manual annotation; no predictions."""
import argparse
from pathlib import Path
import cv2
from PIL import Image,ImageDraw

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--frames',required=True,help='Comma-separated zero-based frame indices')
args=parser.parse_args()
indices=[int(v) for v in args.frames.split(',')]
if not indices or any(i<0 for i in indices):parser.error('Frame indices must be non-negative')
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
cap=cv2.VideoCapture(args.input)
if not cap.isOpened():raise ValueError('Cannot open video')
sheet=Image.new('RGB',(960,386*((len(indices)+1)//2)),'white');draw=ImageDraw.Draw(sheet)
for n,i in enumerate(indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read()
    if not ok:raise ValueError(f'Cannot decode frame {i}')
    cv2.imwrite(str(out/f'original-{i:05d}.jpg'),frame)
    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((480,360))
    x=(n%2)*480;y=(n//2)*386;sheet.paste(im,(x,y));draw.text((x+8,y+364),f'Frame {i}',fill='black')
sheet.save(out/'review.jpg');cap.release()
print(out/'review.jpg')
