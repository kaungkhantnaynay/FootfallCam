"""Create readable Markdown and plots from a completed run; no invented metrics."""
import argparse
import csv
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',required=True);args=parser.parse_args()
run=Path(args.run);manifest=json.loads((run/'run_manifest.json').read_text())
with open(run/'frames.csv') as f:rows=list(csv.DictReader(f))
times=[float(r['timestamp_s']) for r in rows];counts=[int(r['staff_count']) for r in rows]
intervals=[]
start=None
for i,row in enumerate(rows):
    if int(row['staff_present']) and start is None:start=i
    if start is not None and (not int(row['staff_present']) or i==len(rows)-1):
        end=i if int(row['staff_present']) else i-1
        intervals.append({'start_frame':int(rows[start]['frame_index']),'end_frame':int(rows[end]['frame_index']),
                          'start_s':float(rows[start]['timestamp_s']),'end_s':float(rows[end]['timestamp_s'])})
        start=None
with open(run/'presence_intervals.csv','w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['start_frame','end_frame','start_s','end_s']);writer.writeheader();writer.writerows(intervals)
fig,ax=plt.subplots(figsize=(10,3))
ax.step(times,counts,where='post',color='#167c75');ax.set(xlabel='Video time (seconds)',ylabel='Predicted staff count',title='Predicted staff presence — not ground truth')
ax.set_ylim(bottom=-.1);fig.tight_layout();fig.savefig(run/'presence.png',dpi=160);plt.close(fig)
evaluation=json.loads((run/'evaluation.json').read_text()) if (run/'evaluation.json').exists() else None
lines=['# Run summary','',f"- Decoded frames: {manifest['frames_decoded']}",f"- Input: {manifest['source']}",
       f"- Mode: {manifest['mode']}",f"- Predicted positive frames: {manifest['staff_positive_frames']}",
       f"- Person detection cache used: {manifest['detector_cache_used']}",
       f"- Inference seconds: {manifest['inference_seconds']}",f"- Inference FPS: {manifest['inference_fps']}",
       '- Speed excludes the second-pass video render. Cache reuse removes person inference from this timing.','',
       '![Predicted staff presence](presence.png)','','## Evaluation','']
if evaluation:
    lines.extend(['```json',json.dumps(evaluation,indent=2),'```'])
else:lines.append('Accuracy has not been evaluated for this run.')
lines.extend(['','## Predicted positive intervals (inclusive frame indices)','',
              '| Start frame | End frame | Start seconds | End seconds |','| --- | --- | --- | --- |'])
lines.extend([f"| {r['start_frame']} | {r['end_frame']} | {r['start_s']:.2f} | {r['end_s']:.2f} |" for r in intervals])
lines.extend(['','## Inspect visually','','Play `annotated.mp4`, inspect the `examples/` images and compare against original frames. Predictions can contain misses and false positives.',''])
(run/'SUMMARY.md').write_text('\n'.join(lines))
print(run/'SUMMARY.md')
