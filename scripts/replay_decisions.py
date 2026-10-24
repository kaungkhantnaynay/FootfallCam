"""Replay temporal/motion decisions on frozen observations, clearly labelled.

This does not benchmark detector inference and does not process a new video.
Only decision-stage settings can change; detector and tag evidence stay frozen.
"""
import argparse
from pathlib import Path
import json,time,copy
from footfallcam.config import Config
from footfallcam.decisions import resolve_observations
from footfallcam.geometry import centre
from footfallcam.pipeline import write_csv,render,sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
parser.add_argument('--config',default='configs/default.json')
args=parser.parse_args();started=time.perf_counter()
source_run=Path(args.run).resolve();out=Path(args.output).resolve()
if out.exists() and any(out.iterdir()):raise ValueError('Choose an empty output folder')
manifest=json.loads((source_run/'run_manifest.json').read_text())
source=Path(manifest['source'])
if sha256(source)!=manifest['source_sha256']:raise ValueError('Source video changed')
config=Config.load(args.config)
old_config=json.loads((source_run/'resolved_config.json').read_text())
allowed={'min_track_travel_px','confirm_hits','confirm_window_s','evidence_ttl_s','offline_backfill_s'}
for k,v in config.as_dict().items():
    comparable=list(v) if isinstance(v,tuple) else v
    if k not in allowed and comparable!=old_config.get(k):raise ValueError(f'Frozen evidence does not support changing {k}')
observations=[json.loads(line) for line in (source_run/'observations.jsonl').read_text().splitlines()]
with open(source_run/'frames.csv') as f:
    import csv
    previous=list(csv.DictReader(f))
times=[float(r['timestamp_s']) for r in previous]
diagnostics=resolve_observations(observations,manifest['fps'],config,manifest['mode'])
by_frame=[[] for _ in times]
for row in observations:by_frame[row['frame_index']].append(row)
frame_rows=[];staff_rows=[]
for i,people in enumerate(by_frame):
    staff=[r for r in people if r['staff']]
    frame_rows.append({'frame_index':i,'timestamp_s':times[i],'staff_present':int(bool(staff)),'staff_count':len(staff)})
    for r in staff:
        x,y=centre(r['box']);x1,y1,x2,y2=r['box']
        staff_rows.append({'frame_index':i,'timestamp_s':times[i],'track_id':r['track_id'],
                           'x':round(x,2),'y':round(y,2),'x1':round(x1,2),'y1':round(y1,2),
                           'x2':round(x2,2),'y2':round(y2,2),'evidence':r['evidence'],
                           'tag_score':r['tag_score'],'person_confidence':round(r['person_confidence'],5)})
out.mkdir(parents=True,exist_ok=True)
write_csv(out/'frames.csv',['frame_index','timestamp_s','staff_present','staff_count'],frame_rows)
write_csv(out/'staff_observations.csv',['frame_index','timestamp_s','track_id','x','y','x1','y1','x2','y2','evidence','tag_score','person_confidence'],staff_rows)
(out/'observations.jsonl').write_text('\n'.join(json.dumps(r) for r in observations)+'\n')
(out/'resolved_config.json').write_text(json.dumps(config.as_dict(),indent=2)+'\n')
(out/'track_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
render(source,out,manifest['fps'],manifest['width'],manifest['height'],by_frame,times)
manifest['decision_replay']=True
manifest['evidence_source_run']=str(source_run)
manifest['evidence_source_sha256']=sha256(source_run/'observations.jsonl')
manifest['original_inference_seconds']=manifest.pop('inference_seconds')
manifest['original_inference_fps']=manifest.pop('inference_fps')
manifest['inference_seconds']=None;manifest['inference_fps']=None
manifest['total_seconds']=round(time.perf_counter()-started,3)
manifest['staff_positive_frames']=sum(r['staff_present'] for r in frame_rows)
manifest['staff_tracks']=sorted({r['track_id'] for r in staff_rows})
(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'output':str(out),'positive_frames':manifest['staff_positive_frames'],'staff_tracks':manifest['staff_tracks'],'decision_replay':True},indent=2))
