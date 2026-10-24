"""Two-pass local processing: detection/evidence, then decisions/rendering."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import cv2
import numpy as np
from .config import Config
from .detector import PersonDetector
from .geometry import centre
from .tags import TagMatcher
from .tracking import Tracker, staff_intervals
from .decisions import resolve_observations


def sha256(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_csv(path, fields, rows):
    with open(path, 'w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process(args):
    started = time.perf_counter()
    config = Config.load(args.config, device=args.device, model=args.model,
                         tag_threshold=args.tag_threshold)
    source = Path(args.input).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output}. Choose a new run folder.")
    reference_dir = Path(args.references).resolve()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or not np.isfinite(fps) or min(width, height) <= 0:
        cap.release()
        raise ValueError("Video has invalid FPS or dimensions; provide a valid constant-rate MP4.")
    signature = {'source_sha256': sha256(source), 'model_sha256': sha256(config.model),
                 'detector': {k: config.as_dict()[k] for k in ('image_size', 'person_confidence', 'detector_rotations', 'min_person_size')}}
    # Convert tuples to JSON arrays for stable cache comparison.
    signature = json.loads(json.dumps(signature))
    cached, cache_rows = None, []
    cache_path = Path(args.detection_cache) if args.detection_cache else None
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached['signature'] != signature:
            raise ValueError("Detection cache belongs to another input, model, or detector configuration.")
        print('Reusing verified person detections; tag evidence is recomputed.')
    detector = None if cached is not None else PersonDetector(config)
    matcher = TagMatcher(reference_dir, config)
    tracker = Tracker(fps, config)
    output.mkdir(parents=True, exist_ok=True)
    observations, timestamps = [], []
    frame_index = 0
    inference_start = time.perf_counter()
    try:
        while args.max_frames is None or frame_index < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not np.isfinite(timestamp) or timestamp < 0 or (timestamps and timestamp <= timestamps[-1]):
                timestamp = frame_index / fps
            timestamps.append(float(timestamp))
            if cached is not None:
                if frame_index >= len(cached['frames']):
                    raise ValueError("Detection cache is incomplete for this run.")
                detections = [(np.array(row['box'], dtype=float), row['confidence']) for row in cached['frames'][frame_index]]
            else:
                detections = detector.detect(frame)
                cache_rows.append([{'box': box.tolist(), 'confidence': score} for box, score in detections])
            for track, confidence in tracker.update(frame_index, frame, detections):
                check_tag = frame_index % config.tag_stride == 0 or not track.observations
                match = matcher.match(frame, track.box) if check_tag else None
                # Acquisition motion rejects static desk/garment patterns. Once
                # confirmed, the identity may continue while the person stops.
                recent = [o for o in track.observations
                          if frame_index-o['frame_index'] <= config.confirm_window_s*fps]
                displacement = max([np.linalg.norm(np.array(centre(track.box))-centre(o['box']))
                                    for o in recent] or [0.0])
                moving = displacement >= config.acquisition_motion_px
                already_confirmed = bool(staff_intervals(track.hit_frames, fps, config))
                hit = bool(match and match.box is not None and match.score >= config.tag_threshold
                           and (moving or already_confirmed or config.acquisition_motion_px == 0))
                if hit:
                    track.hit_frames.append(frame_index)
                row = {'frame_index': frame_index, 'timestamp_s': round(timestamp, 6),
                       'track_id': track.id, 'box': track.box.tolist(),
                       'person_confidence': float(confidence), 'tag_checked': check_tag,
                       'acquisition_displacement_px': round(float(displacement), 3),
                       'tag_score': float(match.score) if match else None,
                       'tag_box': match.box if hit else None, 'tag_reference': match.reference if hit else '',
                       'tag_hit': hit}
                observations.append(row)
                track.observations.append(row)
            frame_index += 1
            if frame_index % 100 == 0:
                elapsed = time.perf_counter()-inference_start
                print(f'{frame_index} frames; {len(observations)} person observations; {frame_index/elapsed:.2f} processing FPS', flush=True)
    finally:
        cap.release()
    if frame_index == 0:
        raise ValueError("Video contains no decodable frames.")
    inference_seconds = time.perf_counter() - inference_start
    if cache_path and cached is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({'signature': signature, 'frames': cache_rows}))
    track_diagnostics = resolve_observations(observations, fps, config, args.mode)
    (output/'track_diagnostics.json').write_text(json.dumps(track_diagnostics,indent=2)+'\n')
    by_frame = [[] for _ in range(frame_index)]
    for row in observations:
        i = row['frame_index']
        by_frame[i].append(row)
    frame_rows, staff_rows = [], []
    for i, people in enumerate(by_frame):
        staff = [p for p in people if p['staff']]
        frame_rows.append({'frame_index': i, 'timestamp_s': round(timestamps[i], 6),
                           'staff_present': int(bool(staff)), 'staff_count': len(staff)})
        for person in staff:
            x1, y1, x2, y2 = person['box']; x, y = centre(person['box'])
            staff_rows.append({'frame_index': i, 'timestamp_s': round(timestamps[i], 6),
                               'track_id': person['track_id'], 'x': round(x, 2), 'y': round(y, 2),
                               'x1': round(x1, 2), 'y1': round(y1, 2), 'x2': round(x2, 2), 'y2': round(y2, 2),
                               'evidence': person['evidence'], 'tag_score': person['tag_score'],
                               'person_confidence': round(person['person_confidence'], 5)})
    write_csv(output/'frames.csv', ['frame_index','timestamp_s','staff_present','staff_count'], frame_rows)
    write_csv(output/'staff_observations.csv', ['frame_index','timestamp_s','track_id','x','y','x1','y1','x2','y2','evidence','tag_score','person_confidence'], staff_rows)
    with open(output/'observations.jsonl', 'w') as file:
        for row in observations:
            file.write(json.dumps(row)+'\n')
    (output/'resolved_config.json').write_text(json.dumps(config.as_dict(), indent=2)+'\n')
    if not args.no_video:
        render(source, output, fps, width, height, by_frame, timestamps)
    manifest = {'status': 'completed', 'source': str(source), **signature,
                'frames_decoded': frame_index, 'fps': fps, 'width': width, 'height': height,
                'mode': args.mode, 'limited_by_max_frames': args.max_frames is not None,
                'detector_cache_used': cached is not None, 'staff_positive_frames': sum(r['staff_present'] for r in frame_rows),
                'staff_tracks': sorted({r['track_id'] for r in staff_rows}),
                'inference_seconds': round(inference_seconds, 3),
                'inference_fps': round(frame_index / inference_seconds, 3),
                'total_seconds': round(time.perf_counter()-started, 3),
                'coordinate_definition': 'person bounding-box centre; pixels; origin top-left; original image',
                'evaluation_status': 'No accuracy measured by inference. Run scripts/evaluate.py with reviewed labels.',
                'verifier_sha256': sha256(config.verifier) if config.verifier else None,
                'reference_sha256': {p.name: sha256(p) for p in reference_dir.glob('tag-*.png')},
                'platform': platform.platform(), 'machine': platform.machine(), 'python': sys.version.split()[0],
                'packages': {p: importlib.metadata.version(p) for p in ('numpy','opencv-python','scipy','scikit-learn','torch','ultralytics')}}
    (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))
    return manifest


def render(source, output, fps, width, height, by_frame, timestamps):
    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(output/'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    if not writer.isOpened():
        raise RuntimeError('Cannot create MP4 writer on this machine.')
    examples = output/'examples'
    examples.mkdir(exist_ok=True)
    positives = [i for i,p in enumerate(by_frame) if any(o['staff'] for o in p)]
    snapshots = set(int(x) for x in np.linspace(0, len(by_frame)-1, 8))
    if positives:
        snapshots.update(positives[int(x)] for x in np.linspace(0,len(positives)-1,8))
    try:
        for i, people in enumerate(by_frame):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Render decode stopped early at frame {i}')
            count = sum(p['staff'] for p in people)
            cv2.rectangle(frame,(0,0),(width,42),(22,26,30),-1)
            cv2.putText(frame,f'Frame {i} | {timestamps[i]:.2f}s | Staff: {count}',(12,28),0,.65,(255,255,255),2)
            for person in people:
                if not person['staff'] and not person['tag_hit']:
                    continue
                color=(60,220,80) if person['staff'] else (0,190,255)
                x1,y1,x2,y2=map(lambda n:int(round(n)),person['box']);x,y=map(int,centre(person['box']))
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2);cv2.circle(frame,(x,y),4,color,-1)
                label=f"{'STAFF' if person['staff'] else 'TENTATIVE'} #{person['track_id']} ({x},{y})"
                cv2.putText(frame,label,(max(0,x1),max(58,y1-6)),0,.45,color,1)
                if person['tag_box']:
                    a,b,c,d=person['tag_box'];cv2.rectangle(frame,(a,b),(c,d),(255,120,20),2)
            writer.write(frame)
            if i in snapshots:
                cv2.imwrite(str(examples/f'frame-{i:05d}.jpg'),frame)
    finally:
        writer.release();cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--references', default='data/reference')
    parser.add_argument('--device', choices=['cpu','mps','cuda'])
    parser.add_argument('--model')
    parser.add_argument('--tag-threshold', type=float)
    parser.add_argument('--mode', choices=['offline','causal'], default='offline')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--detection-cache', help='Optional input-hash-validated person detection cache')
    args=parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    try:
        process(args)
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as error:
        parser.exit(1,f'Error: {error}\n')
