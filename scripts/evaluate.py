"""Evaluate only explicitly reviewed frame labels; never assume unlabelled=negative."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from footfallcam.pipeline import sha256


def evaluate(run_dir, labels_path):
    run = Path(run_dir)
    labels = json.loads(Path(labels_path).read_text())
    manifest = json.loads((run/'run_manifest.json').read_text())
    if labels['source_sha256'] != manifest['source_sha256']:
        raise ValueError('Labels belong to another video.')
    with open(run/'frames.csv') as file:
        predicted = {int(row['frame_index']): row for row in csv.DictReader(file)}
    coords = {}
    with open(run/'staff_observations.csv') as file:
        for row in csv.DictReader(file):
            coords.setdefault(int(row['frame_index']), []).append([float(row['x']),float(row['y'])])
    counts = {'tp':0,'fp':0,'tn':0,'fn':0}
    distances, labelled, ignored, missed_positions = [], [], [], 0
    for label in labels['frames']:
        i = int(label['frame_index'])
        if i not in predicted:
            raise ValueError(f'Labelled frame {i} is missing from predictions')
        if label.get('staff_present') is None:
            ignored.append(i);continue
        truth = bool(label['staff_present']);guess = bool(int(predicted[i]['staff_present']))
        counts['tp' if truth and guess else 'fn' if truth else 'fp' if guess else 'tn'] += 1
        labelled.append(i)
        expected = label.get('centres', [])
        actual = coords.get(i, [])
        if expected and actual:
            cost = np.linalg.norm(np.asarray(expected)[:,None,:] - np.asarray(actual)[None,:,:], axis=2)
            rows, cols = linear_sum_assignment(cost)
            # Explicit gate avoids reporting a distant wrong person as a matched position.
            accepted = [float(cost[a,b]) for a,b in zip(rows,cols) if cost[a,b] <= labels.get('position_gate_px',100)]
            distances.extend(accepted);missed_positions += len(expected)-len(accepted)
        elif expected:
            missed_positions += len(expected)
    def ratio(a,b): return a/b if b else None
    precision=ratio(counts['tp'],counts['tp']+counts['fp'])
    recall=ratio(counts['tp'],counts['tp']+counts['fn'])
    f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else 0.0 if precision is not None and recall is not None else None
    return {'scope':labels['scope'], 'labels_sha256':sha256(labels_path),
            'evaluation_type':labels.get('evaluation_type','sample_diagnostics'),
            'calibration_overlap_frames':labels.get('calibration_overlap_frames',[]),
            'reviewed_frames':len(labelled), 'ignored_frames':ignored,
            'unlabelled_frames':len(predicted)-len(labelled)-len(ignored), **counts,
            'precision':precision,'recall':recall,'f1':f1,
            'matched_position_count':len(distances),'missed_labelled_positions':missed_positions,
            'mean_position_error_px':float(np.mean(distances)) if distances else None,
            'median_position_error_px':float(np.median(distances)) if distances else None,
            'warning':'Sparse, approximate sample diagnostics. No unseen-video accuracy is established.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--labels',required=True)
    args=parser.parse_args()
    result=evaluate(args.run,args.labels)
    (Path(args.run)/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
