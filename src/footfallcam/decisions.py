"""Resolve frozen visual evidence into inspectable staff decisions.

This stage can be replayed without rerunning neural inference. Offline mode may
use a whole track's motion; causal mode only uses observations available so far.
"""
from collections import defaultdict
import numpy as np
from .geometry import centre
from .tracking import staff_intervals


def robust_travel(points):
    if len(points)<2:return 0.0
    array=np.asarray(points)
    return float(np.linalg.norm(np.percentile(array,95,axis=0)-np.percentile(array,5,axis=0)))


def resolve_observations(observations, fps, config, mode):
    grouped=defaultdict(list)
    for row in observations:grouped[row['track_id']].append(row)
    diagnostics=[]
    for tid,rows in grouped.items():
        points=[centre(row['box']) for row in rows]
        travel=robust_travel(points)
        intervals=staff_intervals([r['frame_index'] for r in rows if r['tag_hit']],fps,config,backfill=mode=='offline')
        for j,row in enumerate(rows):
            eligible=(travel if mode=='offline' else robust_travel(points[:j+1]))>=config.min_track_travel_px
            row['staff']=bool(eligible and any(a<=row['frame_index']<=b for a,b in intervals))
            row['evidence']=('direct_tag' if row['tag_hit'] else 'tracked_identity') if row['staff'] else 'tentative_tag' if row['tag_hit'] else 'unconfirmed'
        diagnostics.append({'track_id':tid,'observations':len(rows),'robust_travel_px':round(travel,3),
                            'tag_hits':sum(r['tag_hit'] for r in rows),
                            'staff_frames':sum(r['staff'] for r in rows)})
    return diagnostics
