"""Exercise real video I/O and exports while replacing expensive model inference."""
from types import SimpleNamespace
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from footfallcam import pipeline
from footfallcam.tags import TagMatch


def fixture_args(tmp_path, monkeypatch, positive):
    source=tmp_path/'input.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120))
    assert writer.isOpened()
    for i in range(6):writer.write(np.full((120,160,3),80+i,np.uint8))
    writer.release()
    model=tmp_path/'model.pt';model.write_bytes(b'unit-test-model')
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'model':str(model),'verifier':'','acquisition_motion_px':0,
                                 'min_track_travel_px':0,'confirm_hits':2,'tag_stride':1}))
    class Detector:
        def __init__(self,config):pass
        def detect(self,frame):return [(np.array([20.,20.,80.,100.]),.9)]
    class Matcher:
        def __init__(self,*args):pass
        def match(self,frame,box):return TagMatch(.95,(30,40,50,50),'fixture') if positive else TagMatch(0,None)
    monkeypatch.setattr(pipeline,'PersonDetector',Detector)
    monkeypatch.setattr(pipeline,'TagMatcher',Matcher)
    return SimpleNamespace(config=str(config),device=None,model=None,tag_threshold=None,
                           input=str(source),output=str(tmp_path/'output'),references=str(tmp_path),
                           max_frames=None,no_video=False,mode='offline',detection_cache=None)


@pytest.mark.parametrize('positive',[False,True])
def test_every_frame_and_render_have_consistent_outputs(tmp_path,monkeypatch,positive):
    args=fixture_args(tmp_path,monkeypatch,positive)
    manifest=pipeline.process(args)
    with open(Path(args.output)/'frames.csv') as f:rows=list(csv.DictReader(f))
    assert len(rows)==6 and [int(r['frame_index']) for r in rows]==list(range(6))
    assert all(int(r['staff_present'])==int(positive) for r in rows)
    with open(Path(args.output)/'staff_observations.csv') as f:coords=list(csv.DictReader(f))
    assert len(coords)==(6 if positive else 0)
    if positive:assert float(coords[0]['x'])==50 and float(coords[0]['y'])==60
    cap=cv2.VideoCapture(str(Path(args.output)/'annotated.mp4'))
    decoded=0
    while cap.read()[0]:decoded+=1
    cap.release()
    assert decoded==manifest['frames_decoded']==6


def test_existing_results_are_not_overwritten(tmp_path,monkeypatch):
    args=fixture_args(tmp_path,monkeypatch,False)
    output=Path(args.output);output.mkdir();(output/'important.txt').write_text('keep')
    with pytest.raises(FileExistsError):pipeline.process(args)
    assert (output/'important.txt').read_text()=='keep'


def test_cache_for_different_input_is_rejected(tmp_path,monkeypatch):
    args=fixture_args(tmp_path,monkeypatch,False)
    cache=tmp_path/'wrong.json';cache.write_text(json.dumps({'signature':{'source_sha256':'wrong'},'frames':[]}))
    args.detection_cache=str(cache)
    with pytest.raises(ValueError,match='another input'):pipeline.process(args)
