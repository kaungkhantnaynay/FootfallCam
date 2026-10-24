import numpy as np
from footfallcam.config import Config
from footfallcam.detector import inverse_rotated_box
from footfallcam.geometry import iou
from footfallcam.tracking import staff_intervals, Tracker


def test_rotation_coordinates_use_original_dimensions():
    original=[20,30,80,120]
    assert np.allclose(inverse_rotated_box([600,20,690,80],90,960,720),original)
    assert np.allclose(inverse_rotated_box([880,600,940,690],180,960,720),original)
    assert np.allclose(inverse_rotated_box([30,880,120,940],270,960,720),original)


def test_single_or_duplicate_hit_cannot_confirm_staff():
    assert staff_intervals([5,5,5],25,Config())==[]
    assert staff_intervals([5,80,160],25,Config())==[]


def test_causal_cannot_backdate_and_offline_is_bounded():
    config=Config(confirm_hits=3,evidence_ttl_s=1,offline_backfill_s=.2)
    assert staff_intervals([10,12,14],25,config)==[(14,39)]
    assert staff_intervals([10,12,14],25,config,backfill=True)==[(5,39)]


def test_reentry_after_expiry_gets_new_id_and_no_staff_hits():
    frame=np.full((200,200,3),100,np.uint8)
    tracker=Tracker(25,Config(max_track_gap_s=.2))
    first=tracker.update(0,frame,[(np.array([30,30,90,100]),.9)])[0][0]
    first.hit_frames.extend([0,1,2])
    second=tracker.update(20,frame,[(np.array([30,30,90,100]),.9)])[0][0]
    assert first.id!=second.id
    assert not second.hit_frames


def test_iou_disjoint_and_identical():
    assert iou([0,0,10,10],[20,20,30,30])==0
    assert iou([0,0,10,10],[0,0,10,10])==1
