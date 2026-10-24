import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('evaluation',Path(__file__).parents[1]/'scripts/evaluate.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_sparse_labels_do_not_turn_other_frames_into_negatives(tmp_path):
    (tmp_path/'run_manifest.json').write_text(json.dumps({'source_sha256':'abc'}))
    (tmp_path/'frames.csv').write_text('frame_index,staff_present\n0,0\n1,1\n2,1\n3,0\n')
    (tmp_path/'staff_observations.csv').write_text('frame_index,x,y\n1,20,30\n2,40,50\n')
    labels={'scope':'unit fixture','source_sha256':'abc','frames':[
        {'frame_index':0,'staff_present':False},
        {'frame_index':1,'staff_present':True,'centres':[[20,30]]}]}
    path=tmp_path/'labels.json';path.write_text(json.dumps(labels))
    result=module.evaluate(tmp_path,path)
    assert result['unlabelled_frames']==2
    assert result['tp']==1 and result['tn']==1
    assert result['mean_position_error_px']==0


def test_evaluation_rejects_labels_from_another_input(tmp_path):
    (tmp_path/'run_manifest.json').write_text(json.dumps({'source_sha256':'abc'}))
    path=tmp_path/'labels.json';path.write_text(json.dumps({'source_sha256':'other'}))
    with pytest.raises(ValueError,match='another video'):
        module.evaluate(tmp_path,path)
