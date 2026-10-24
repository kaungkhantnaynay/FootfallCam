from footfallcam.config import Config
from footfallcam.decisions import resolve_observations


def rows(step):
    return [{'track_id':1,'frame_index':i,'box':[20+step*i,20,80+step*i,100],
             'tag_hit':i in (0,1,2),'staff':False} for i in range(12)]


def test_stationary_pattern_cannot_pass_whole_track_motion_gate():
    observations=rows(0)
    resolve_observations(observations,25,Config(), 'offline')
    assert not any(r['staff'] for r in observations)


def test_causal_motion_does_not_use_future_travel():
    observations=rows(15)
    resolve_observations(observations,25,Config(), 'causal')
    assert not observations[2]['staff']
    assert observations[-1]['staff']


def test_offline_motion_allows_bounded_earlier_decisions():
    observations=rows(15)
    resolve_observations(observations,25,Config(), 'offline')
    assert observations[0]['staff']
