import json
from types import SimpleNamespace

import h5py
import pytest

from scripts.data_prep.robodojo import rdj_rollout_dataset as rollout


def test_existing_episode_from_other_recording_is_rejected(tmp_path):
    path=tmp_path/'record.hdf5'
    with h5py.File(path,'w') as f:f.attrs['task']='build_tower'
    dest=rollout.episode_dir(tmp_path/'out','build_tower','v2',7)
    dest.mkdir(parents=True)
    (dest/'meta.json').write_text(json.dumps({'source':str(tmp_path/'other.hdf5')}))
    with pytest.raises(ValueError,match='episode collision'):
        rollout._build(path,tmp_path/'out','v2',7)


def test_focus_filter_includes_random_and_applies_source_offset(tmp_path,monkeypatch):
    path=tmp_path/'records/fold_clothes_random/layout_001.hdf5'
    path.parent.mkdir(parents=True)
    with h5py.File(path,'w') as f:
        f.attrs.update(task='fold_clothes_random',variant='fold_clothes_random',frames=40,success=True,layout_id=1)
    captured=[]
    class Pool:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def imap_unordered(self,fn,jobs):
            captured.extend(jobs)
            return [{'status':'exists'} for _ in jobs]
    monkeypatch.setattr(rollout,'Pool',Pool)
    rollout.cmd_build(SimpleNamespace(out=tmp_path/'out',record=tmp_path/'records',tag='v2',workers=1,
                                     tasks='fold_clothes',min_score=50,successes_only=False,episode_offset=10000))
    assert len(captured)==1 and captured[0][-1]==10001


def test_next_round_repeat_replaces_previous_sampling_weight(tmp_path,monkeypatch):
    original=tmp_path/'official';original.mkdir()
    row=dict(task='build_tower',variant='train_4d',episode=0,frames=40,split='train')
    (original/'index.jsonl').write_text(json.dumps(row)+'\n'+json.dumps({**row,'episode':99,'split':'val'})+'\n')
    (original/'episode_instructions_official.jsonl').write_text('')
    previous=tmp_path/'previous';previous.mkdir()
    old={**row,'variant':'rollout_v2','rollout':True,'success':True}
    (previous/'index.jsonl').write_text((json.dumps(old)+'\n')*3)
    (previous/'episode_instructions_official.jsonl').write_text('')
    dest=tmp_path/'out/build_tower/rollout_v3/episode10000';dest.mkdir(parents=True)
    (dest/'meta.json').write_text(json.dumps(dict(task='build_tower',episode=10000,frames=40,status='ok',
                                                success=True,final_score=100,instruction='Build a tower.')))
    monkeypatch.setattr(rollout,'ORIGINAL',original)
    rollout.cmd_finalize(SimpleNamespace(out=tmp_path/'out',tag='v3',repeat=5,tasks='build_tower',include_dataset=previous))
    rows=[json.loads(l) for l in (tmp_path/'out/index.jsonl').read_text().splitlines()]
    assert len(rows)==12
    assert sum(r['variant']=='rollout_v2' for r in rows)==5
    assert sum(r['variant']=='rollout_v3' for r in rows)==5
    assert sum(r['split']=='val' for r in rows)==1
