"""Validate the focused training index, source uniqueness, held-out rows, and real loader windows."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from metiswam4d.data.robodojo.episode_dataset import RDJEpisodeDataset, RDJWindowConfig
from scripts.data_prep.robodojo.rdj_object_track import TASKS, ORIGINAL


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--repeat',type=int,required=True)
    args=ap.parse_args()
    rows=[json.loads(l) for l in (args.root/'index.jsonl').read_text().splitlines()]
    original=[json.loads(l) for l in (ORIGINAL/'index.jsonl').read_text().splitlines()]
    assert [r for r in rows if r['split']=='val']==[r for r in original if r['split']=='val']
    assert all(r['task'].removesuffix('_random') in TASKS for r in rows if r['split']=='train')
    unique={}
    counts=Counter()
    for r in rows:
        if not r.get('rollout'): continue
        key=(r['task'],r['variant'],r['episode'])
        unique[key]=r
        counts[key]+=1
    assert all(n==args.repeat for n in counts.values())
    sources=set(); accepted=Counter(); successes=Counter(); source_counts=Counter()
    for (task,variant,episode),r in unique.items():
        meta=json.loads((args.root/task/variant/f'episode{episode}'/'meta.json').read_text())
        assert meta['success'] or meta['final_score']>=50
        assert meta['source'] not in sources, meta['source']
        sources.add(meta['source'])
        source=Path(meta['source']).parent.parent.name
        if source=='records_v1': assert meta['layout_id']>=10
        accepted[task.removesuffix('_random')]+=1
        successes[task.removesuffix('_random')]+=int(meta['success'])
        source_counts[source]+=1
    ds=RDJEpisodeDataset(RDJWindowConfig(root=str(args.root),samples_per_episode=1,fixed_windows=True))
    seen=set(); windows=[]
    for i,r in enumerate(ds.rows):
        group=(r['task'],r.get('rollout',False))
        if group in seen:continue
        item=ds._load(i)  # Direct load: a broken sample must not silently fall back in this acceptance check.
        assert item['action'].shape==(32,80) and torch.isfinite(item['action']).all()
        assert item['track_rgb'].shape==(9,240,320,3)
        assert not item['head_depth_mm'][~item['head_mask']].any()
        assert item['prompt'] and item['text_context'].shape[1]==4096
        seen.add(group);windows.append(item['key'])
    val=RDJEpisodeDataset(RDJWindowConfig(root=str(args.root),split='val',samples_per_episode=1,fixed_windows=True))
    val._load(0)
    official=Counter(r['task'] for r in rows if r['split']=='train' and not r.get('rollout'))
    result=dict(train_rows=len(ds.rows),val_rows=len(val.rows),official_train=sum(official.values()),
                unique_rollouts=len(unique),repeat=args.repeat,source_counts=dict(source_counts),
                per_task={t:dict(official=official[t],rollout=accepted[t],success=successes[t]) for t in TASKS},
                checked_windows=windows)
    (args.root/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
