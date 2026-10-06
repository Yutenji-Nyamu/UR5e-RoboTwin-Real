import json

from scripts.eval import rdj_paired_report as report


def test_routing_keeps_paired_layouts_and_one_policy_per_base_task(tmp_path, monkeypatch):
    dims = dict(zip(report.DIM_ORDER, [['g'], ['p'], ['l'], ['m'], ['o']]))
    variants = []
    details = {}
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir(); new.mkdir()
    for dim, names in dims.items():
        task = names[0]
        for split in (['standard', 'random'] if task == 'g' else ['standalone']):
            name = task + ('_random' if split == 'random' else '')
            n = 42 if task == 'g' else 84
            variants.append(dict(name=name, task=name, base_task=task, dimension=dim, split=split, episodes=n))
            details[(old, name)] = [dict(layout_id=i, success=False, score=0) for i in range(n)]
    (old / 'campaign.json').write_text(json.dumps(dict(variants=variants, dimensions=dims)))
    newvars = []
    for name in ['g', 'g_random']:
        for label in ['base', 'hide']:
            v = next(v for v in variants if v['name'] == name)
            newvars.append({**v, 'name': name+'@'+label, 'label': label, 'episodes': 47})
            wins = 2 if (name, label) == ('g', 'base') else 3 if (name, label) == ('g_random', 'hide') else 0
            details[(new, name+'@'+label)] = [dict(layout_id=i, success=i<wins or i>=42, score=float(i<wins or i>=42))
                                              for i in range(47)]
    (new / 'campaign.json').write_text(json.dumps(dict(variants=newvars)))
    monkeypatch.setattr(report, 'read_details', lambda out, name: details[(out,name)])
    monkeypatch.setattr(report, 'baseline_tasks', lambda *_: {t:dict(successes=0,completed=10) for ts in dims.values() for t in ts})
    result = report.focus_report(old,[new],tmp_path/'report')
    assert result['routed']['episodes'] == 420
    assert result['routed']['successes'] == 3   # Extra layouts and cherry-picking standard separately are excluded.
    assert result['route']['g'].endswith('/hide')
    assert result['paired_totals'][result['route']['g']]['success'] == 3
    assert result['expanded_totals'][result['route']['g']]['success'] == 13
    old_base=tmp_path/'old_base'
    for name in ['g','g_random']:
        details[(old_base,name)]=[dict(layout_id=i,success=name=='g' and i<4,score=float(name=='g' and i<4))
                                  for i in range(42)]
    routed=report.focus_report(old,[new],tmp_path/'report_with_old_base',old_base)
    assert routed['route']['g']=='old-base'
    assert routed['routed']['successes']==4 and routed['routed']['episodes']==420
