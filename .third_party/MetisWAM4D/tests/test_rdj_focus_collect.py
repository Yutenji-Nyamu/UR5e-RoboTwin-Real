import json

from scripts.robodojo import focus_collect as collect
from metiswam4d.eval.rdj_campaign import save_dir,read_details


def test_remaining_collection_copies_ledger_without_rewriting_old_or_new_results(tmp_path,monkeypatch):
    variant=dict(name='fold_clothes',task='fold_clothes',episodes=55)
    def campaign(info):
        return dict(variants=[variant],policy_name='MetisWAM4D',config_name='arx_x5',env_seed=1,additional_info=info)
    old=tmp_path/'old';new=tmp_path/'new'
    path=save_dir(old,campaign('old'),variant)/'_result.json'
    path.parent.mkdir(parents=True)
    original=json.dumps(dict(details={'0':dict(layout_id=0,success=True,score=1),
                                      '1':dict(layout_id=2,success=False,score=.2)}))
    path.write_text(original)
    monkeypatch.setattr(collect,'prepare',lambda **kw:campaign(kw['info']))
    args=(tmp_path/'config.yaml',tmp_path/'weights.pt',new,tmp_path/'records',1,{'hide_track':True},old)
    collect.prepare_collection(*args)
    assert [r['layout_id'] for r in read_details(new,variant)]==[0,2]
    provenance=json.loads((new/'collection_provenance.json').read_text())
    assert provenance['copied_layout_ids']=={'fold_clothes':[0,2]}
    target=save_dir(new,campaign('new'),variant)/'_result.json'
    current=json.loads(target.read_text());current['details']['2']=dict(layout_id=3,success=True,score=1)
    target.write_text(json.dumps(current))
    collect.prepare_collection(*args)
    assert len(read_details(new,variant))==3
    assert path.read_text()==original
