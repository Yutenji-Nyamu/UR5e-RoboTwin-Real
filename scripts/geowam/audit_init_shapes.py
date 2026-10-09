"""Compare official Alpha tensor metadata to current Metis modules without allocating weights."""
import argparse,collections,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();root=a.workspace
sys.path.insert(0,str(root/'repos/MetisWAM4D'))
import torch
from metiswam4d.build import build_model
from metiswam4d.config import ModelConfig
from metiswam4d.experts.alpha import action_key_from_alpha
from metiswam4d.experts.init import source_key_for
header=json.loads((root/'evidence/alpha_franka_header.json').read_text())
with torch.device('meta'):model=build_model(ModelConfig())
report={'parameter_counts':dict(collections.Counter()),'alpha':{},'track':{}}
for name,param in model.named_parameters():
 group=name.split('.')[0];report['parameter_counts'][group]=report['parameter_counts'].get(group,0)+param.numel()
for label,module,prefix,rename in [('video',model.video,'video_backbone.dit.',lambda x:x),('action',model.action,'action_backbone.',action_key_from_alpha),('proprio',model.proprio_encoder,'proprio_encoder.',lambda x:x)]:
 target=module.state_dict();source={rename(k[len(prefix):]):v for k,v in header.items() if k.startswith(prefix)}
 missing=[k for k in target if k not in source and k!='demo_embedding'];fresh=[k for k in target if k=='demo_embedding']
 mismatch={k:[list(target[k].shape),source[k]['shape']] for k in target if k in source and list(target[k].shape)!=source[k]['shape']}
 report['alpha'][label]={'matched':sum(k in source and list(v.shape)==source[k]['shape'] for k,v in target.items()),'missing':missing,'mismatched':mismatch,'fresh':fresh,'unexpected':sorted(set(source)-set(target))}
 assert not missing and not mismatch
track=collections.defaultdict(list)
for k,v in model.track.state_dict().items():
 src=source_key_for(k,20,30,'video_backbone.dit.')
 kind='derived_condition' if k.startswith('condition_fusion.') else ('fresh' if src not in header else ('copy' if list(v.shape)==header[src]['shape'] else 'interpolate'))
 track[kind].append(k)
report['track']=dict(track);report['total_parameters']=sum(report['parameter_counts'].values())
(root/'evidence/initialization_shapes.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'total_parameters':report['total_parameters'],'parameter_counts':report['parameter_counts'],'alpha':report['alpha'],'track_counts':{k:len(v) for k,v in track.items()},'track_fresh':track['fresh']},indent=2))
