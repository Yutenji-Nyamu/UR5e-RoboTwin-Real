"""Validate all cached samples and assemble equal-task manifests; no synthetic replacement."""
import json,sys,hashlib
from pathlib import Path
from pipeline_common import *
sys.path.insert(0,str(ROOT/'repos/MetisWAM4D'))
import torch
from metiswam4d.data.contract import SampleSpec,validate_sample
spec=SampleSpec();groups={k:[] for k in TASKS};audit=[]
for row in rows():
 ep=CACHE/row['key'];g=json.loads((ep/'geometry.json').read_text());v=json.loads((ep/'latents.json').read_text());m=np.load(ep/'metadata.npz')
 assert g['state']==v['state']=='complete' and not g['missing_initial']
 cfg=json.loads((CONFIG/'perception.json').read_text());assert g['perception_signature']==perception_signature(cfg,row['task'],row['key'])
 signature=hashlib.sha256((ep/'geometry.npz').read_bytes()+(CACHE/'normalization.json').read_bytes()+b'latent_contract_v1').hexdigest();assert v['input_signature']==signature
 assert v['windows']==v['all_windows']==len(m['starts'])
 entries=[json.loads(x) for x in (ep/'manifest.jsonl').read_text().splitlines()]
 assert len(entries)==len(m['starts'])
 for e in entries:
  sample=torch.load(e['path'],map_location='cpu',weights_only=True);validate_sample(sample,spec)
  for k,x in sample.items():
   if torch.is_tensor(x) and x.is_floating_point():assert torch.isfinite(x).all(),(e['key'],k)
  assert sample['action_mask'].sum()==32*7 and sample['proprio_mask'].sum()==7
  assert set(torch.nonzero(sample['proprio_mask'][0]).flatten().tolist())==set(range(10,17))
 groups[row['task']].extend(entries);audit.append({'episode':row['key'],'windows':len(entries),'fg_valid':g['mean_valid_fg'],'object_valid':g['mean_valid_object']})
manifest=CACHE/'manifests';manifest.mkdir(exist_ok=True);allrows=[]
for task,entries in groups.items():
 content=''.join(json.dumps(e)+'\n' for e in entries);(manifest/f'{task}.jsonl').write_text(content);allrows+=entries
(manifest/'all.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in allrows))
atomic_json(CACHE/'cache_audit.json',{'state':'passed','finished':time.time(),'perception_sha256':hashlib.sha256((CONFIG/'perception.json').read_bytes()).hexdigest(),'normalization_sha256':hashlib.sha256((CACHE/'normalization.json').read_bytes()).hexdigest(),'episodes':len(audit),'windows':len(allrows),'per_task':{k:len(v) for k,v in groups.items()},'sample_validation':'all tensors finite and exact source contract','sha256':hashlib.sha256((manifest/'all.jsonl').read_bytes()).hexdigest(),'episodes_report':audit});print((CACHE/'cache_audit.json').read_text()[:1600])
