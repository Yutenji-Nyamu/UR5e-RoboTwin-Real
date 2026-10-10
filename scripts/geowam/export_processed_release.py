"""Export the active RGB-D training cache and optional geometry/QA as release assets."""
import argparse, hashlib, json, tarfile, time, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=a.workspace.absolute();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
cache=r/'cache/real_v1';repo=r/'repos/UR5e-RoboTwin-Real'
manifest=[json.loads(x) for x in (cache/'manifests/all.jsonl').read_text().splitlines() if x.strip()]
assert len(manifest)==2405 and len({x['episode'] for x in manifest})==72
assert json.loads((cache/'cache_audit.json').read_text())['sha256']==hashlib.sha256((cache/'manifests/all.jsonl').read_bytes()).hexdigest()
assets=[]
def digest(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def pack(name,kind,paths):
 paths=sorted(set(paths));target=out/name;temp=out/(name+'.part')
 with tarfile.open(temp,'w:gz',compresslevel=1) as tar:
  for f in paths:
   assert f.is_file() and not f.is_symlink()
   tar.add(f,arcname=str(f.relative_to(r)),recursive=False)
 temp.replace(target)
 # Read through every archived member to catch truncated payloads before upload.
 count=0
 with tarfile.open(target,'r:gz') as tar:
  for member in tar:
   stream=tar.extractfile(member);n=0
   for block in iter(lambda:stream.read(8*1024*1024),b''):n+=len(block)
   assert n==member.size;count+=1
 assert count==len(paths)
 info={'name':name,'kind':kind,'bytes':target.stat().st_size,'sha256':digest(target),'files':count}
 assert info['bytes']<1900000000
 assets.append(info);print(json.dumps(info),flush=True)
 (out/'pack_status.json').write_text(json.dumps({'phase':'packing','assets':assets},indent=2))
for task in sorted({x['task'] for x in manifest}):
 pack('geowam-rgbd72-v1-train-'+task+'.tar.gz','train',[Path(x['path']) for x in manifest if x['task']==task])
shared=[*cache.glob('*.json'),cache/'text_context.pt',*cache.glob('manifests/*.jsonl'),*r.glob('data/manifests/*.json*')]
for ep in sorted({x['episode'] for x in manifest}):
 shared += [cache/ep/n for n in ['manifest.jsonl','latents.json','geometry.json','metadata.npz']]
shared += [repo/'configs/geowam'/n for n in ['perception.json','episode_policy.json','ur5e_single_arm_contract.json','train_rgbd72.yaml']]
shared += list((repo/'docs/geowam-real').glob('*.json'))
shared += [r/'runs'/n for n in ['qa_signoff.json','observation_contract_check.json','uvd_semantics_check.json']]
pack('geowam-rgbd72-v1-shared.tar.gz','shared',shared)
geometry=[]
for ep in sorted({x['episode'] for x in manifest}):
 geometry += [cache/ep/'geometry.npz',cache/ep/'masks_full.npz',*list((cache/ep).glob('*.jpg'))]
geometry += [f for f in (r/'evidence').rglob('*') if f.is_file()]
pack('geowam-rgbd72-v1-geometry-qa.tar.gz','geometry_qa',geometry)
receipt={'schema':1,'tag':'processed-rgbd72-v1-20261010','created_unix':time.time(),'original_workspace':str(r),'episodes':72,'windows':2405,'per_task':json.loads((cache/'cache_audit.json').read_text())['per_task'],'source_manifest_sha256':digest(cache/'manifests/all.jsonl'),'code_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'data_commit':subprocess.check_output(['git','-C',str(r/'repos/data_ur5e_gwam'),'rev-parse','HEAD'],text=True).strip(),'assets':assets}
(out/'processed_downloads.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
(out/'SHA256SUMS.txt').write_text(''.join(x['sha256']+'  '+x['name']+'\n' for x in assets))
(out/'pack_status.json').write_text(json.dumps({'phase':'complete','assets':assets,'total_bytes':sum(x['bytes'] for x in assets)},indent=2))
print('PACK_COMPLETE',sum(x['bytes'] for x in assets),flush=True)
