from pathlib import Path
import json,os,time,traceback,urllib.request
from huggingface_hub import snapshot_download
ROOT=Path('/data/chenyiteng/projects/geowam-real')
JOBS=[
 ('OpenWAM/OpenWAM-Alpha-Real-Single-Arm-Franka','e75522eb38f9b4148d5ecb985ab80c490ddb3854',['checkpoint_step_9860.safetensors','config.yaml','README.md','normalization_stats.npy','tokenizer/*'],'alpha-franka'),
 ('Wan-AI/Wan2.2-TI2V-5B-Diffusers','b8fff7315c768468a5333511427288870b2e9635',['vae/*','text_encoder/*','tokenizer/*'],'wan22-ti2v-5b'),
]
status={}
for repo,rev,patterns,name in JOBS:
 status[name]={'repo':repo,'revision':rev,'state':'downloading','started':time.time()}
 (ROOT/'runs/assets-status.json').write_text(json.dumps(status,indent=2))
 try:
  snapshot_download(repo,revision=rev,allow_patterns=patterns,local_dir=ROOT/'models'/name,max_workers=2)
  status[name].update(state='complete',finished=time.time())
 except Exception as e:
  status[name].update(state='error',error=str(e));traceback.print_exc()
 (ROOT/'runs/assets-status.json').write_text(json.dumps(status,indent=2))
 print(name,status[name]['state'],flush=True)
