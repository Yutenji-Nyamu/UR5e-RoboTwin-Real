from pathlib import Path
import subprocess,os,json,time,argparse
from pipeline_common import *
parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int,action='append',default=[]);args=parser.parse_args()
for pid in args.wait_pid:
 while Path(f'/proc/{pid}').exists():time.sleep(5)
gpu='GPU-dc5d6921-fa81-b666-bac7-566c126f1dd4'
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert not any(l.split(',')[0].strip()==gpu for l in active.splitlines() if ',' in l),active
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=gpu,USE_TF='0',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4')
subprocess.run([str(ROOT/'envs/preprocess/bin/python'),str(REPO/'scripts/geowam/cache_latents.py'),'latents','--worker','0','--workers','2','--wait'],env=env,check=True)
receipt=json.loads((ROOT/'runs/cache-latents-0.json').read_text());assert receipt['phase']=='complete' and not receipt['errors']
p=ROOT/'runs/refine-worker0.json';old=json.loads(p.read_text());atomic_json(p,{'phase':'complete','recovered':True,'first_attempt':old,'recovery_cache_pid':receipt['pid'],'heartbeat':time.time()})
subprocess.run([str(ROOT/'envs/preprocess/bin/python'),str(REPO/'scripts/geowam/finalize_cache.py')],env={**env,'CUDA_VISIBLE_DEVICES':''},check=True)
print('FINAL_CACHE_RECOVERY_AND_AUDIT_PASSED',flush=True)
