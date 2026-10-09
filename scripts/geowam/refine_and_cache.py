"""Finish scoped mask refinement, rebuild role-consistent UVD, then cache disjoint episode shards."""
import argparse,json,os,subprocess,time
from pathlib import Path
from pipeline_common import *
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args()
assert a.worker in (0,1)
gpu=['GPU-dc5d6921-fa81-b666-bac7-566c126f1dd4','GPU-3c6321c1-3e58-c071-3867-533391152fe7'][a.worker]
status=ROOT/f'runs/refine-worker{a.worker}.json';state={'phase':'waiting_previous','pid':os.getpid(),'gpu':gpu,'wait_pid':a.wait_pid,'heartbeat':time.time()};atomic_json(status,state)
while Path(f'/proc/{a.wait_pid}').exists():time.sleep(10)
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert not any(line.split(',')[0].strip()==gpu for line in active.splitlines() if ',' in line),active
if a.worker==0:
 cfg=json.loads((CONFIG/'perception.json').read_text())
 for row in rows():
  if row['key'] in cfg['episode_repairs']:
   rep=json.loads((CACHE/row['key']/'mask_repair_report.json').read_text());assert rep['perception_signature']==perception_signature(cfg,row['task'],row['key'])
cmds=[('geometry',[str(REPO/'scripts/geowam/process_geometry.py'),'--reuse-masks','--tasks',*(['block_drawer','sort_block','stack_blocks_two','plug_charger'] if a.worker==0 else ['blocks_box']),'--status-name',f'geometry-refined{a.worker}.json']),('latents',[str(REPO/'scripts/geowam/cache_latents.py'),'latents','--worker',str(a.worker),'--workers','2','--wait'])]
env={k:v for k,v in os.environ.items() if not k.startswith('GIT_CONFIG')};env.update(CUDA_VISIBLE_DEVICES=gpu,USE_TF='0',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4')
for phase,cmd in cmds:
 with (ROOT/f'runs/refine-{phase}-{a.worker}.log').open('w') as log:
  child=subprocess.Popen([str(ROOT/'envs/preprocess/bin/python'),*cmd],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=REPO,start_new_session=True)
  state.update(phase=phase,child_pid=child.pid,child_starttime=Path(f'/proc/{child.pid}/stat').read_text().split()[21])
  while child.poll() is None:state['heartbeat']=time.time();atomic_json(status,state);time.sleep(10)
  receipt=json.loads((ROOT/'runs'/(f'geometry-refined{a.worker}.json' if phase=='geometry' else f'cache-latents-{a.worker}.json')).read_text())
  if child.returncode or receipt['phase']!='complete':
   state.update(phase='failed',failed_stage=phase,returncode=child.returncode,receipt=receipt);atomic_json(status,state);raise RuntimeError(state)
state.update(phase='complete',heartbeat=time.time());atomic_json(status,state)
