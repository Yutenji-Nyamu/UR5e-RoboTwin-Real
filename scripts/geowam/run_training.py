"""Scoped two-GPU launcher, progress receipt and bounded checkpoint recovery."""
import argparse,fcntl,json,math,os,shutil,signal,subprocess,time,hashlib
from pathlib import Path
from pipeline_common import ROOT,REPO,CACHE,atomic_json
import yaml
p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','train']);p.add_argument('--resume-check',action='store_true');args=p.parse_args()
GPUS=['GPU-dc5d6921-fa81-b666-bac7-566c126f1dd4','GPU-3c6321c1-3e58-c071-3867-533391152fe7']
lock=open(ROOT/'runs/training.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
audit=json.loads((CACHE/'cache_audit.json').read_text());assert audit['state']=='passed' and audit['episodes']==72
if args.phase=='train':
 assert audit['sha256']==hashlib.sha256((CACHE/'manifests/all.jsonl').read_bytes()).hexdigest()
 assert audit['perception_sha256']==hashlib.sha256((REPO/'configs/geowam/perception.json').read_bytes()).hexdigest()
 assert json.loads((ROOT/'runs/observation_contract_check.json').read_text())['state']=='passed'
 assert json.loads((ROOT/'runs/qa_signoff.json').read_text())['state']=='passed'
cfg=yaml.safe_load((REPO/'configs/geowam/train_rgbd72.yaml').read_text())
cfg['data']['batch_size']=2;cfg['training'].update(grad_accumulation=1,fsdp_reshard_after_forward=False)
if args.phase=='smoke':
 cfg['name']='alpha_fsdp_b2_checkpoint_smoke';cfg['output_dir']=str(ROOT/'runs/smoke_alpha_b2')
 cfg['training'].update(max_steps=3 if args.resume_check else 2,log_every=1,checkpoint_every_minutes=0,checkpoint_every=2,keep_last=2,visualize=True)
 for v in cfg['training']['groups'].values():v.update(freeze_steps=0,ramp_steps=0)
 for k in cfg['training']['dropout']:cfg['training']['dropout'][k]=0.
 cfg['training']['dense_to_compact_steps']=0
else:
 cfg['name']='ur5e_rgbd72_alpha_joint7';cfg['output_dir']=str(ROOT/'runs/train_rgbd72_v1')
run=Path(cfg['output_dir']);run.mkdir(exist_ok=True)
atomic_json(run/'data_audit.json',audit)
config=run/'launch.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
state={'supervisor_pid':os.getpid(),'phase':'starting','mode':args.phase,'gpus':GPUS,'output_dir':str(run),'started':time.time(),'resume_check':args.resume_check,'attempts':[]}
status=ROOT/'runs'/('training_status.json' if args.phase=='train' else 'smoke_status.json');atomic_json(status,state)
for attempt in range(1,4 if args.phase=='train' else 2):
 free=shutil.disk_usage(ROOT).free;assert free>200*1024**3,free
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert not any(line.split(',')[0].strip() in GPUS for line in active.splitlines() if ',' in line),active
 env={k:v for k,v in os.environ.items() if not k.startswith('GIT_CONFIG')};env.update(CUDA_VISIBLE_DEVICES=','.join(GPUS),CUDA_DEVICE_ORDER='PCI_BUS_ID',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',TF_CPP_MIN_LOG_LEVEL='3',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
 env.pop('GEOWAM_ANOMALY',None)
 cmd=[str(ROOT/'envs/preprocess/bin/python'),'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(REPO/'scripts/geowam/train_real.py'),'--config',str(config)]
 logfile=run/f'attempt_{int(time.time())}.log'
 with logfile.open('w') as log:
  child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  receipt={'attempt':attempt,'pid':child.pid,'starttime':Path(f'/proc/{child.pid}/stat').read_text().split()[21],'log':str(logfile),'started':time.time(),'cmd':cmd};state['attempts'].append(receipt);state['phase']='running'
  while child.poll() is None:
   state['heartbeat']=time.time();state['disk_free_gb']=shutil.disk_usage(ROOT).free/1e9
   logs=run/'train_log.jsonl'
   if logs.exists():
    lines=logs.read_text().splitlines()
    if lines:
     try:
      latest=json.loads(lines[-1]);state['last_log']=latest
     except json.JSONDecodeError:pass
   state['checkpoints']=[p.name for p in sorted((run/'checkpoints').glob('step_*')) if (p/'complete.json').exists()]
   atomic_json(status,state);time.sleep(10)
  receipt.update(exit_code=child.returncode,finished=time.time());state['heartbeat']=time.time()
  if child.returncode==0:state['phase']='complete';atomic_json(status,state);break
  state['phase']='retrying' if args.phase=='train' and attempt<3 else 'failed';atomic_json(status,state)
  if args.phase=='train' and attempt<3:time.sleep(30)
print(json.dumps({'phase':state['phase'],'status':str(status),'output_dir':str(run)}),flush=True)
raise SystemExit(0 if state['phase']=='complete' else 1)
