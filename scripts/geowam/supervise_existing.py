"""Adopt this run's verified torchrun process without restarting its workers."""
import argparse, fcntl, json, os, subprocess, time
from pathlib import Path
import yaml
from pipeline_common import ROOT,REPO,atomic_json
from supervision import process_identity,identity_matches,refresh_progress,completed_run

def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--starttime',required=True);a=p.parse_args()
    lock=open(ROOT/'runs/training.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=ROOT/'runs/training_status.json';state=json.loads(path.read_text());run=Path(state['output_dir'])
    assert run.resolve()==(ROOT/'runs/train_rgbd72_v1').resolve()
    previous=process_identity(state['supervisor_pid']);assert previous is None,'Existing supervisor still has a process identity'
    receipt=state['attempts'][-1];expected={'pid':a.pid,'starttime':a.starttime,'uid':os.getuid(),'cmd':receipt['cmd']}
    assert receipt['pid']==a.pid and receipt['starttime']==a.starttime
    assert identity_matches(process_identity(a.pid),expected),'Torchrun identity changed'
    workers=[]
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
        uuid,pid=[x.strip() for x in line.split(',')]
        if uuid not in state['gpus']:continue
        identity=process_identity(int(pid));assert identity and identity['uid']==os.getuid() and identity['ppid']==a.pid
        assert str(REPO/'scripts/geowam/train_real.py') in identity['cmd'] and str(run/'launch.yaml') in identity['cmd']
        workers.append(dict(identity,gpu_uuid=uuid))
    assert len(workers)==2 and {w['gpu_uuid'] for w in workers}==set(state['gpus'])
    backup=run/f'supervision_before_adoption_{int(time.time())}.json';atomic_json(backup,state)
    state.setdefault('supervisor_history',[]).append({'pid':state['supervisor_pid'],'last_heartbeat':state.get('heartbeat'),'reason':'compact dense_bias -inf was rejected by strict status JSON'})
    state.update(supervisor_pid=os.getpid(),supervisor_starttime=process_identity(os.getpid())['starttime'],phase='running',adopted_at=time.time(),worker_identities=workers)
    max_steps=int(yaml.safe_load((run/'launch.yaml').read_text())['training']['max_steps'])
    while identity_matches(process_identity(a.pid),expected):
        refresh_progress(state,run,ROOT);atomic_json(path,state);time.sleep(10)
    refresh_progress(state,run,ROOT);receipt['observed_finished']=time.time();receipt['exit_code']=None
    if completed_run(run,max_steps):
        state['phase']='complete';state['completion_evidence']='complete checkpoint reached configured max_steps';atomic_json(path,state);return
    state['phase']='retrying';atomic_json(path,state)
    # The regular launcher rechecks GPU occupancy and all data guards before resume.
    for _ in range(6):
        if not any(identity_matches(process_identity(w['pid']),w) for w in workers):break
        time.sleep(10)
    lock.close()
    py=str(ROOT/'envs/preprocess/bin/python')
    os.execv(py,[py,str(REPO/'scripts/geowam/run_training.py'),'train'])

if __name__=='__main__':main()
