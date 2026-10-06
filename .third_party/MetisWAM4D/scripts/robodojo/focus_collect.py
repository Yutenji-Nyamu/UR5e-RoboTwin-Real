"""Collect round-2 data from remaining seed-1/2 layouts, then seed-0 layouts >=40.

Run inside tmux. Existing collection results seed new, isolated resume ledgers;
only new trajectories are recorded. Managers receive TERM directly at the time
limit, so no shell retry loop can restart a stopped collection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from metiswam4d.eval.rdj_campaign import prepare, read_details, result_file, save_dir, write_json
from scripts.robodojo.focus_after_train import TASK_CONFIGS, REFERENCE

ACTIVATE='/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/files/RoboDojo/scripts/activate_robodojo.sh'


def ensure_keepalive():
    pattern=r'^(/usr/bin/)?python(3(\.10)?)? (/m2v_intern_v3/danglingwei/)?wangrunqi_nvml_busy.py --gpus all'
    if subprocess.run(['pgrep','-f',pattern],capture_output=True).returncode==0:return
    stamp=time.strftime('%Y%m%d_%H%M%S')
    logs=Path('/m2v_intern_v3/danglingwei/logs')
    subprocess.Popen(['/usr/bin/python3','/m2v_intern_v3/danglingwei/wangrunqi_nvml_busy.py','--gpus','all','--size','2000',
                      '--pid-file',str(logs/f't2_after_collect_{stamp}.pid'),'--log-file',str(logs/f't2_after_collect_{stamp}.log')],
                     stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)


def prepare_collection(config, model, out, record, seed, policy, prior=None):
    campaign=prepare(out=out,config=str(config),model_file=str(model),source_checkpoint=str(model),
                     episodes_per_round=5,execute_steps=32,clients_per_gpu=2,policy=policy,info=out.name,
                     only=TASK_CONFIGS,episodes=1000,video_episodes=0,policy_seed=42,
                     layout_offset=40 if seed==0 else 0,record_dir=str(record),env_seed=seed)
    copied={}
    if prior:
        for variant in campaign['variants']:
            source=result_file(prior,variant)
            if source is None:continue
            destination=save_dir(out,campaign,variant)/'_result.json'
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():destination.write_text(source.read_text())
            copied[variant['name']]=[int(r['layout_id']) for r in read_details(prior,variant)]
    write_json(out/'collection_provenance.json',dict(model=str(model),policy=policy,env_seed=seed,
                                                   prior=str(prior) if prior else None,copied_layout_ids=copied,
                                                   record_dir=str(record)))
    return campaign


def run_phase(jobs, seconds):
    # The activation script prepares the existing Isaac runtime. exec preserves
    # the Popen PID as the manager PID for exact, graceful termination.
    shell='source "$1" || exit 2; cd "$2" || exit 2; export PYTHONPATH="$2"; exec /usr/bin/python3.10 -m metiswam4d.eval.rdj_campaign run --output "$3" --gpus 0,1,2,3,4,5,6,7 --clients "$4" --base-port "$5"'
    deadline=time.monotonic()+seconds
    active=[]
    def launch(job):
        out,clients,port=job
        log=(out/'collection_manager.log').open('a')
        proc=subprocess.Popen(['bash','-c',shell,'focus-manager',ACTIVATE,str(PROJECT),str(out),str(clients),str(port)],
                              stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        log.close()
        print(json.dumps(dict(event='manager_start',pid=proc.pid,output=str(out),clients=clients,port=port)),flush=True)
        return proc
    try:
        for job in jobs:
            active.append(dict(job=job,process=launch(job),failures=0,done=False))
        while time.monotonic()<deadline and not all(x['done'] for x in active):
            for item in active:
                proc=item['process']
                if item['done'] or proc.poll() is None:continue
                if proc.returncode==0:
                    item['done']=True
                else:
                    item['failures']+=1
                    if item['failures']>=3:
                        raise RuntimeError(f"Repeated manager failure: {item['job'][0]}")
                    item['process']=launch(item['job'])
            time.sleep(10)
    finally:
        for item in active:
            proc=item['process']
            if proc.poll() is None:proc.terminate()
        for item in active:
            proc=item['process']
            try:proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                # Leave its handle visible for diagnosis instead of orphaning clients.
                raise RuntimeError(f'Manager {proc.pid} did not finish TERM cleanup')
        ensure_keepalive()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--record-root',type=Path,required=True)
    ap.add_argument('--hours',type=float,default=4.0)
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    def interrupted(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM,interrupted)
    if os.uname().nodename.split('.')[0]!='a800bcctest0080-bd':
        raise RuntimeError('Only t2 is authorized')
    run=Path(yaml.safe_load(args.config.read_text())['output_dir'])
    status=json.loads((run/'focus_pipeline_status.json').read_text())
    if status.get('stage')!='round_complete' or not status['decision'].startswith('round2_'):
        raise RuntimeError('Round 1 has not qualified for another collection round')
    policy={'hide_track':True} if status['best'].endswith('/hide') else {}
    model=run/'simeval_focus/checkpoint/model_bf16_tail3.pt'
    if not model.exists():raise FileNotFoundError(model)
    root=run/'selfplay_round2'
    campaigns=[]
    for seed,previous in [(1,'selfplay_collect_v2'),(2,'selfplay_collect_v2b'),(0,None)]:
        out=root/f'seed{seed}'
        prior=REFERENCE.parent/previous if previous else None
        campaign=prepare_collection(args.config.resolve(),model,out,args.record_root/f'seed{seed}',seed,policy,prior)
        campaigns.append((out,campaign))
    write_json(root/'plan.json',dict(policy=policy,model=str(model),hours=args.hours,
                                   phases=['seed1 + seed2 (75% of time)','seed0 layout >=40 (remaining time)']))
    if args.prepare_only:return
    if subprocess.run(['pgrep','-f','^/usr/bin/python3.10 -m torch.distributed.run'],capture_output=True).returncode==0:
        raise RuntimeError('Training is still running')
    started=time.monotonic()
    budget=args.hours*3600
    # Two services plus three recording clients per GPU fit below the earlier
    # observed OOM combination of two services plus five clients.
    run_phase([(root/'seed1',2,33980),(root/'seed2',1,34080)],budget*.75)
    run_phase([(root/'seed0',3,34180)],max(1,budget-(time.monotonic()-started)))
    delta={}
    for out,campaign in campaigns:
        provenance=json.loads((out/'collection_provenance.json').read_text())
        per_task={}
        for v in campaign['variants']:
            old=set(provenance['copied_layout_ids'].get(v['name'],[]))
            rows=[r for r in read_details(out,v) if int(r['layout_id']) not in old]
            per_task[v['name']]=dict(episodes=len(rows),successes=sum(bool(r['success']) for r in rows))
        delta[out.name]=per_task
    write_json(root/'collection_delta.json',delta)
    print(json.dumps(dict(event='collection_complete',elapsed_seconds=time.monotonic()-started,delta=delta)),flush=True)


if __name__=='__main__':
    main()
