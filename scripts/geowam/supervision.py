"""CPU-only progress serialization and exact Linux process identity checks."""
import json, math, os, shutil, time
from pathlib import Path

def serializable_log(record):
    out=dict(record);unexpected=[]
    for key,value in record.items():
        if isinstance(value,float) and not math.isfinite(value):
            out[key]=str(value)
            if not (key=='anneal/dense_bias' and value==float('-inf')):
                unexpected.append(key)
    if unexpected:out['unexpected_nonfinite_metrics']=unexpected
    return out

def process_identity(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        return {'pid':int(pid),'starttime':fields[19],'state':fields[0],
                'ppid':int(fields[1]),'uid':int((p/'status').read_text().split('Uid:')[1].split()[0]),
                'cmd':(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')}
    except (FileNotFoundError,ProcessLookupError):return None

def identity_matches(actual,expected):
    return bool(actual and actual['state']!='Z' and all(actual[k]==expected[k] for k in ('pid','starttime','uid','cmd')))

def refresh_progress(state,run,workspace):
    state['heartbeat']=time.time();state['disk_free_gb']=shutil.disk_usage(workspace).free/1e9
    logs=Path(run)/'train_log.jsonl'
    if logs.exists():
        for line in reversed(logs.read_text().splitlines()):
            try:latest=json.loads(line)
            except json.JSONDecodeError:continue
            state['last_log']=serializable_log(latest);break
    state['checkpoints']=[p.name for p in sorted((Path(run)/'checkpoints').glob('step_*')) if (p/'complete.json').exists()]
    return state

def completed_run(run,max_steps):
    for p in (Path(run)/'checkpoints').glob('step_*/complete.json'):
        try:
            if int(json.loads(p.read_text())['step'])>=max_steps:return True
        except (ValueError,KeyError,json.JSONDecodeError):continue
    return False
