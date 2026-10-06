"""Minute GPU records and hourly training summaries for an unattended focus run."""
import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args()
    history=deque(maxlen=360)
    last_hour=0
    with (args.run/'health.jsonl').open('a',buffering=1) as log:
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                         '--format=csv,noheader,nounits'],text=True)
            gpus=[list(map(int,line.split(','))) for line in raw.strip().splitlines()]
            history.append(gpus)
            rows=[]
            path=args.run/'train_log.jsonl'
            if path.exists():
                for line in path.read_text().splitlines():
                    try: rows.append(json.loads(line))
                    except json.JSONDecodeError: pass
            record=dict(time=time.time(),gpus=gpus,training_last=rows[-1] if rows else None,
                        observed_minutes=len(history),
                        mean_gpu_util=[sum(sample[i][3] for sample in history)/len(history) for i in range(len(gpus))])
            (args.run/'health_latest.json').write_text(json.dumps(record,indent=2))
            log.write(json.dumps(record)+'\n')
            now=time.time()
            if now-last_hour>=3600:
                checkpoints=sorted(p.name for p in (args.run/'checkpoints').glob('step_*') if (p/'complete.json').exists())
                vals=[r for r in rows if any(k.startswith('val/') for k in r)]
                hourly={**record,'complete_checkpoints':checkpoints,'latest_validation':vals[-1] if vals else None}
                with (args.run/'hourly_health.jsonl').open('a') as f:f.write(json.dumps(hourly)+'\n')
                last_hour=now
            status=args.run/'focus_pipeline_status.json'
            if status.exists() and json.loads(status.read_text()).get('stage')=='round_complete': break
            exit_file=args.run/'train_exit_code'
            if exit_file.exists() and exit_file.read_text().strip()!='0': break
            time.sleep(60)


if __name__=='__main__':
    main()
