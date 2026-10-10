import subprocess,json,time
from pathlib import Path
out=Path('/data/chenyiteng/projects/geowam-real/runs/performance_review_20261010')
keys=['index','uuid','memory.total','memory.used','memory.free','utilization.gpu','utilization.memory','power.draw','power.limit','clocks.sm','temperature.gpu']
rows=[];start=time.time()
for n in range(180):
 t=time.time();lines=subprocess.check_output(['nvidia-smi','--id=6,7','--query-gpu='+','.join(keys),'--format=csv,noheader,nounits'],text=True)
 for line in lines.splitlines():
  v=dict(zip(keys,[x.strip() for x in line.split(',')]));v.update(time=t,sample=n);rows.append(v)
 delay=start+n+1-time.time()
 if delay>0:time.sleep(delay)
(out/'gpu_180s.json').write_text(json.dumps({'started':start,'finished':time.time(),'samples':rows},indent=2)+'\n')
print('DONE',len(rows),flush=True)
