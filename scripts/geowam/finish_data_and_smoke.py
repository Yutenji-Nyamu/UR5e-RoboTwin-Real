"""Wait for the two preprocessing workers, validate every cache, then run the b2 smoke."""
import json,subprocess,time
from pipeline_common import *
status=ROOT/'runs/finish_data_status.json'
while True:
 states=[json.loads((ROOT/'runs'/x).read_text()) for x in ['geometry-worker0.json','cache-latents-0.json']]
 if any(x['errors'] for x in states):raise RuntimeError('preprocessing errors; inspect worker status')
 atomic_json(status,{'phase':'waiting_preprocessing','heartbeat':time.time(),'completed':[len(x.get('completed',x.get('finished_episodes',[]))) for x in states]})
 if all(x['phase']=='complete' for x in states):break
 time.sleep(20)
py=str(ROOT/'envs/preprocess/bin/python');scripts=REPO/'scripts/geowam'
subprocess.run([py,str(scripts/'finalize_cache.py')],check=True)
subprocess.run([py,str(scripts/'qa_overview.py')],check=True)
atomic_json(status,{'phase':'cache_passed_smoke_running','heartbeat':time.time()})
subprocess.run([py,str(scripts/'run_training.py'),'smoke'],check=True)
atomic_json(status,{'phase':'smoke_passed','heartbeat':time.time()})
