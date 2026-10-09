import argparse,concurrent.futures,hashlib,json,os,time,urllib.request,threading
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();r=a.workspace
url='https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-Single-Arm-Franka/resolve/e75522eb38f9b4148d5ecb985ab80c490ddb3854/checkpoint_step_9860.safetensors'
size=24813767464;sha='76e53e963525d9eebdb70e5285908c6497082270de66f9227874844e96bc43bb'
dest=r/'models/alpha-franka/checkpoint_step_9860.safetensors';part=dest.with_suffix('.safetensors.part');parts=r/'cache/alpha_ranges';parts.mkdir(exist_ok=True)
offset=part.stat().st_size if part.exists() else 0
lock=threading.Lock();state={'state':'downloading','started':time.time(),'pid':os.getpid(),'base_bytes':offset,'expected_bytes':size,'parts':{}};status=r/'runs/alpha-parallel-status.json'
def update():
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
update()
def fetch(bounds):
 lo,hi=bounds;path=parts/f'{lo:012d}-{hi:012d}.bin';expected=hi-lo+1
 for attempt in range(8):
  have=path.stat().st_size if path.exists() else 0
  if have==expected:break
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={lo+have}-{hi}','User-Agent':'geowam-alpha-resume'})
   with urllib.request.urlopen(req,timeout=60) as resp:
    assert resp.status==206 and resp.headers['Content-Range'].startswith(f'bytes {lo+have}-'), 'range'
    with path.open('ab') as f:
     while chunk:=resp.read(min(1024*1024,expected-have)):
      f.write(chunk);have+=len(chunk)
      with lock:state['parts'][str(lo)]={'bytes':have,'expected':expected};update()
      if have>=expected:break
   assert have==expected;break
  except Exception as e:
   with lock:state['parts'][str(lo)]={'bytes':have,'expected':expected,'attempt':attempt,'error':type(e).__name__};update()
   if attempt==7:raise
   time.sleep(2+attempt)
 return path
try:
 if not dest.exists():
  chunk=256*1024*1024;jobs=[(s,min(size-1,s+chunk-1)) for s in range(offset,size,chunk)]
  with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:files=list(pool.map(fetch,jobs))
  with part.open('ab') as out:
   for path in files:
    with path.open('rb') as f:
     while b:=f.read(8*1024*1024):out.write(b)
  assert part.stat().st_size==size
  state['state']='verifying';update();h=hashlib.sha256()
  with part.open('rb') as f:
   while b:=f.read(8*1024*1024):h.update(b)
  assert h.hexdigest()==sha;part.replace(dest)
 state.update(state='complete',sha256=sha,finished=time.time());update()
except Exception as e:
 state.update(state='error',error=type(e).__name__+': '+str(e).split('?')[0]);update();raise
