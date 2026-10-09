"""Download pinned preprocessing weights and verify their upstream digests."""
import argparse,concurrent.futures,hashlib,json,threading,time,urllib.request,urllib.parse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
r=a.workspace.resolve();state={};lock=threading.Lock();status=r/'runs/processing-models-status.json'
def update(name,**fields):
 with lock:
  state.setdefault(name,{}).update(fields)
  tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def fetch(job):
 name,url,dest,size,sha=job;dest.parent.mkdir(parents=True,exist_ok=True);part=dest.with_suffix(dest.suffix+'.part')
 try:
  if not dest.exists():
   for attempt in range(4):
    try:
     offset=part.stat().st_size if part.exists() else 0
     if size and offset==size:break
     req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-','User-Agent':'geowam-preprocessing-models'})
     with urllib.request.urlopen(req,timeout=90) as response:
      if response.status==206:
       if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('range mismatch')
       mode='ab'
      else:mode='wb';offset=0
      with part.open(mode) as out:
       last=0
       while chunk:=response.read(8*1024*1024):
        out.write(chunk);offset+=len(chunk)
        if time.time()-last>5:update(name,state='downloading',bytes=offset,expected_bytes=size);last=time.time()
     if size and part.stat().st_size!=size:raise ValueError('size mismatch')
     break
    except Exception:
     if attempt==3:raise
     time.sleep(2*(attempt+1))
   part.replace(dest)
  h=hashlib.sha256()
  with dest.open('rb') as f:
   while chunk:=f.read(8*1024*1024):h.update(chunk)
  digest=h.hexdigest()
  if sha and not digest.startswith(sha):
   dest.rename(dest.with_suffix(dest.suffix+'.sha-mismatch-'+str(int(time.time()))))
   raise ValueError('sha256 mismatch')
  update(name,state='complete',bytes=dest.stat().st_size,sha256=digest,source=url,finished=time.time())
  print(name,'complete',flush=True)
 except Exception as e:update(name,state='error',error=type(e).__name__);print(name,'error',type(e).__name__,flush=True)
ms=json.loads((Path(__file__).resolve().parents[2]/'configs/geowam/model_sources/modelscope_sam31_files.json').read_text())['Data']['Files']
jobs=[]
for f in ms:
 if f['Type']!='blob' or f['Path'].startswith('assets/') or f['Path']=='.gitattributes':continue
 u='https://www.modelscope.cn/api/v1/models/facebook/sam3.1/repo?'+urllib.parse.urlencode({'Revision':f['Revision'],'FilePath':f['Path']})
 jobs.append(('sam31/'+f['Path'],u,r/'models/sam3.1'/f['Path'],f['Size'],f['Sha256']))
jobs.append(('raft','https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth',r/'models/raft/raft_large_C_T_SKHT_V2-ff5fadd5.pth',None,'ff5fadd5'))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(fetch,jobs))
