"""Restore repository release images; token is read once from stdin, kept in memory."""
import argparse,concurrent.futures,hashlib,json,os,shutil,sys,tarfile,threading,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
root=a.workspace.resolve();repo=root/'repos/data_ur5e_gwam';dest=root/'data/raw';cache=root/'cache/data_archives'
dest.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
token=sys.stdin.readline().strip();assert token
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None
api=urllib.request.build_opener(NoRedirect)
def api_json(url):
 req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'geowam-data-restore'})
 with urllib.request.urlopen(req,timeout=60) as f:return json.load(f)
manifest=json.loads((repo/'downloads.json').read_text());remote={}
for tag in manifest['release_tags']:
 release=api_json('https://api.github.com/repos/'+manifest['repository']+'/releases/tags/'+tag)
 remote.update({x['name']:x for x in release['assets']})
for item in repo.iterdir():
 if item.name=='.git':continue
 target=dest/item.name
 if item.is_dir():shutil.copytree(item,target,dirs_exist_ok=True)
 elif item.is_file():shutil.copy2(item,target)
state={};lock=threading.Lock();status=root/'runs/data-download-status.json'
def update(name,**fields):
 with lock:
  state.setdefault(name,{}).update(fields)
  temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(status)
def get_signed(asset,offset=0):
 req=urllib.request.Request(asset['url'],headers={'Authorization':'Bearer '+token,'Accept':'application/octet-stream','User-Agent':'geowam-data-restore'})
 try:return api.open(req,timeout=60)
 except urllib.error.HTTPError as e:
  if e.code not in (301,302,303,307,308):raise
  location=e.headers['Location'];assert location.startswith('https://')
  return urllib.request.urlopen(urllib.request.Request(location,headers={'Range':f'bytes={offset}-'}),timeout=60)
def restore(item):
 name=item['name'];asset=remote[name];assert asset['size']==item['bytes']
 archive=cache/name;partial=cache/(name+'.part');update(name,state='downloading',expected_bytes=item['bytes'],started=time.time())
 try:
  if not archive.exists() or archive.stat().st_size!=item['bytes']:
   for attempt in range(5):
    try:
     copied=partial.stat().st_size if partial.exists() else 0
     if copied==item['bytes']:break
     with get_signed(asset,copied) as response:
      if response.status==206:
       assert response.headers.get('Content-Range','').startswith(f'bytes {copied}-');mode='ab'
      else:mode='wb';copied=0
      with partial.open(mode) as out:
       last=0
       while chunk:=response.read(4*1024*1024):
        out.write(chunk);copied+=len(chunk)
        if time.time()-last>10:update(name,downloaded_bytes=copied);last=time.time()
     assert partial.stat().st_size==item['bytes'];break
    except Exception:
     if attempt==4:raise
     time.sleep(2*(attempt+1))
   assert partial.stat().st_size==item['bytes'];partial.replace(archive)
  digest=hashlib.sha256()
  with archive.open('rb') as f:
   while chunk:=f.read(8*1024*1024):digest.update(chunk)
  sha=digest.hexdigest();expected=asset.get('digest')
  if expected:assert expected=='sha256:'+sha
  update(name,state='extracting',sha256=sha,github_digest_verified=bool(expected))
  with tarfile.open(archive,'r:') as tf:
   for member in tf:
    target=(dest/member.name).resolve();assert target.is_relative_to(dest)
    assert member.isfile() or member.isdir()
    if member.isfile():
     target.parent.mkdir(parents=True,exist_ok=True)
     with tf.extractfile(member) as src,target.open('wb') as out:shutil.copyfileobj(src,out,1024*1024)
    else:target.mkdir(parents=True,exist_ok=True)
  update(name,state='complete',finished=time.time());print(name,'complete',flush=True)
 except Exception as e:update(name,state='error',error=type(e).__name__+': '+str(e).split('?')[0]);print(name,'error',type(e).__name__,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(restore,manifest['assets']))
