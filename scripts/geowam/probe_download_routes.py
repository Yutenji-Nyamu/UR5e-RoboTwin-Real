import json,sys,time,urllib.request,urllib.error,concurrent.futures
from pathlib import Path
token=sys.stdin.readline().strip()
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**k):return None
proxy={"http":"http://127.0.0.1:7890","https":"http://127.0.0.1:7890"}
api=urllib.request.build_opener(urllib.request.ProxyHandler(proxy),NoRedirect)
manifest=json.loads(Path("/data/chenyiteng/projects/geowam-real/repos/data_ur5e_gwam/downloads.json").read_text())
req=urllib.request.Request("https://api.github.com/repos/"+manifest["repository"]+"/releases/tags/"+manifest["release_tags"][-1],headers={"Authorization":"Bearer "+token,"User-Agent":"route-probe"})
with api.open(req,timeout=20) as f:d=json.load(f)
req=urllib.request.Request(d["assets"][0]["url"],headers={"Authorization":"Bearer "+token,"Accept":"application/octet-stream","User-Agent":"route-probe"})
try:api.open(req,timeout=20)
except urllib.error.HTTPError as e:
 assert e.code==302;url=e.headers["Location"]
alpha="https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-Single-Arm-Franka/resolve/e75522eb38f9b4148d5ecb985ab80c490ddb3854/checkpoint_step_9860.safetensors"
def test(item):
 name,url,useproxy=item;t=time.time();n=0
 try:
  op=urllib.request.build_opener(urllib.request.ProxyHandler(proxy if useproxy else {}))
  with op.open(urllib.request.Request(url,headers={"Range":"bytes=0-8388607"}),timeout=12) as f:
   while n<8388608 and time.time()-t<20:
    c=f.read(262144)
    if not c:break
    n+=len(c)
  return {"name":name,"proxy":useproxy,"bytes":n,"seconds":time.time()-t}
 except Exception as e:return {"name":name,"proxy":useproxy,"bytes":n,"seconds":time.time()-t,"error":type(e).__name__}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:out=list(ex.map(test,[("data",url,False),("data",url,True),("alpha",alpha,False),("alpha",alpha,True)]))
Path("/data/chenyiteng/projects/geowam-real/runs/download-route-probe.json").write_text(json.dumps(out,indent=2))
