"""Single-arm real-recording contract. Canonical data live in the server workspace."""
import csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get('GEOWAM_WORKSPACE','/data/chenyiteng/projects/geowam-real'))
REPO=ROOT/'repos/UR5e-RoboTwin-Real'
CONFIG=REPO/'configs/geowam'
CACHE=ROOT/'cache/real_v1'
TASKS=json.loads((CONFIG/'ur5e_single_arm_contract.json').read_text())['task_instructions']
SLOTS=np.arange(10,17);HORIZON=32;STRIDE=4

def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)

def rows():
 out=[json.loads(x) for x in (ROOT/'data/manifests/train_rgbd.jsonl').read_text().splitlines() if x.strip()]
 assert len(out)==72 and all(x['has_depth'] and x['train'] and x['split']=='train' for x in out)
 return out

def perception_signature(cfg,task,key=None):
 payload={k:v for k,v in cfg.items() if k not in ("tasks","episode_boxes","episode_repairs")};payload["prompts"]=cfg["tasks"][task]
 if key in cfg.get("episode_boxes",{}):payload["episode_boxes"]=cfg["episode_boxes"][key]
 if key in cfg.get("episode_repairs",{}):payload["episode_repairs"]=cfg["episode_repairs"][key]
 return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def csvrows(path):
 with Path(path).open() as f:return list(csv.DictReader(f))

def episode_meta(row):
 ep=ROOT/'data/raw'/row['path'];sync=csvrows(ep/'sync.csv');action=csvrows(ep/'action.csv')
 times=np.array([float(x['controller_time_s']) for x in action]);st=np.array([float(x['controller_time_s']) for x in sync])
 order=np.searchsorted(times,st);order=np.minimum(order,len(times)-1)
 prev=np.maximum(order-1,0);order=np.where(abs(times[prev]-st)<abs(times[order]-st),prev,order)
 assert np.max(abs(times[order]-st))<.001
 state=np.array([[*[float(action[i][f'actual_q_{k}']) for k in range(6)],float(action[i]['gripper_state'])] for i in order],np.float32)
 assert np.isfinite(state).all() and set(np.unique(state[:,6]))<={0.,1.}
 names=np.array([x['head_image'] for x in sync]);assert all(x['head_image']==x['wrist_image'] for x in sync)
 n=len(sync);starts=[s for s in range(0,n-HORIZON,STRIDE) if np.all((np.diff(st[s:s+33])>.05)&(np.diff(st[s:s+33])<.15))]
 assert starts
 cal=json.loads((ep/'images/camera_calibration.json').read_text());scale=cal['cameras']['head']['depth_scale_m_per_unit']
 return dict(state=state,time=st,names=names,starts=np.array(starts),keyframes=np.arange(0,n,STRIDE),depth_scale=np.float64(scale))

def paths_ready(row,meta):
 ep=ROOT/'data/raw'/row['path']/'images'
 return all((ep/c/str(n)).is_file() for c in ['head','wrist','head_depth'] for n in meta['names'])

def read_rgb(row,name,camera='head'):
 from PIL import Image
 return np.array(Image.open(ROOT/'data/raw'/row['path']/'images'/camera/str(name)).convert('RGB'))

def read_depth(row,name,scale):
 from PIL import Image
 return np.asarray(Image.open(ROOT/'data/raw'/row['path']/'images/head_depth'/str(name)),dtype=np.float32)*scale

def normalise(x,stats,kind,inverse=False):
 mean=np.array(stats[kind+'_mean'],np.float32);std=np.array(stats[kind+'_std'],np.float32)
 return x*std+mean if inverse else (x-mean)/std

def targets(meta,s,stats):
 q=meta['state'];action=q[s+1:s+33].copy();action[:,:6]-=q[s,:6]
 a=np.zeros((32,80),np.float32);a[:,SLOTS]=normalise(action,stats,'action')
 p=np.zeros((1,80),np.float32);p[0,SLOTS]=normalise(q[s],stats,'state')
 mask=np.zeros(80,bool);mask[SLOTS]=True
 restored=normalise(a[:,SLOTS],stats,'action',True);restored[:,:6]+=q[s,:6]
 assert np.max(abs(restored-q[s+1:s+33]))<2e-6
 return dict(action=a,proprio=p,action_mask=np.broadcast_to(mask,a.shape).copy(),proprio_mask=mask[None].copy())

def prepare_metadata():
 CACHE.mkdir(parents=True,exist_ok=True);allstates=[];allactions=[];reports=[]
 for row in rows():
  m=episode_meta(row);ep=CACHE/row['key'];ep.mkdir(exist_ok=True);np.savez_compressed(ep/'metadata.npz',**m)
  for s in m['starts']:
   action=m['state'][s+1:s+33].copy();action[:,:6]-=m['state'][s,:6];allactions.append(action);allstates.append(m['state'][s])
  reports.append({**row,'frames':len(m['names']),'keyframes':len(m['keyframes']),'windows':len(m['starts']),'gaps':int((np.diff(m['time'])>=.15).sum()),'images_ready':paths_ready(row,m)})
 st=np.stack(allstates);ac=np.concatenate(allactions);stats={}
 for name,arr in [('state',st),('action',ac)]:
  stats[name+'_mean']=arr.mean(0).tolist();stats[name+'_std']=np.maximum(arr.std(0),1e-3).tolist()
  stats[name+'_mean'][6]=.5;stats[name+'_std'][6]=.5
 stats.update(version='ur5e_joint7_rgbd_h32_v1',episodes=72,slots=SLOTS.tolist(),horizon=32,hz=10,method='z-score joint delta and state; gripper 2*g-1')
 atomic_json(CACHE/'normalization.json',stats);atomic_json(CACHE/'depth_stats.json',{'head_camera':{'min_m':.2,'max_m':1.6},'choice':'fixed tabletop metric range; invalid depth is black'})
 for row in rows():
  m=np.load(CACHE/row['key']/'metadata.npz')
  for s in m['starts']:targets(m,int(s),stats)
 atomic_json(CACHE/'episodes.json',reports)
 atomic_json(ROOT/'runs/preprocessing_metadata.json',{'episodes':72,'frames':sum(x['frames'] for x in reports),'keyframes':sum(x['keyframes'] for x in reports),'windows':sum(x['windows'] for x in reports),'images_ready':sum(x['images_ready'] for x in reports),'action_roundtrip':'passed every window','by_task':{k:sum(x['task']==k for x in reports) for k in TASKS}})
 print((ROOT/'runs/preprocessing_metadata.json').read_text())
if __name__=='__main__':prepare_metadata()
