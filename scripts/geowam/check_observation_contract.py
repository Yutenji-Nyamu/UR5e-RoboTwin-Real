"""Verify raw current RGB-D/state produces the same causal conditions as the training cache."""
import os,sys,time,json,subprocess
from pathlib import Path
from pipeline_common import *
while True:
 p=ROOT/'runs/refine-worker1.json'
 if p.exists() and json.loads(p.read_text())['phase']=='complete':break
 time.sleep(10)
gpu=os.environ['CUDA_VISIBLE_DEVICES']
while any(l.split(',')[0].strip()==gpu for l in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines() if ',' in l):time.sleep(5)
import torch,cv2
sys.path.insert(0,str(ROOT/'repos/MetisWAM4D'))
from metiswam4d.data.rt2.online_encoder import RT2OnlineEncoder,RT2EncoderConfig,center_pad
torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.manual_seed(123)
enc=RT2OnlineEncoder(RT2EncoderConfig(vae_path=str(ROOT/'models/wan22-ti2v-5b'),depth_stats=str(CACHE/'depth_stats.json')),torch.device('cuda'))
result=[];out=ROOT/'runs/observation_contract';out.mkdir(exist_ok=True)
for task in TASKS:
 row=next(x for x in rows() if x['task']==task);ep=CACHE/row['key']
 # Other worker may still be caching this episode; wait for its current geometry signature.
 while True:
  rep=json.loads((ep/'latents.json').read_text());want=__import__('hashlib').sha256((ep/'geometry.npz').read_bytes()+(CACHE/'normalization.json').read_bytes()+b'latent_contract_v1').hexdigest()
  if rep.get('input_signature')==want:break
  time.sleep(5)
 meta=np.load(ep/'metadata.npz');g=np.load(ep/'geometry.npz');s=int(meta['starts'][len(meta['starts'])//2]);name=meta['names'][s]
 entry=next(json.loads(l) for l in (ep/'manifest.jsonl').read_text().splitlines() if json.loads(l)['key']==row['key']+f'/s{s}');cached=torch.load(entry['path'],map_location='cpu',weights_only=True)
 rgb=np.concatenate([cv2.resize(read_rgb(row,name,v),(320,192),interpolation=cv2.INTER_AREA) for v in ['head','wrist']],axis=0)
 current=torch.from_numpy(rgb)[None,None];vid=enc.encode_pixels(current).cpu()[0]
 anchor=np.zeros((240,320,3),np.uint8);anchor[g['foreground'][s//4]]=128;anchor_t=center_pad(torch.from_numpy(anchor)[None,None],256,320);track=enc.encode_pixels(anchor_t).cpu()[0]
 head=cv2.resize(read_rgb(row,name),(320,240),interpolation=cv2.INTER_AREA);depth=cv2.resize(read_depth(row,name,float(meta['depth_scale'])),(320,240),interpolation=cv2.INTER_NEAREST);mask=g['foreground'][s//4]
 cond=enc.encode_pixels(enc.condition_pixels(torch.from_numpy(head)[None],torch.from_numpy(depth*1000)[None],torch.from_numpy(mask)[None])).cpu()
 diffs={}
 for tag,x,y in [('video',vid,cached['video_clean'][:,:1]),('track_anchor',track,cached['track_clean'][:,:1]),*[(k,cond[i],cached[k]) for i,k in enumerate(['rgb_condition','depth_condition','mask_condition'])]]:
  err=(x.float()-y.float());diffs[tag]={'max_abs':float(err.abs().max()),'rmse':float(err.square().mean().sqrt())};assert diffs[tag]['rmse']<.005 and diffs[tag]['max_abs']<.08,(task,tag,diffs[tag])
 noisy=torch.cat([current,torch.randint(0,256,(1,8,384,320,3),dtype=torch.uint8)],dim=1);future=enc.encode_pixels(noisy).cpu()[0,:,:1];err=(vid.float()-future.float());diffs['future_replacement']={'max_abs':float(err.abs().max()),'rmse':float(err.square().mean().sqrt())};assert diffs['future_replacement']['max_abs']<.08 and diffs['future_replacement']['rmse']<.005
 request={'video_first_frame':vid,'track_anchor':track,'rgb_condition':cond[0],'depth_condition':cond[1],'mask_condition':cond[2],'text_context':cached['text_context'],'proprio':cached['proprio'],'proprio_mask':cached['proprio_mask'],'embodiment':torch.tensor(9),'task':task,'episode':row['key'],'frame':s,'action_horizon':32,'hz':10}
 torch.save(request,out/f'{task}.pt');result.append({'task':task,'episode':row['key'],'frame':s,'differences':diffs});print('CAUSAL_INPUT_PASS',task,diffs,flush=True)
 atomic_json(ROOT/'runs/observation_contract_check.json',{'state':'running','tasks':result,'heartbeat':time.time()})
atomic_json(ROOT/'runs/observation_contract_check.json',{'state':'passed','tasks':result,'heartbeat':time.time(),'source':'current RGB-D, mask, state and instruction; recorded future replaced with independent random frames'})
