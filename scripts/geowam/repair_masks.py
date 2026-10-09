"""Re-ground only inspected SAM dropouts; preserve visible-label provenance and instance IDs."""
import argparse,json,os,time,subprocess,hashlib
from pathlib import Path
import numpy as np
from pipeline_common import *
p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--keys',nargs='*');p.add_argument('--share-preprocess',action='store_true');a=p.parse_args()
# Wait for the validated two-GPU resume smoke to finish before using physical GPU 6.
gpu=os.environ['CUDA_VISIBLE_DEVICES']
if a.share_preprocess:
 for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
  if ',' in line and line.split(',')[0].strip()==gpu:
   proc=Path('/proc')/line.split(',')[1].strip();cmd=(proc/'cmdline').read_bytes().decode().replace('\0',' ')
   assert proc.stat().st_uid==os.getuid() and str(REPO/'scripts/geowam/cache_latents.py') in cmd,cmd
else:
 while any(line.split(',')[0].strip()==gpu for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines() if ',' in line):time.sleep(10)
import torch,cv2
from PIL import Image,ImageDraw
from sam3.model_builder import build_sam3_multiplex_video_predictor
torch.set_num_threads(4);torch.set_num_interop_threads(2)
pred=build_sam3_multiplex_video_predictor(checkpoint_path=str(ROOT/'models/sam3.1/sam3.1_multiplex.pt'),compile=False,use_fa3=False,use_rope_real=False,warm_up=False,async_loading_frames=False)
init=pred.model.init_state
def compat(*a,**k):k.pop('offload_state_to_cpu',None);return init(*a,**k)
pred.model.init_state=compat;pred.model.use_batched_grounding=False
cfg=json.loads((CONFIG/'perception.json').read_text());out=ROOT/'evidence/mask_repairs';out.mkdir(parents=True,exist_ok=True)
report=[];panels=[];status=ROOT/'runs/mask_repairs_status.json'
for key,repairs in cfg['episode_repairs'].items():
 if a.keys and key not in a.keys:continue
 ep=CACHE/key;row=next(x for x in rows() if x['key']==key);meta=np.load(ep/'metadata.npz');base=np.load(ep/'masks_full.npz');labels=base['labels'].copy();role=base['role'].copy();gd=json.loads((ep/'geometry.json').read_text());stats=gd['mask_stats'];prompts=cfg['tasks'][row['task']]
 original_hash=hashlib.sha256((ep/'masks_full.npz').read_bytes()).hexdigest();details=[]
 for prompt,fix in repairs.items():
  pi=next(i for i,p in enumerate(prompts) if p['text']==prompt);rid=prompts[pi]['role'];group=(pi+1)*1000
  nonzero=[(j,x) for j,x in enumerate(labels) if ((x>=group)&(x<group+1000)).any()]
  for fi in (list(dict.fromkeys([fix['frames'][0],fix['frames'][-1]])) if a.probe else fix['frames']):
   nearest=min(nonzero,key=lambda z:abs(z[0]-fi))[1];target=(nearest>=group)&(nearest<group+1000);ys,xs=np.where(target);vals,cnts=np.unique(nearest[target],return_counts=True);label=int(vals[cnts.argmax()])
   box=fix.get('box',[max(0,float(xs.min()/640-.08)),max(0,float(ys.min()/480-.08)),0,0])
   if 'box' not in fix:box[2]=min(1-box[0],float((xs.max()-xs.min()+1)/640+.16));box[3]=min(1-box[1],float((ys.max()-ys.min()+1)/480+.16))
   name=meta['names'][meta['keyframes'][fi]];rgb=read_rgb(row,name)
   if prompt=='red cube':
    red=(rgb[...,0].astype(float)>1.7*rgb[...,1])&(rgb[...,0].astype(float)>1.7*rgb[...,2])&(rgb[...,0]>45)
    # This stacking task has exactly one red cube; its largest red component seeds the current frame.
    count,cc,ccstats,_=cv2.connectedComponentsWithStats(red.astype('uint8'),8)
    if count>1:
     target_id=1+ccstats[1:,cv2.CC_STAT_AREA].argmax();red_component=cc==target_id;x,y,bw,bh,area=ccstats[target_id]
     if area>100:box=[max(0,(x-3)/640),max(0,(y-3)/480),min(1,(bw+6)/640),min(1,(bh+6)/480)]
   single=out/'single';single.mkdir(exist_ok=True);Image.fromarray(rgb).save(single/'0.jpg',quality=95)
   sid=pred.handle_request({'type':'start_session','resource_path':str(single)})['session_id'];v=pred.handle_request({'type':'add_prompt','session_id':sid,'frame_index':0,'text':fix['text'],'bounding_boxes':[box],'bounding_box_labels':[1]})['outputs'];pred.handle_request({'type':'close_session','session_id':sid})
   masks=np.asarray(v['out_binary_masks'],bool);m=masks.any(0) if len(masks) else np.zeros((480,640),bool)
   if prompt=='red cube':
    m=(m&cv2.dilate(red_component.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool))|red_component
   if rid==1:m&=(role[fi]!=2)
   old=(labels[fi]>=group)&(labels[fi]<group+1000);labels[fi][old]=0;role[fi][old]=0;labels[fi][m]=label;role[fi][m]=rid
   stats[prompt][fi].update(instances=int(m.any()),area=int(m.sum()),ids=[label-group] if m.any() else [],recovery='single_frame_text_box')
   item={'episode':key,'prompt':prompt,'keyframe':fi,'raw_frame':int(meta['keyframes'][fi])+1,'area':int(m.sum()),'box':box,'required':fix['required']};details.append(item);report.append(item)
   ov=rgb.copy();ov[m]=(rgb[m]*.55+np.array([70,230,90])*.45).astype('uint8');im=Image.fromarray(ov).resize((400,300));ImageDraw.Draw(im).text((5,5),f'{key} {prompt} f{item["raw_frame"]}\narea={item["area"]}',fill='white',stroke_width=2,stroke_fill='black');panels.append(im)
   atomic_json(status,{'phase':'running','pid':os.getpid(),'last':item,'completed_frames':len(report),'heartbeat':time.time()})
 if not a.probe:
  np.savez_compressed(ep/'masks_repaired.npz',labels=labels,role=role,indices=base['indices'])
  atomic_json(ep/'mask_repair_report.json',{'source_mask_sha256':original_hash,'plan':repairs,'stats':stats,'details':details,'perception_signature':perception_signature(cfg,row['task'],key),'source':'SAM3.1 current-frame text and positive box recovery'})
for start in range(0,len(panels),12):
 page=Image.new('RGB',(1200,300*min(4,(len(panels)-start+2)//3)),(255,255,255))
 for i,im in enumerate(panels[start:start+12]):page.paste(im,((i%3)*400,(i//3)*300))
 page.save(out/f'{"probe" if a.probe else "full"}_{start//12:02d}.jpg',quality=90)
atomic_json(out/('probe.json' if a.probe else 'report.json'),report)
missing=[x for x in report if x['required'] and x['area']<100]
atomic_json(status,{'phase':'needs_review' if missing else 'complete','probe':a.probe,'frames':len(report),'missing_required':missing,'heartbeat':time.time()});print('REPAIRS',len(report),'MISSING',missing,flush=True)
