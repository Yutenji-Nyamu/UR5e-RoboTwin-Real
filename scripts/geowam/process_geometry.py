"""SAM3.1 -> RAFT forward/backward -> measured-depth UVD, one isolated GPU per worker."""
import argparse,gc,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from pipeline_common import *
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--tasks',nargs='*');p.add_argument('--keys',nargs='*');p.add_argument('--limit',type=int);p.add_argument('--wait',action='store_true');p.add_argument('--force',action='store_true');p.add_argument('--reuse-masks',action='store_true');p.add_argument('--status-name');a=p.parse_args()
sys.path.insert(0,str(ROOT/'repos/MetisWAM4D'))
import torch,cv2
from PIL import Image,ImageDraw
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from sam3.model_builder import build_sam3_multiplex_video_predictor
from metiswam4d.data.rt2.codec import encode_uvd,CODEC_ID

torch.set_num_threads(4);torch.set_num_interop_threads(2)
assert torch.cuda.device_count()==1
cfg=json.loads((CONFIG/'perception.json').read_text());status_path=ROOT/'runs'/(a.status_name or f'geometry-worker{a.worker}.json')
selected=[x for x in rows() if (not a.tasks or x['task'] in a.tasks) and (not a.keys or x['key'] in a.keys)]
if a.limit:selected=selected[:a.limit]
selected=selected[a.worker::a.workers]
status={'pid':os.getpid(),'starttime':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'phase':'loading','completed':[],'errors':[],'selected':[x['key'] for x in selected]};atomic_json(status_path,status)
if not a.reuse_masks:
 predictor=build_sam3_multiplex_video_predictor(checkpoint_path=str(ROOT/'models/sam3.1/sam3.1_multiplex.pt'),compile=False,use_fa3=False,use_rope_real=False,warm_up=False,async_loading_frames=False)
 original_init=predictor.model.init_state
 def compatible_init(*args,**kwargs):
  assert kwargs.pop('offload_state_to_cpu',False) is False
  return original_init(*args,**kwargs)
 predictor.model.init_state=compatible_init
 # Per-frame grounding preserves the seed geometric prompt; SAM multiplex batched grounding bypasses it.
 predictor.model.use_batched_grounding=False
raft=raft_large(weights=None).cuda().eval();raft.load_state_dict(torch.load(ROOT/'models/raft/raft_large_C_T_SKHT_V2-ff5fadd5.pth',map_location='cpu',weights_only=True),strict=True)

def process(row):
 ep=CACHE/row['key'];meta=np.load(ep/'metadata.npz');idx=meta['keyframes'];names=meta['names'][idx];n=len(names)
 if not a.force and (ep/'geometry.json').exists():
  old=json.loads((ep/'geometry.json').read_text())
  if old.get('state')=='complete' and old.get('perception_signature')==perception_signature(cfg,row['task'],row['key']) and not old.get('missing_initial'):return
 if not paths_ready(row,meta):raise FileNotFoundError('raw images not yet restored')
 frames=ep/'sam_frames';frames.mkdir(exist_ok=True)
 for i,name in enumerate(names):Image.fromarray(read_rgb(row,name)).save(frames/f'{i}.jpg',quality=95)
 prompts=cfg['tasks'][row['task']];h,w=480,640
 labels=np.zeros((n,h,w),np.uint16);role=np.zeros((n,h,w),np.uint8);stats={};panels=[]
 if row['key'] in cfg.get('episode_repairs',{}):
  rep=json.loads((ep/'mask_repair_report.json').read_text())
  assert rep['perception_signature']==perception_signature(cfg,row['task'],row['key'])
  repaired=np.load(ep/'masks_repaired.npz');labels=repaired['labels'];role=repaired['role'];stats=rep['stats']
  assert np.array_equal(repaired['indices'],idx)
  for pi,pc in enumerate(prompts):
   for i in [0,n//2,n-1]:
    rgb=read_rgb(row,names[i]);mask=(labels[i]>=(pi+1)*1000)&(labels[i]<(pi+2)*1000);ov=rgb.copy();ov[mask]=(rgb[mask]*.55+np.array([70,220,90])*.45).astype('uint8')
    im=Image.fromarray(ov);ImageDraw.Draw(im).text((8,8),f'{row["key"]} {pc["text"]} f{int(idx[i])+1}',fill='white',stroke_width=2,stroke_fill='black');panels.append(im.resize((480,360)))
 elif a.reuse_masks:
  saved=np.load(ep/'masks_full.npz');labels=saved['labels'];role=saved['role'];stats=json.loads((ep/'geometry.json').read_text())['mask_stats']
  assert np.array_equal(saved['indices'],idx)
  for pi,pc in enumerate(prompts):
   for i in [0,n//2,n-1]:
    rgb=read_rgb(row,names[i]);mask=(labels[i]>=(pi+1)*1000)&(labels[i]<(pi+2)*1000);ov=rgb.copy();ov[mask]=(rgb[mask]*.55+np.array([70,220,90])*.45).astype('uint8')
    im=Image.fromarray(ov);ImageDraw.Draw(im).text((8,8),f'{row["key"]} {pc["text"]} f{int(idx[i])+1}',fill='white',stroke_width=2,stroke_fill='black');panels.append(im.resize((480,360)))
 else:
  for pi,pc in enumerate(prompts):
   pc=dict(pc);prompt=pc['text'];query=pc.get('episode_text',{}).get(row['key'],prompt)
   if prompt in cfg.get('episode_boxes',{}).get(row['key'],{}):pc['initial_box_if_empty']=cfg['episode_boxes'][row['key']][prompt]
   status.update(phase='sam',episode=row['key'],prompt=prompt,heartbeat=time.time());atomic_json(status_path,status)
   sid=predictor.handle_request({'type':'start_session','resource_path':str(frames),'offload_video_to_cpu':True})['session_id']
   first=predictor.handle_request({'type':'add_prompt','session_id':sid,'frame_index':0,'text':query})
   if not len(first['outputs']['out_binary_masks']) and pc.get('initial_box_if_empty'):
    predictor.handle_request({'type':'add_prompt','session_id':sid,'frame_index':0,'text':query,'bounding_boxes':[pc['initial_box_if_empty']],'bounding_box_labels':[1]})
   counts=[];samples={};seen=set()
   for resp in predictor.handle_stream_request({'type':'propagate_in_video','session_id':sid,'propagation_direction':'forward'}):
    i=resp['frame_index'];v=resp['outputs'];m=np.asarray(v['out_binary_masks'],dtype=bool);ids=np.asarray(v['out_obj_ids']);seen.add(i)
    for k,mask in enumerate(m):
     assert 0<=int(ids[k])<999
     labels[i][mask]=(pi+1)*1000+(0 if pc['role']==1 or pc.get('single_instance') else int(ids[k]));role[i][mask]=pc['role']
    counts.append({'i':int(i),'instances':len(m),'area':int(m.sum()),'ids':ids.tolist()})
    if i in [0,n//2,n-1]:samples[i]=m.copy()
   predictor.handle_request({'type':'close_session','session_id':sid});assert len(seen)==n
   stats[prompt]=counts
   for i in [0,n//2,n-1]:
    rgb=read_rgb(row,names[i]);mask=samples.get(i,np.zeros((0,h,w),bool));ov=rgb.copy()
    for k,m in enumerate(mask):ov[m]=(rgb[m]*.55+np.array([(230,70,70),(70,220,90),(70,100,230)][k%3])*.45).astype('uint8')
    im=Image.fromarray(ov);ImageDraw.Draw(im).text((8,8),f'{row["key"]} {prompt} f{int(idx[i])+1}',fill='white',stroke_width=2,stroke_fill='black');panels.append(im.resize((480,360)))
 sheet=Image.new('RGB',(1440,360*len(prompts)))
 for i,im in enumerate(panels):sheet.paste(im,((i%3)*480,(i//3)*360))
 sheet.save(ep/'mask_review.jpg',quality=88)
 np.savez_compressed(ep/'masks_full.npz',labels=labels,role=role,indices=idx)
 status.update(phase='raft',episode=row['key'],prompt=None,heartbeat=time.time());atomic_json(status_path,status)
 tracks=[];deltas=[];valids=[];diagnostics={};qualities=[];tau=cfg['tau_depth_m'];flow_iters=cfg['raft_updates']
 yy,xx=np.mgrid[:h,:w].astype('float32')
 for k in range(n-1):
  rgbs=np.stack([read_rgb(row,names[k]),read_rgb(row,names[k+1])]);x=torch.from_numpy(rgbs).permute(0,3,1,2).float().cuda()/127.5-1
  with torch.inference_mode():flow=raft(x,x.flip(0),num_flow_updates=flow_iters)[-1]
  fw,bw=flow.permute(0,2,3,1).float().cpu().numpy();mx=xx+fw[...,0];my=yy+fw[...,1]
  warp=lambda z,inter=cv2.INTER_LINEAR:cv2.remap(z,mx,my,inter,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
  d0=read_depth(row,names[k],float(meta['depth_scale']));d1=read_depth(row,names[k+1],float(meta['depth_scale']))
  fb=np.linalg.norm(fw+warp(bw),axis=-1);fg=labels[k]>0
  valid=(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)&(d0>0)&(warp((d1>0).astype('float32'))>.999)&(fb<cfg['fb_max_px'])&fg&(role[k]==warp(role[k+1].astype('float32'),cv2.INTER_NEAREST))
  id_disagreement=valid&(labels[k]!=warp(labels[k+1].astype('float32'),cv2.INTER_NEAREST))
  uvd=np.dstack((fw,warp(d1)-d0));encoded,delta=encode_uvd(uvd,valid,w,tau)
  tracks.append(cv2.resize(encoded,(320,240),interpolation=cv2.INTER_NEAREST));deltas.append(cv2.resize(delta,(320,240),interpolation=cv2.INTER_NEAREST).astype('float16'));valids.append(cv2.resize(valid.astype('uint8'),(320,240),interpolation=cv2.INTER_NEAREST).astype(bool))
  obj=role[k]==2;body=role[k]==1;static=(np.linalg.norm(fw,axis=-1)<.5)&valid
  quality={'k':k,'sam_id_disagreement_pixels':int(id_disagreement.sum()),'valid_fg':float(valid.sum()/max(fg.sum(),1)),'valid_object':float((valid&obj).sum()/max(obj.sum(),1)),'valid_body':float((valid&body).sum()/max(body.sum(),1)),'fg_pixels':int(fg.sum()),'object_pixels':int(obj.sum()),'fb_median':float(np.median(fb[fg])) if fg.any() else 0,'static_depth_abs_p90':float(np.quantile(abs(uvd[...,2][static]),.9)) if static.any() else 0,'clip_fraction':float((abs(delta[valid])>=1).any(axis=-1).mean()) if valid.any() else 0};qualities.append(quality)
  if k in [0,(n-1)//2,n-2]:
   diagnostics[str(k)]={'quality':quality};np.savez_compressed(ep/f'raw_pair_{k:04d}.npz',fw=fw.astype('float16'),bw=bw.astype('float16'),dd=uvd[...,2].astype('float16'),valid=valid,fb=fb.astype('float16'))
   flowrgb=flow_to_image(flow[:1]).cpu()[0].permute(1,2,0).numpy();dp=cv2.applyColorMap(np.clip(d0/1.6*255,0,255).astype('uint8'),cv2.COLORMAP_TURBO)[...,::-1];dp[d0<=0]=0
   panel=Image.new('RGB',(960,480))
   for j,(arr,title) in enumerate([(rgbs[0],'RGB t'),(flowrgb,'RAFT t -> t+4'),(dp,'measured depth'),(encoded,'UVD valid foreground')]):
    im=Image.fromarray(arr).resize((480,240));ImageDraw.Draw(im).text((5,5),title,fill='white',stroke_width=1,stroke_fill='black');panel.paste(im,((j%2)*480,(j//2)*240))
   panel.save(ep/f'flow_review_{k:04d}.jpg',quality=90)
  if k%10==0:status.update(pair=k,pairs=n-1,heartbeat=time.time());atomic_json(status_path,status)
 roles=np.stack([cv2.resize(x,(320,240),interpolation=cv2.INTER_NEAREST) for x in role]);fg=roles>0
 np.savez_compressed(ep/'geometry.npz',track_rgb=np.stack(tracks),track_delta=np.stack(deltas),valid=np.stack(valids),role=roles,foreground=fg,indices=idx)
 missing=[pc['text'] for pc in prompts if pc.get('required_initial') and stats[pc['text']][0]['area']<cfg['min_initial_pixels']]
 report={'state':'complete','episode':row['key'],'task':row['task'],'codec':CODEC_ID,'perception_signature':perception_signature(cfg,row['task'],row['key']),'config':cfg,'frames':len(meta['names']),'keyframes':n,'mask_stats':stats,'quality':qualities,'missing_initial':missing,'mean_valid_fg':float(np.mean([q['valid_fg'] for q in qualities])),'mean_valid_object':float(np.mean([q['valid_object'] for q in qualities])),'finished':time.time()}
 atomic_json(ep/'geometry.json',report);print('COMPLETE',row['key'],report['mean_valid_fg'],'missing_initial',missing,flush=True)

pending=list(selected)
while pending:
 progress=False
 for row in list(pending):
  meta=np.load(CACHE/row['key']/'metadata.npz')
  if not paths_ready(row,meta):continue
  try:process(row);status['completed'].append(row['key'])
  except Exception as ex:status['errors'].append({'episode':row['key'],'error':repr(ex)});traceback.print_exc()
  pending.remove(row);progress=True;atomic_json(status_path,status);gc.collect();torch.cuda.empty_cache()
 if pending:
  status.update(phase='waiting_images',pending=[x['key'] for x in pending],heartbeat=time.time());atomic_json(status_path,status)
  if not a.wait:break
  if not progress:time.sleep(30)
status.update(phase=('complete_with_errors' if status['errors'] else 'complete') if not pending else 'incomplete',finished=time.time());atomic_json(status_path,status)
