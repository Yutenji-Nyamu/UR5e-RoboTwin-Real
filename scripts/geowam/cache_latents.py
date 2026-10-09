"""Frozen Wan VAE/UMT5 offline cache in the author's exact latent contract."""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from pipeline_common import *
p=argparse.ArgumentParser();p.add_argument('mode',choices=['text','latents']);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--keys',nargs='*');p.add_argument('--limit-windows',type=int);p.add_argument('--wait',action='store_true');a=p.parse_args()
sys.path.insert(0,str(ROOT/'repos/MetisWAM4D'))
import torch,cv2
from PIL import Image,ImageDraw
from metiswam4d.data.rt2.online_encoder import RT2OnlineEncoder,RT2EncoderConfig
from metiswam4d.data.rt2.text_cache import format_prompt
from metiswam4d.data.contract import SampleSpec,validate_sample

torch.set_num_threads(4);torch.set_num_interop_threads(2);device=torch.device('cuda');assert torch.cuda.device_count()==1
state={'pid':os.getpid(),'starttime':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'phase':'loading','finished_episodes':[],'errors':[],'started':time.time()};status=ROOT/'runs'/(f'cache-{a.mode}-keys-'+hashlib.sha256(','.join(sorted(a.keys)).encode()).hexdigest()[:10]+'.json' if a.keys else f'cache-{a.mode}-{a.worker}.json');atomic_json(status,state)
if a.mode=='text':
 from metiswam4d.data.human.text import UMT5Online
 enc=UMT5Online(str(ROOT/'models/wan22-ti2v-5b'),device,max_len=512);texts={};info={}
 for task,instruction in TASKS.items():
  prompt=format_prompt(instruction);z,mask=enc([prompt]);n=int(mask[0].sum());text=z[0,:n].cpu().contiguous();assert text.shape[-1]==4096 and torch.isfinite(text).all();texts[task]=text;info[task]={'instruction':instruction,'prompt':prompt,'length':n}
 torch.save(texts,CACHE/'text_context.pt');atomic_json(CACHE/'text_context.json',info);state.update(phase='complete',finished=time.time());atomic_json(status,state);print('TEXT_COMPLETE',info,flush=True);sys.exit(0)
encoder=RT2OnlineEncoder(RT2EncoderConfig(vae_path=str(ROOT/'models/wan22-ti2v-5b'),depth_stats=str(CACHE/'depth_stats.json')),device)
text=torch.load(CACHE/'text_context.pt',map_location='cpu',weights_only=True);stats=json.loads((CACHE/'normalization.json').read_text());spec=SampleSpec()
selected=[x for x in rows() if not a.keys or x['key'] in a.keys][a.worker::a.workers]

def sample_raw(row,meta,g,s):
 i=s//4;frames=[]
 for j in range(9):
  name=meta['names'][s+j*4]
  frames.append(np.concatenate([cv2.resize(read_rgb(row,name,c),(320,192),interpolation=cv2.INTER_AREA) for c in ['head','wrist']],axis=0))
 name=meta['names'][s];rgb=cv2.resize(read_rgb(row,name),(320,240),interpolation=cv2.INTER_AREA);dep=cv2.resize(read_depth(row,name,float(meta['depth_scale'])),(320,240),interpolation=cv2.INTER_NEAREST)
 roles=g['role'][i:i+8];fg=g['valid'][i:i+8];anchor=g['foreground'][i];anchor_rgb=np.zeros((240,320,3),np.uint8);anchor_rgb[anchor]=128
 raw={'video_frames':np.stack(frames),'track_rgb':np.concatenate([anchor_rgb[None],g['track_rgb'][i:i+8]]),'track_foreground':np.concatenate([anchor[None],fg]),'track_role_px':np.concatenate([roles[:1],roles]),'track_delta':g['track_delta'][i:i+8].astype('float32'),'head_rgb':rgb,'head_depth_mm':dep*1000,'head_mask':anchor,**targets(meta,s,stats),'progress_video':np.array([s/(len(meta['names'])-1)],np.float32),'progress_body':np.array([s/(len(meta['names'])-1)],np.float32)}
 assert raw['track_rgb'].shape==(9,240,320,3)
 out={k:torch.from_numpy(v)[None] for k,v in raw.items()};out['text_context']=text[row['task']][None];out['embodiment']=torch.tensor([9]);out['key']=[row['key']+f'/s{s}'];return out

def process(row):
 ep=CACHE/row['key'];meta=np.load(ep/'metadata.npz');g=np.load(ep/'geometry.npz');report=json.loads((ep/'geometry.json').read_text())
 if report['missing_initial']:raise ValueError('perception review required: '+str(report['missing_initial']))
 assert report['state']=='complete' and np.array_equal(g['indices'],meta['keyframes'])
 starts=meta['starts'].tolist()
 if a.limit_windows:starts=starts[:a.limit_windows]
 signature=hashlib.sha256((ep/'geometry.npz').read_bytes()+ (CACHE/'normalization.json').read_bytes()+b'latent_contract_v1').hexdigest()
 out=ep/('latents_'+signature[:12]);out.mkdir(exist_ok=True);entries=[];quality=[]
 previous=json.loads((ep/'latents.json').read_text()) if (ep/'latents.json').exists() else {}
 if previous.get('input_signature')==signature:quality=previous.get('quality',[])
 for si,s in enumerate(starts):
  state.update(phase='encoding',episode=row['key'],window=si,windows=len(starts),heartbeat=time.time());atomic_json(status,state)
  dest=out/f's{s:05d}.pt'
  if not dest.exists():
   raw=sample_raw(row,meta,g,int(s))
   with torch.inference_mode():encoded=encoder(raw)
   sample={k:v[0].detach().cpu().contiguous() for k,v in encoded.items() if torch.is_tensor(v)};sample.pop('embodiment',None)
   validate_sample(sample,spec)
   for k,v in sample.items():
    if v.is_floating_point():assert torch.isfinite(v).all(),k
   tmp=dest.with_suffix('.tmp');torch.save(sample,tmp);tmp.replace(dest)
   if si in [0,len(starts)//2,len(starts)-1]:
    recon=encoder.decode_latents(encoded['video_clean'])[0].cpu().numpy();true=raw['video_frames'][0].numpy();err=float(np.mean((recon.astype('float32')-true.astype('float32'))**2));quality.append({'s':s,'video_reconstruction_mse':err})
    rec_t=encoder.decode_latents(encoded['track_clean'])[0].cpu().numpy();sheet=Image.new('RGB',(960,384*3))
    for j,k in enumerate([0,4,8]):
     sheet.paste(Image.fromarray(true[k]),(j*320,0));sheet.paste(Image.fromarray(recon[k]),(j*320,384));sheet.paste(Image.fromarray(rec_t[k]),(j*320,768))
    ImageDraw.Draw(sheet).text((4,4),f'{row["key"]} s{s} RGB / VAE / UVD VAE',fill='white',stroke_width=2,stroke_fill='black');sheet.save(ep/f'vae_review_s{s:05d}.jpg',quality=90)
  entries.append({'path':str(dest),'key':row['key']+f'/s{s}','embodiment':'ur5e_single_joint','task':row['task'],'episode':row['key']})
 tmp=ep/'manifest.jsonl.tmp';tmp.write_text(''.join(json.dumps(x)+'\n' for x in entries));tmp.replace(ep/'manifest.jsonl')
 atomic_json(ep/'latents.json',{'state':'complete','input_signature':signature,'windows':len(entries),'all_windows':len(meta['starts']),'quality':quality,'finished':time.time(),'spec':vars(spec),'video_layout':'head 320x192 above wrist 320x192','track_view':'head','action':'future joint delta; absolute gripper; 80 slots 10:17','embodiment':9})
 print('LATENTS_COMPLETE',row['key'],len(entries),flush=True)
pending=list(selected)
while pending:
 progress=False
 for row in list(pending):
  ep=CACHE/row['key']
  if not (ep/'geometry.json').exists():continue
  report=json.loads((ep/'geometry.json').read_text());cfg=json.loads((CONFIG/'perception.json').read_text())
  if report.get('perception_signature')!=perception_signature(cfg,row['task'],row['key']):continue
  try:process(row);state['finished_episodes'].append(row['key'])
  except Exception as e:state['errors'].append({'episode':row['key'],'error':repr(e)});traceback.print_exc()
  pending.remove(row);progress=True;atomic_json(status,state)
 if pending:
  state.update(phase='waiting_geometry',pending=[x['key'] for x in pending],heartbeat=time.time());atomic_json(status,state)
  if not a.wait:break
  if not progress:time.sleep(30)
state.update(phase=('complete_with_errors' if state['errors'] else 'complete') if not pending else 'incomplete',finished=time.time());atomic_json(status,state)
