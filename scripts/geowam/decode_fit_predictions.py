"""Decode saved fit-probe latents on CPU while both GPUs continue training."""
import argparse,fcntl,json,os,sys,time
from pathlib import Path
from pipeline_common import *
import torch
from PIL import Image,ImageDraw
sys.path.insert(0,str(ROOT/'repos/MetisWAM4D'))
from metiswam4d.data.rt2.online_encoder import RT2OnlineEncoder,RT2EncoderConfig
p=argparse.ArgumentParser();p.add_argument('--run',default='train_rgbd72_v1');p.add_argument('--step',type=int);p.add_argument('--tasks',nargs='*');a=p.parse_args()
run=ROOT/'runs'/a.run;assert run.resolve().parent==(ROOT/'runs').resolve()
folders=sorted(p for p in (run/'vis').glob('step_*') if (p/'scalars.json').exists())
if a.step is None and not folders:print('WAITING_FOR_FIRST_COMPLETE_FIT_PROBE');sys.exit(0)
folder=run/'vis'/f'step_{a.step:07d}' if a.step is not None else folders[-1]
lock=open(folder/'decode.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
files=[f for f in sorted(folder.glob('*_predictions.pt')) if not a.tasks or f.name.removesuffix('_predictions.pt') in a.tasks]
pending=[f for f in files if not (folder/(f.name.removesuffix('_predictions.pt')+'_decoded.jpg')).exists()]
if not pending:print('DECODE_ALREADY_COMPLETE');sys.exit(0)
torch.set_num_threads(8);torch.set_num_interop_threads(2)
enc=RT2OnlineEncoder(RT2EncoderConfig(vae_path=str(ROOT/'models/wan22-ti2v-5b'),depth_stats=str(CACHE/'depth_stats.json')),torch.device('cpu'))
info=[{'task':f.name.removesuffix('_predictions.pt'),'image':str(folder/(f.name.removesuffix('_predictions.pt')+'_decoded.jpg')),'reused':True} for f in files if f not in pending]
atomic_json(folder/'decode_status.json',{'phase':'running','completed':info,'device':'cpu'})
for p in pending:
 task=p.name.removesuffix('_predictions.pt');z=torch.load(p,map_location='cpu',weights_only=True);decoded={}
 for name in ['video_gt','video_pred','track_gt','track_pred']:
  with torch.inference_mode():decoded[name]=enc.decode_latents(z[name])[0].numpy()
 page=Image.new('RGB',(960,384*2+256*2+5*24),'white');y=0
 for name in ['video_gt','video_pred','track_gt','track_pred']:
  ImageDraw.Draw(page).text((5,y),task+' | '+name+' (frames 0 / 4 / 8)',fill='black');y+=24
  for col,frame in enumerate([0,4,8]):page.paste(Image.fromarray(decoded[name][frame]),(col*320,y))
  y+=decoded[name].shape[1]
 page.crop((0,0,960,y)).save(folder/f'{task}_decoded.jpg',quality=92)
 # 9-frame predicted video and Track are retained as PNG strips; source latent tensors remain available.
 for name in ['video_pred','track_pred']:
  arr=decoded[name];strip=Image.new('RGB',(320*len(arr),arr.shape[1]))
  for i,frame in enumerate(arr):strip.paste(Image.fromarray(frame),(320*i,0))
  strip.save(folder/f'{task}_{name}_strip.png')
 info.append({'task':task,'finished':time.time(),'image':str(folder/f'{task}_decoded.jpg')});atomic_json(folder/'decode_status.json',{'phase':'running','completed':info,'device':'cpu'});print('DECODED',task,flush=True)
atomic_json(folder/'decode_status.json',{'phase':'complete','completed':info,'device':'cpu'})
