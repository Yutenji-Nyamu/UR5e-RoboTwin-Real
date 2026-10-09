import os,sys,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from pipeline_common import *
from sam3.model_builder import build_sam3_multiplex_video_predictor
torch.set_num_threads(4);torch.set_num_interop_threads(2)
pred=build_sam3_multiplex_video_predictor(checkpoint_path=str(ROOT/'models/sam3.1/sam3.1_multiplex.pt'),compile=False,use_fa3=False,use_rope_real=False,warm_up=False,async_loading_frames=False)
init=pred.model.init_state
def compat(*a,**k):k.pop('offload_state_to_cpu',None);return init(*a,**k)
pred.model.init_state=compat
out=ROOT/'runs/prompt_probe_v2';out.mkdir(exist_ok=True);report={}
for key,prompts in [('20261006_165400',['robot','robotic gripper','robot gripper','mechanical gripper','wooden drawer','wooden cabinet']),('20261006_170154',['robot','robotic gripper','red cube','yellow cube','blue cube','plastic bin','plastic tray','box'])]:
 row=next(x for x in rows() if x['key']==key);m=np.load(CACHE/key/'metadata.npz');panels=[];report[key]={}
 for prompt in prompts:
  info=[]
  for fi in [0,len(m['names'])//2,len(m['names'])-1]:
   rgb=read_rgb(row,m['names'][fi]);frame=out/'single';frame.mkdir(exist_ok=True);Image.fromarray(rgb).save(frame/'0.jpg',quality=95)
   sid=pred.handle_request({'type':'start_session','resource_path':str(frame)})['session_id']
   resp=pred.handle_request({'type':'add_prompt','session_id':sid,'frame_index':0,'text':prompt});v=resp['outputs'];masks=np.asarray(v['out_binary_masks'],bool);pred.handle_request({'type':'close_session','session_id':sid})
   ov=rgb.copy()
   for k,mask in enumerate(masks):ov[mask]=(rgb[mask]*.55+np.array([(250,70,80),(70,230,90),(80,100,250)][k%3])*.45).astype('uint8')
   im=Image.fromarray(ov).resize((400,300));ImageDraw.Draw(im).text((5,5),f'{prompt} f{fi+1} N={len(masks)}',fill='white',stroke_width=2,stroke_fill='black');panels.append(im);info.append({'frame':fi,'instances':len(masks),'areas':[int(x.sum()) for x in masks]})
  report[key][prompt]=info
 sheet=Image.new('RGB',(1200,300*len(prompts)))
 for i,im in enumerate(panels):sheet.paste(im,((i%3)*400,(i//3)*300))
 sheet.save(out/f'{key}.jpg',quality=90)
atomic_json(out/'status.json',{'state':'complete','report':report});print('PROMPT_PROBE_COMPLETE',flush=True)
