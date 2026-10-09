"""Small real-recording mask QA; outputs stay outside Git."""
import argparse,json,time,os,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--tag',default='drawer');p.add_argument('--prompts',nargs='+',default=['red block','robot arm','drawer']);a=p.parse_args();r=a.workspace
out=r/('runs/qa_sam31_'+a.tag);out.mkdir(parents=True,exist_ok=True)
status={'state':'starting','pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'started':time.time()}
(out/'status.json').write_text(json.dumps(status,indent=2))
try:
 import numpy as np,torch
 from PIL import Image,ImageDraw
 from sam3.model_builder import build_sam3_multiplex_video_predictor
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 inputs=sorted((r/'cache/qa_input/drawer_165400/head').glob('*.png'));assert len(inputs)==9
 frames=out/'frames';frames.mkdir(exist_ok=True)
 for i,q in enumerate(inputs):Image.open(q).convert('RGB').save(frames/f'{i}.jpg',quality=95)
 predictor=build_sam3_multiplex_video_predictor(checkpoint_path=str(r/'models/sam3.1/sam3.1_multiplex.pt'),compile=False,use_fa3=False,use_rope_real=False,warm_up=False,async_loading_frames=False)
 # SAM3 0570b3a passes an unused CPU-state flag to the multiplex initializer.
 original_init=predictor.model.init_state
 def compatible_init(*args,**kwargs):
  assert kwargs.pop('offload_state_to_cpu',False) is False
  return original_init(*args,**kwargs)
 predictor.model.init_state=compatible_init
 allstats={};panels=[]
 for prompt in a.prompts:
  key=prompt.replace(' ','_');session=predictor.handle_request({'type':'start_session','resource_path':str(frames)})['session_id']
  predictor.handle_request({'type':'add_prompt','session_id':session,'frame_index':0,'text':prompt})
  stats=[];outputs={}
  for resp in predictor.handle_stream_request({'type':'propagate_in_video','session_id':session}):
   i=resp['frame_index'];v=resp['outputs'];m=np.asarray(v['out_binary_masks']).astype(bool);ids=np.asarray(v.get('out_obj_ids',np.arange(len(m))))
   outputs[i]=m;np.savez_compressed(out/f'{key}_{i:03}.npz',masks=m,ids=ids)
   stats.append({'frame':i,'instances':len(m),'pixels':[int(z.sum()) for z in m],'ids':ids.tolist()})
  predictor.handle_request({'type':'close_session','session_id':session});allstats[key]=stats
  for i in [0,4,8]:
   rgb=np.array(Image.open(inputs[i]).convert('RGB'));m=outputs[i];overlay=rgb.copy();colors=[(250,80,50),(40,210,90),(40,100,250),(220,80,220)]
   for k,mask in enumerate(m):overlay[mask]=(rgb[mask]*.55+np.array(colors[k%len(colors)])*.45).astype('uint8')
   im=Image.fromarray(overlay);ImageDraw.Draw(im).text((8,8),f'{prompt} | source {inputs[i].stem} | {len(m)} objects',fill='white',stroke_width=2,stroke_fill='black');panels.append(im)
 sheet=Image.new('RGB',(640*3,480*len(a.prompts)))
 for i,im in enumerate(panels):sheet.paste(im,((i%3)*640,(i//3)*480))
 sheet.save(out/'masks_contact_sheet.jpg',quality=90)
 status.update(state='complete',finished=time.time(),seconds=time.time()-status['started'],gpu_name=torch.cuda.get_device_name(),peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,prompts=allstats)
except Exception as e:
 status.update(state='error',error=type(e).__name__+': '+str(e));traceback.print_exc()
finally:(out/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps({k:v for k,v in status.items() if k!='prompts'}),flush=True)
