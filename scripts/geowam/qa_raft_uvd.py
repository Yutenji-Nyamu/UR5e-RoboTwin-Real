import argparse,json,time,sys,os,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();r=a.workspace
out=r/'runs/qa_raft_uvd_drawer';out.mkdir(parents=True,exist_ok=True);status={'state':'starting','started':time.time(),'pid':os.getpid()}
try:
 import numpy as np,torch,cv2
 from PIL import Image,ImageDraw
 from torchvision.models.optical_flow import raft_large
 from torchvision.utils import flow_to_image
 sys.path.insert(0,str(r/'repos/MetisWAM4D'))
 from metiswam4d.data.rt2.codec import encode_uvd,CODEC_ID
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 frames=sorted((r/'cache/qa_input/drawer_165400/head').glob('*.png'));rgb=[np.array(Image.open(x).convert('RGB')) for x in frames[:2]];h,w=rgb[0].shape[:2]
 cal=json.loads((r/'repos/data_ur5e_gwam/2026-10-06/block_drawer/20261006_165400/images/camera_calibration.json').read_text());scale=cal['cameras']['head']['depth_scale_m_per_unit']
 depth=[np.array(Image.open(r/'cache/qa_input/drawer_165400/head_depth'/x.name)).astype('float32')*scale for x in frames[:2]]
 model=raft_large(weights=None,progress=False).cuda().eval();model.load_state_dict(torch.load(r/'models/raft/raft_large_C_T_SKHT_V2-ff5fadd5.pth',map_location='cpu',weights_only=True),strict=True)
 batch=torch.from_numpy(np.stack(rgb)).permute(0,3,1,2).float().cuda()/127.5-1
 with torch.inference_mode():flows=model(batch,batch.flip(0),num_flow_updates=24)[-1]
 fw,bw=flows.permute(0,2,3,1).cpu().numpy();yy,xx=np.mgrid[:h,:w].astype('float32');mx=xx+fw[...,0];my=yy+fw[...,1]
 warp=lambda img,inter=cv2.INTER_LINEAR:cv2.remap(img,mx,my,inter,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
 inside=(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1);fb=np.linalg.norm(fw+warp(bw),axis=-1)
 d1=warp(depth[1]);valid_depth=(depth[0]>0)&(warp((depth[1]>0).astype('float32'))>.999)
 labels=[]
 for i in range(2):
  lab=np.zeros((h,w),np.float32)
  for value,tag,key in [(1,'drawer','drawer'),(2,'gripper','robot'),(3,'drawer','red_block')]:
   masks=np.load(r/f'runs/qa_sam31_{tag}'/f'{key}_{i:03}.npz')['masks'];mask=masks.any(axis=0) if len(masks) else np.zeros((h,w),bool);lab[mask]=value
  labels.append(lab)
 same=(labels[0]>0)&(labels[0]==warp(labels[1],cv2.INTER_NEAREST));valid=inside&valid_depth&(fb<1.)&same
 uvd=np.dstack((fw,d1-depth[0]));enc,z=encode_uvd(uvd,valid,w,.002)
 np.savez_compressed(out/'pair_00050_00054.npz',flow_forward=fw,flow_backward=bw,uvd_pixels_m=uvd,valid=valid,track_delta=z,instance=labels[0],fb_error=fb)
 Image.fromarray(enc).save(out/'uvd_encoded.png');flow=flow_to_image(flows[:1]).cpu()[0].permute(1,2,0).numpy()
 depthvis=cv2.applyColorMap(np.clip(depth[0]/1.5*255,0,255).astype('uint8'),cv2.COLORMAP_TURBO)[...,::-1];depthvis[depth[0]<=0]=0
 dd=np.clip((uvd[...,2]/.1+1)*127.5,0,255).astype('uint8');dd=cv2.applyColorMap(dd,cv2.COLORMAP_TURBO)[...,::-1];dd[~valid]=0
 panels=[(rgb[0],'RGB t'),(rgb[1],'RGB t+4'),(flow,'RAFT forward'),(depthvis,'Depth 0-1.5m'),(dd,'Depth change -0.1 to +0.1m'),(enc,'Track UVD valid foreground')];sheet=Image.new('RGB',(w*3,h*2))
 for i,(arr,title) in enumerate(panels):
  im=Image.fromarray(arr);ImageDraw.Draw(im).text((8,8),title,fill='white',stroke_width=2,stroke_fill='black');sheet.paste(im,((i%3)*w,(i//3)*h))
 sheet.save(out/'flow_depth_uvd.jpg',quality=90);fg=labels[0]>0
 status.update(state='complete',seconds=time.time()-status['started'],pair=[x.name for x in frames[:2]],depth_scale=scale,codec=CODEC_ID,tau_d_display=.002,foreground_pixels=int(fg.sum()),valid_pixels=int(valid.sum()),foreground_valid_fraction=float(valid.sum()/max(1,fg.sum())),foreground_fb_median_px=float(np.median(fb[fg])),valid_depth_change_median_m=float(np.median(uvd[...,2][valid])),peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
except Exception as e:status.update(state='error',error=type(e).__name__+': '+str(e));traceback.print_exc()
finally:(out/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
