"""Known RGB translation and depth increment check for RAFT direction and UVD units."""
import sys
import numpy as np,torch,cv2
from torchvision.models.optical_flow import raft_large
from pipeline_common import *
sys.path.insert(0,str(ROOT/'repos/MetisWAM4D'))
from metiswam4d.data.rt2.codec import encode_uvd,inverse_mu_law
torch.set_num_threads(4);assert torch.cuda.device_count()==1
row=next(x for x in rows() if x['task']=='sort_block');meta=np.load(CACHE/row['key']/'metadata.npz');rgb=read_rgb(row,meta['names'][0]);h,w=rgb.shape[:2]
shift=np.array([[1,0,8],[0,1,-4]],np.float32);future=cv2.warpAffine(rgb,shift,(w,h));x=torch.from_numpy(np.stack([rgb,future])).permute(0,3,1,2).float().cuda()/127.5-1
model=raft_large(weights=None).cuda().eval();model.load_state_dict(torch.load(ROOT/'models/raft/raft_large_C_T_SKHT_V2-ff5fadd5.pth',map_location='cpu',weights_only=True))
with torch.inference_mode():flow=model(x[:1],x[1:],num_flow_updates=24)[-1][0].permute(1,2,0).cpu().numpy()
region=np.zeros((h,w),bool);region[40:-40,40:-40]=True;e=np.linalg.norm(flow-np.array([8,-4]),axis=-1);median=float(np.median(e[region]));p90=float(np.quantile(e[region],.9));assert median<1.,(median,p90)
uvd=np.zeros((h,w,3),np.float32);uvd[:]=[8,-4,.020];encoded,delta=encode_uvd(uvd,region,w,.002);expected=np.array([(8-.5)/w*6,(-4+.5)/w*6,.18]);assert np.max(abs(delta[region]-expected))<1e-6;assert (encoded[~region]==0).all() and (delta[~region]==0).all();quantization=float(np.max(abs(inverse_mu_law(encoded[region])-delta[region])));assert quantization<.01
atomic_json(ROOT/'runs/uvd_semantics_check.json',{'state':'passed','raft_translation':[8,-4],'median_flow_error_px':median,'p90_flow_error_px':p90,'depth_increment_m':.02,'normalised_expected':expected.tolist(),'max_codec_quantization_error':quantization,'invalid_pixels':'zeroed'})
print('UVD_SEMANTICS_PASSED',median,p90,quantization,flush=True)
