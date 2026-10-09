"""Thin single-arm registration and diagnostics around the unmodified author trainer."""
import os,sys,json,time,hashlib,subprocess,zipfile
from pathlib import Path
import numpy as np
from pipeline_common import ROOT,REPO,CACHE,atomic_json
sys.path.insert(0,str(ROOT/'repos/MetisWAM4D'))
import torch
import metiswam4d.train.train as trainer
import metiswam4d.train.data as train_data
from metiswam4d.data.embodiments import Embodiment,EmbodimentRegistry,default_registry
from metiswam4d.data.latent_dataset import LatentShardDataset
from metiswam4d.data.contract import collate,move_batch
from metiswam4d.train.visualize import Visualizer

torch.set_num_threads(8);torch.set_num_interop_threads(2)
if os.environ.get('GEOWAM_ANOMALY')=='1':torch.autograd.set_detect_anomaly(True)
def real_registry():
 original=default_registry();assert len(original.names())==9
 return EmbodimentRegistry([original[n] for n in original.names()]+[Embodiment('ur5e_single_joint',7,('10-16',),mode='joint',description='one UR5e arm; six joint deltas plus absolute gripper')])
train_data.default_registry=real_registry
_original_init=trainer.initialize_model
def initialize(model,stage,**kwargs):
 report=_original_init(model,stage,**kwargs)
 if int(os.environ.get('RANK','0'))==0:
  out=Path(stage.output_dir);atomic_json(out/'initialization.json',report)
  provenance={'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'physical_gpus':os.environ.get('CUDA_VISIBLE_DEVICES'),'normalization':json.loads((CACHE/'normalization.json').read_text()),'sources':json.loads((REPO/'docs/geowam-real/sources.lock.json').read_text())}
  provenance['adapter_git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
  provenance['data_audit']=json.loads((CACHE/'cache_audit.json').read_text())
  files=[*sorted((REPO/'scripts/geowam').glob('*.py')),*sorted((REPO/'configs/geowam').glob('*.json')),*sorted((REPO/'configs/geowam').glob('*.yaml'))]
  provenance['adapter_source_hashes']={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
  with zipfile.ZipFile(out/f'adapter_source_{int(time.time())}.zip','w',zipfile.ZIP_DEFLATED) as archive:
   for p in files:archive.write(p,p.relative_to(REPO))
  atomic_json(out/'real_contract.json',provenance)
 return report
trainer.initialize_model=initialize
_original_losses=trainer.compute_losses
def losses(*args,**kwargs):
 result=_original_losses(*args,**kwargs)
 if not torch.isfinite(result['total']).all():raise FloatingPointError('non-finite training objective')
 return result
trainer.compute_losses=losses
_original_clip=torch.nn.utils.clip_grad_norm_
def finite_clip(*args,**kwargs):
 kwargs['error_if_nonfinite']=True
 return _original_clip(*args,**kwargs)
torch.nn.utils.clip_grad_norm_=finite_clip

class FitVisualizer(Visualizer):
 def __init__(self,fit_batches,**kwargs):
  super().__init__(raw_batch=None,**kwargs);self.fit_batches=fit_batches
  self.stats=json.loads((CACHE/'normalization.json').read_text())
 def _batches(self,step):return [(task,move_batch(b,self.device,self.dtype)) for task,b in self.fit_batches]
 def _scalars(self,batch,result):
  out=super()._scalars(batch,result)
  if result.action is not None:
   std=torch.tensor(self.stats['action_std'],device=result.action.device);e=(result.action[...,10:17].float()-batch['action'][...,10:17].float())*std
   out['vis/joint_mae_rad']=float(e[...,:6].abs().mean());out['vis/gripper_accuracy']=float(((result.action[...,16]>0)==(batch['action'][...,16]>0)).float().mean())
  return out
 def _write(self,batch,result,step,scalars,focus=(),source=''):
  # Physical-unit curves keep all seven action dimensions and the full 32-step horizon.
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  folder=self.out_dir/'vis'/f'step_{step:07d}';folder.mkdir(parents=True,exist_ok=True)
  std=np.array(self.stats['action_std']);mean=np.array(self.stats['action_mean']);gt=batch['action'][0,:,10:17].float().cpu().numpy()*std+mean;pred=result.action[0,:,10:17].float().cpu().numpy()*std+mean
  fig,axs=plt.subplots(2,4,figsize=(12,5));tt=np.arange(1,33)/10
  for j in range(7):
   ax=axs.flat[j];ax.plot(tt,gt[:,j],label='recorded');ax.plot(tt,pred[:,j],label='predicted');ax.set_title(f'Joint {j+1} delta (rad)' if j<6 else 'Gripper closedness');ax.set_xlabel('seconds')
  axs.flat[0].legend();axs.flat[-1].axis('off');fig.suptitle(f'{source} training-fit probe at step {step}');fig.tight_layout();fig.savefig(folder/f'{source}_physical_actions.png');plt.close(fig)
  saved={'key':batch['key'],'action_gt':batch['action'].cpu(),'action_pred':result.action.cpu(),'video_gt':batch['video_clean'].cpu(),'video_pred':result.video.cpu(),'track_gt':batch['track_clean'].cpu(),'track_pred':result.track.cpu()};torch.save(saved,folder/f'{source}_predictions.pt')
  dest=folder/'scalars.json';current=json.loads(dest.read_text()) if dest.exists() else {};current.update({k.replace('vis/',f'fit/{source}/'):v for k,v in scalars.items()});atomic_json(dest,current)

def visualizer(stage,**kw):
 if not stage.training.visualize:return None
 fit=[]
 for comp in stage.data.components:
  ds=LatentShardDataset(comp.manifest,stage.data.spec,registry=real_registry());index=len(ds)//2;fit.append((comp.name,collate([ds[index]])))
 return FitVisualizer(fit_batches=fit,spec=stage.data.spec,schedule=kw['schedule'],rounds=stage.training.visualize_rounds,encoder=None,device=kw['device'],dtype=kw['dtype'],out_dir=kw['out_dir'],is_main=kw['is_main'],tb=kw['tb'],seed=1234,samples=1)
trainer.build_visualizer=visualizer
if __name__=='__main__':trainer.main()
