import json,pathlib,statistics,datetime,collections,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
R=pathlib.Path('/data/chenyiteng/projects/geowam-real'); P=R/'runs/performance_review_20261010'
TZ=datetime.timezone(datetime.timedelta(hours=8))
rows=[]
for line in (R/'runs/train_rgbd72_v1/train_log.jsonl').read_text().splitlines():
 try: x=json.loads(line)
 except json.JSONDecodeError: continue
 if 'loss/total' in x: rows.append(x)
steps=np.array([x['step'] for x in rows]);last=int(steps[-1]);recent=[x for x in rows if x['step']>last-1000]
metrics=['loss/total','loss/action','loss/track','loss/video']
intervals={}
for name,lo,hi in [('start',1,1000),('all_experts',1600,2600),('after_anneal',4000,5000),('middle',8000,9000),('recent',last-1000,last)]:
 a=[x for x in rows if lo<x['step']<=hi]
 intervals[name]={'steps':[lo,hi],**{k:statistics.mean(x[k] for x in a) for k in metrics}}
timing={k:statistics.mean(x[k] for x in recent) for k in ['step_time','time/data','time/encode','time/fwd_bwd','time/optim']}
timing['windows_per_second']=4/timing['step_time']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
colors=['#176B87','#D47729','#5372B7','#34866D']
fig,axs=plt.subplots(2,2,figsize=(13,7.4),layout='constrained')
for ax,k,c,title in zip(axs.flat,metrics,colors,['Total training loss','Action loss','Track loss','Video loss']):
 y=np.array([x[k] for x in rows]);smooth=np.array([y[max(0,i-19):i+1].mean() for i in range(len(y))])
 ax.plot(steps,y,color=c,alpha=.22,lw=.8,label='Logged 20-step average')
 ax.plot(steps,smooth,color=c,lw=2.1,label='400-step trailing average')
 ax.axvline(3000,color='#8A929B',ls=':',lw=1)
 ax.set(title=title,xlabel='Training step',ylabel='Loss',xlim=(0,last))
 ax.grid(alpha=.13)
 if k=='loss/total':ax.legend(fontsize=9,frameon=False)
fig.suptitle(f'GeoWAM | 72 RGB-D episodes | step {last:,} / 30,000',fontsize=17,weight='bold')
fig.supxlabel('Dashed line: dense-to-compact readout transition completed. Curves show optimization losses.',fontsize=10)
fig.savefig(P/'loss_curves.png',dpi=170);plt.close(fig)
fits=[]
for f in sorted((R/'runs/train_rgbd72_v1/vis').glob('step_*/scalars.json')):
 z=json.loads(f.read_text());st=int(f.parent.name.split('_')[1]);fits.append((st,z))
tasks=['block_drawer','blocks_box','sort_block','stack_blocks_two','plug_charger']; labels=['Drawer','Shoe box','Sorting','Stacking','Charger']
fitmetrics=['joint_mae_rad','gripper_accuracy','video_latent_mse','track_latent_mse']
fig,axs=plt.subplots(2,2,figsize=(13,7.4),layout='constrained')
for ax,m,title in zip(axs.flat,fitmetrics,['Joint error (6 joints x 32 future steps)','Gripper state accuracy (32 future steps)','Predicted RGB latent MSE','Predicted Track latent MSE']):
 for t,label in zip(tasks,labels):
  a=[(s,d['fit/'+t+'/'+m]) for s,d in fits if 'fit/'+t+'/'+m in d]
  ax.plot([s for s,y in a],[y for s,y in a],marker='o',markersize=4,lw=1.6,label=label)
 ax.set(title=title,xlabel='Checkpoint step',ylabel='Radians' if m=='joint_mae_rad' else ('Accuracy' if m=='gripper_accuracy' else 'Latent MSE'))
 ax.grid(alpha=.15)
 if m=='gripper_accuracy':ax.set_ylim(0,1.05)
 if m=='joint_mae_rad':ax.legend(fontsize=9,frameon=False,ncol=2)
fig.suptitle('Five fixed training windows | native 16-round prediction',fontsize=17,weight='bold')
fig.supxlabel('One window per task, unchanged across checkpoints. Gripper accuracy measures per-step open/close agreement.',fontsize=10)
fig.savefig(P/'fit_curves.png',dpi=170);plt.close(fig)
fit_summary=[]
for s,d in fits:
 fit_summary.append({'step':s,'means':{m:statistics.mean(d['fit/'+t+'/'+m] for t in tasks) for m in fitmetrics},'per_task':{t:{m:d['fit/'+t+'/'+m] for m in fitmetrics} for t in tasks}})
trainrows=[json.loads(s) for s in (R/'data/manifests/train_rgbd.jsonl').read_text().splitlines() if s.strip()]
suffix=collections.defaultdict(lambda:{'files':0,'bytes':0})
for x in trainrows:
 for root,dirs,files in os.walk(R/'data/raw'/x['path']):
  for name in files:
   q=pathlib.Path(root)/name;k=q.suffix or '<none>';suffix[k]['files']+=1;suffix[k]['bytes']+=q.stat().st_size
cache=[json.loads(s) for s in (R/'cache/real_v1/manifests/all.jsonl').read_text().splitlines() if s.strip()]
example=torch.load(cache[0]['path'],map_location='cpu',weights_only=False)
def tensor_info(x):
 if torch.is_tensor(x): return {'shape':list(x.shape),'dtype':str(x.dtype),'bytes':x.numel()*x.element_size()}
 if isinstance(x,dict):return {k:tensor_info(v) for k,v in x.items()}
 if isinstance(x,(str,int,float,bool)) or x is None:return x
 return str(type(x))
rawindex=json.loads((R/'data/raw/downloads.json').read_text());export=json.loads((R/'exports/processed-rgbd72-v1-20261010/processed_downloads.json').read_text())
sizes={'all_87_raw_release_bytes':sum(a['bytes'] for a in rawindex['assets']),'selected_72_raw_file_bytes':sum(x['bytes'] for x in suffix.values()),'selected_72_raw_by_suffix':dict(suffix),'active_2405_pt_bytes':sum(pathlib.Path(x['path']).stat().st_size for x in cache),'training_and_shared_archive_bytes':sum(a['bytes'] for a in export['assets'] if a['kind'] in ('train','shared')),'geometry_qa_archive_bytes':sum(a['bytes'] for a in export['assets'] if a['kind'] not in ('train','shared')),'windows':len(cache),'episodes':len(trainrows)}
summary={'captured_at':datetime.datetime.now(TZ).isoformat(),'latest_step':last,'latest_log_time':datetime.datetime.fromtimestamp(rows[-1]['time'],TZ).isoformat(),'loss_intervals':intervals,'recent_1000_step_timing':timing,'finite_loss_and_gradient':all(np.isfinite(x[k]) for x in rows for k in metrics+['grad_norm']),'fit':fit_summary,'sizes':sizes,'example_tensor_contents':tensor_info(example)}
(P/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('fit','example_tensor_contents')},ensure_ascii=False,indent=2));print('FIT_FIRST_LAST',json.dumps([fit_summary[0],fit_summary[-1]]));print('TENSOR_CONTENTS',json.dumps(summary['example_tensor_contents']))
gpu_file=P/'gpu_180s.json'
if gpu_file.exists():
 g=json.loads(gpu_file.read_text());gs={}
 fig,axs=plt.subplots(2,1,figsize=(13,5.6),sharex=True,layout='constrained')
 for gpu,c in [('6','#176B87'),('7','#D47729')]:
  a=[x for x in g['samples'] if x['index']==gpu];ts=np.array([x['time']-g['started'] for x in a]);util=np.array([float(x['utilization.gpu']) for x in a]);mem=np.array([float(x['memory.used'])/1024 for x in a]);free=np.array([float(x['memory.free'])/1024 for x in a])
  gs[gpu]={'n':len(a),'uuid':a[0]['uuid'],'gpu_util_mean':util.mean(),'gpu_util_min':util.min(),'gpu_util_max':util.max(),'used_GiB':mem.mean(),'free_GiB':free.mean(),'total_GiB':float(a[0]['memory.total'])/1024,'power_mean_W':statistics.mean(float(x['power.draw']) for x in a)}
  smooth=np.array([util[max(0,i-9):i+1].mean() for i in range(len(util))]);axs[0].plot(ts,util,c=c,alpha=.17,lw=.7);axs[0].plot(ts,smooth,c=c,lw=1.8,label=f'GPU {gpu}: mean {util.mean():.1f}%');axs[1].plot(ts,mem,c=c,lw=1.8,label=f'GPU {gpu}: {mem.mean():.1f} GiB used, {free.mean():.1f} GiB free')
 axs[0].set(ylabel='GPU utilization (%)',ylim=(0,105));axs[1].set(ylabel='Device memory (GiB)',xlabel='Elapsed seconds',ylim=(0,82));axs[1].axhline(81559/1024,c='#8A929B',ls=':',label='Total 79.6 GiB')
 for ax in axs:ax.grid(alpha=.13);ax.legend(frameon=False,fontsize=10)
 fig.suptitle('Physical GPU 6 / 7 | 180 seconds, 1 Hz sampling',fontsize=17,weight='bold');fig.supxlabel('Faint: 1-second readings. Solid utilization: 10-second trailing average. Source: nvidia-smi.',fontsize=10)
 fig.savefig(P/'gpu_usage.png',dpi=170);plt.close(fig)
 gs['started_CST']=datetime.datetime.fromtimestamp(g['started'],TZ).isoformat();gs['finished_CST']=datetime.datetime.fromtimestamp(g['finished'],TZ).isoformat();(P/'gpu_summary.json').write_text(json.dumps(gs,indent=2)+'\n');print('GPU_SUMMARY',json.dumps(gs))
else:print('GPU_180_PENDING')
