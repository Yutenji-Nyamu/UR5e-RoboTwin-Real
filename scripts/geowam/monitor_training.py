"""Read-only run audit plus small derived health report and loss curves; uses CPU only."""
import json,math,os,time,statistics,subprocess
from pathlib import Path
from pipeline_common import ROOT,atomic_json
status_path=ROOT/'runs/training_status.json';state=json.loads(status_path.read_text()) if status_path.exists() else {'phase':'not_started'}
run=Path(state.get('output_dir',str(ROOT/'runs/train_rgbd72_v1')));run.mkdir(exist_ok=True)
records=[];log=run/'train_log.jsonl'
if log.exists():
 for line in log.read_text().splitlines():
  try:d=json.loads(line)
  except json.JSONDecodeError:continue
  if 'loss/total' in d:records.append(d)
by_step={int(d['step']):d for d in records};records=[by_step[k] for k in sorted(by_step)]
checks=[p for p in sorted((run/'checkpoints').glob('step_*')) if (p/'complete.json').exists()]
latest=records[-1] if records else {};recent=records[-20:];step=int(latest.get('step',0));timings=[d['step_time'] for d in recent if d.get('step_time',0)>0];seconds=statistics.median(timings) if timings else None
health={'phase':state['phase'],'observed_at':time.time(),'step':step,'max_steps':30000,'latest_loss':latest.get('loss/total'),'recent_step_seconds':seconds,'estimated_compute_hours_remaining':(30000-step)*seconds/3600 if seconds else None,'last_training_log_age_seconds':time.time()-latest['time'] if latest else None,'supervisor_heartbeat_age_seconds':time.time()-state.get('heartbeat',time.time()),'complete_checkpoints':[p.name for p in checks],'finite_loss_gradient':all(math.isfinite(float(v)) for d in records for k,v in d.items() if k.startswith('loss/') or k=='grad_norm'),'recent_means':{k:statistics.mean([d[k] for d in recent if k in d]) for k in ['loss/total','loss/action','loss/track','loss/video','grad_norm'] if any(k in d for d in recent)}}
for p in sorted((run/'vis').glob('step_*/scalars.json'))[-1:]:health['latest_fit_probe']=json.loads(p.read_text());health['latest_fit_probe_path']=str(p)
health['gpu_status']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).strip().splitlines()
atomic_json(ROOT/'runs/training_health.json',health)
if records:
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,3,figsize=(13,7));steps=[d['step'] for d in records]
 for ax,key in zip(axs.flat,['loss/total','loss/action','loss/track','loss/video','grad_norm','step_time']):
  vals=[d.get(key,float('nan')) for d in records];smooth=[statistics.mean(vals[max(0,i-19):i+1]) for i in range(len(vals))];ax.plot(steps,vals,alpha=.25,lw=.6);ax.plot(steps,smooth,lw=1.5);ax.set_title(key);ax.set_xlabel('optimizer steps');ax.grid(alpha=.2)
 fig.suptitle('UR5e RGB-D training | raw values and 20-log moving mean');fig.tight_layout();fig.savefig(run/'training_curves.png',dpi=140);plt.close(fig)
text=f"# 训练运行快照\n\n状态：{health['phase']}。步数：{step}/30000。\n\n最近损失：{health['recent_means']}。\n\n完整检查点：{', '.join(health['complete_checkpoints']) or '等待第一次保存'}。\n\n训练曲线：`{run/'training_curves.png'}`。\n\n五任务拟合曲线与预测：`{run/'vis'}`。\n\n本快照由 `monitor_training.py` 从日志生成；实时进度以 `training_status.json` 为准。\n"
(run/'STATUS.md').write_text(text);print(json.dumps({k:v for k,v in health.items() if k not in ['gpu_status','latest_fit_probe']},indent=2))
