"""Inspect episode metadata without decoding or modifying recordings."""
import argparse,csv,json,statistics,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def rows(p):
 if not p.exists():return []
 with p.open(newline='') as f:return list(csv.DictReader(f))
def dt(rs):
 t=[float(x['controller_time_s']) for x in rs];g=[b-a for a,b in zip(t,t[1:])]
 return {'median_dt_s':statistics.median(g) if g else None,'max_gap_s':max(g) if g else None,'nonpositive_deltas':sum(x<=0 for x in g)}
index={x['run_id']:x for x in rows(a.data/'index.csv')};out=[]
for p in sorted(a.data.rglob('session.json')):
 s=json.loads(p.read_text());d=p.parent;cal=d/'images/camera_calibration.json';c=json.loads(cal.read_text()) if cal.exists() else {};rgbd=rows(d/'images/rgbd_frames.csv');r=rows(d/'action.csv');v=rows(d/'sync.csv');review=index.get(s['run_id'],{}).get('visual_review','')
 flags=[name for name,words in {'lighting':['光照','彩色光'],'tablecloth':['绿色图案桌布','背景变体'],'appearance':['黄色方块变体','蓝色方块变体']}.items() if any(w in review for w in words)]
 out.append({'run_id':s['run_id'],'task':s['task'],'path':str(d.relative_to(a.data)),'success':s.get('outcome'),'depth':s.get('depth_recording',{}).get('enabled',False),'state_rows':len(r),'sync_rows':len(v),'state_timing':dt(r),'visual_timing':dt(v),'joint_fields':all('actual_q_'+str(i) in r[0] for i in range(6)) if r else False,'camera_save_hz':s.get('cameras',{}).get('save_hz'),'rtde_hz':s.get('robot',{}).get('rtde_frequency_hz'),'depth_scale':{k:z.get('depth_scale_m_per_unit') for k,z in c.get('cameras',{}).items()},'minimum_depth_fraction':{k:min(float(z['valid_depth_fraction']) for z in rgbd if z['camera']==k) for k in sorted(set(z['camera'] for z in rgbd))},'condition_review_flags':flags,'visual_review':review,'split':'needs_review'})
summary={'episodes':len(out),'rgbd':sum(bool(x['depth']) for x in out),'rgb_only':sum(not x['depth'] for x in out),'task_counts':dict(collections.Counter(x['task'] for x in out)),'save_hz':sorted(set(x['camera_save_hz'] for x in out)),'state_gap_over_0p15s':sum(x['state_timing']['max_gap_s']>.15 for x in out),'condition_flagged_episodes':sum(bool(x['condition_review_flags']) for x in out),'all_joint_fields':all(x['joint_fields'] for x in out)}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'summary':summary,'episodes':out},ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
