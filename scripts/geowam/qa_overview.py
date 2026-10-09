"""Inspectable first/middle/last role overlays for every processed episode."""
import json,zipfile
import numpy as np
from PIL import Image,ImageDraw
from pipeline_common import *
out=ROOT/'evidence/geometry_overview';out.mkdir(exist_ok=True);reports=[];images=[]
cfg=json.loads((CONFIG/'perception.json').read_text())
for row in rows():
 ep=CACHE/row['key'];p=ep/'geometry.json'
 if not p.exists():continue
 d=json.loads(p.read_text())
 if d.get('perception_signature')!=perception_signature(cfg,row['task'],row['key']):continue
 m=np.load(ep/'metadata.npz');g=np.load(ep/'masks_full.npz');panels=[]
 for k in [0,len(g['indices'])//2,len(g['indices'])-1]:
  rgb=read_rgb(row,m['names'][g['indices'][k]]);role=g['role'][k];a=rgb.copy()
  for val,col in [(1,(255,70,70)),(2,(60,220,70))]:
   mask=role==val;a[mask]=(a[mask]*.55+np.array(col)*.45).astype('uint8')
  panels.append(Image.fromarray(a).resize((320,240)))
 images.append((row,panels));q=d['quality'];reports.append({'episode':row['key'],'task':row['task'],'mean_valid_fg':d['mean_valid_fg'],'mean_valid_object':d['mean_valid_object'],'mean_valid_body':float(np.mean([v['valid_body'] for v in q])),'static_depth_p90_median':float(np.median([v['static_depth_abs_p90'] for v in q])),'clip_mean':float(np.mean([v['clip_fraction'] for v in q])),'missing':d['missing_initial']})
for p in range((len(images)+5)//6):
 sheet=Image.new('RGB',(960,270*6),'white');draw=ImageDraw.Draw(sheet)
 for i,(row,ims) in enumerate(images[p*6:(p+1)*6]):
  draw.text((5,i*270+6),row['key']+' '+row['task']+' | red robot; green objects',fill='black')
  for j,im in enumerate(ims):sheet.paste(im,(j*320,i*270+28))
 sheet.save(out/f'page{p:02d}.jpg',quality=88)
atomic_json(out/'quality.json',reports)
with zipfile.ZipFile(ROOT/'evidence/geometry_overview.zip','w') as z:
 for p in out.glob('*.jpg'):z.write(p,p.name)
 z.write(out/'quality.json','quality.json')
print('QA_OVERVIEW',len(reports),flush=True)
