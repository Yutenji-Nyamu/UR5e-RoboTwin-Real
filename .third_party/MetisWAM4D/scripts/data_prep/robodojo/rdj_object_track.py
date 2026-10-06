"""CPU object Track4D overlay from existing four-phase RAFT and DA3 assets.

Original RGB/robot depth and FK tracks remain read-only. The overlay keeps exact
FK displacement at robot pixels and writes object displacement in a new root.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from multiprocessing import Pool
import os
from pathlib import Path
import time

import cv2
import h5py
import numpy as np

ORIGINAL = Path('/ytech_milm_intern/danglingwei/datas/RDJ_MetisWAM4D')
IMPERFECT = Path('/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRoboDojo_JanusAct4D_Imperfect')
TASKS = ('match_and_pick_from_conveyor', 'cover_blocks', 'build_tower', 'pour_balls_into_vase',
         'insert_tubes', 'play_tic_tac_toe', 'stack_blocks', 'pour_liquid_into_cup', 'fold_clothes')


def rows(tasks=TASKS):
    return [r for l in (ORIGINAL / 'index.jsonl').read_text().splitlines()
            if (r := json.loads(l))['task'] in tasks]


def paths(row):
    task, ep = row['task'], f"episode{row['episode']}"
    return (ORIGINAL / task / row['variant'] / ep,
            IMPERFECT / '_pipeline/work' / task / ep,
            IMPERFECT / task / 'train_4d/masks' / f'{ep}.h5')


def qc(out):
    """Scale and camera alignment determine whether native DA3 differences are usable."""
    records = []
    for task in TASKS:
        candidates = [r for r in rows() if r['task'] == task]
        for row in [candidates[0], candidates[len(candidates)//2], candidates[-1]]:
            original, work, _ = paths(row)
            try:
                with h5py.File(original / 'source.hdf5') as src, h5py.File(work / 'da3.h5') as da3:
                    group = src['observation/head_camera']
                    n = group['depth'].shape[0]
                    if len(da3['depth_m']) != n:
                        raise ValueError('frame count differs')
                    ratios, errors = [], []
                    kd, ed = [], []
                    for t in [0, n//2, n-1]:
                        gt = group['depth'][t].astype(np.float32) * .001
                        pred = da3['depth_m'][t].astype(np.float32)
                        mask = (group['instance_id'][t] > 0) & (gt > 0) & np.isfinite(pred) & (pred > 0)
                        ratios.append(pred[mask] / gt[mask])
                        errors.append(np.abs(pred[mask] - gt[mask]))
                        kd.append(float(np.max(np.abs(group['intrinsic_cv'][t] - da3['intrinsics'][t]))))
                        ed.append(float(np.max(np.abs(group['extrinsic_cv'][t] - da3['extrinsics_w2c'][t][:3]))))
                    ratio, err = np.concatenate(ratios), np.concatenate(errors)
                    result = dict(task=task, episode=row['episode'], pixels=len(ratio),
                                  ratio_p10_p50_p90=np.percentile(ratio, [10,50,90]).tolist(),
                                  abs_depth_error_m_p50_p90=np.percentile(err,[50,90]).tolist(),
                                  max_intrinsic_diff=max(kd), max_extrinsic_diff=max(ed))
            except (OSError, KeyError, ValueError) as exc:
                result = dict(task=task, episode=row['episode'], error=repr(exc))
            records.append(result)
            print(json.dumps(result), flush=True)
    (out / 'depth_scale_qc.json').write_text(json.dumps(records, indent=2))


def build_one(job):
    row, output, threshold, shift = job
    original, work, masks = paths(row)
    dest = Path(output) / row['task'] / row['variant'] / f"episode{row['episode']}"
    started = time.time()
    try:
        if (dest / 'object_meta.json').exists():
            return dict(task=row['task'], episode=row['episode'], status='exists')
        cv2.setNumThreads(1)
        dest.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            fk = stack.enter_context(h5py.File(original / 'track4d.h5'))
            da3 = stack.enter_context(h5py.File(work / 'da3.h5'))
            sam = stack.enter_context(h5py.File(masks))
            flows = [stack.enter_context(h5py.File(work / 'flow' / f'phase{p}.h5')) for p in range(4)]
            n, h, w = fk['role'].shape
            if da3['depth_m'].shape != (n,h,w) or sam['head_camera/mask_bits'].shape[0] != n:
                raise ValueError('DA3/mask/FK frame or image shape mismatch')
            indices = {int(t): (p,i) for p,f in enumerate(flows) for i,t in enumerate(f['source_frame_index'][:])}
            if set(range(n-4)) - indices.keys():
                raise ValueError('four-phase flow does not cover all transitions')
            # Read one episode at a time. Four workers keep CPU and memory modest during training.
            depth = da3['depth_m'][:].astype(np.float32)
            yy, xx = np.mgrid[:h,:w].astype(np.float32)
            # episodes whose grounding found no manipulated object fall back to flow-only object pixels
            flow_only = row['task'] in ('cover_blocks','stack_blocks') or not sam['head_camera/mask_bits'][:,1].any()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
            tmp = dest / 'track4d.tmp.h5'
            objects, robots, invalid = 0, 0, 0
            with h5py.File(tmp,'w') as target:
                target.attrs.update(dict(fk.attrs))
                target.attrs.update(schema='metiswam4d.rdj_track4d.uvd.objects.v2', complete=False,
                                    role_labels='0 stationary background, 1 robot, 2 moving object',
                                    object_source=str(work), flow_threshold_px=threshold, rgb_shift=shift,
                                    object_depth='native DA3 t+4 warped minus t; FK robot unchanged')
                for name in fk:
                    if name not in ('delta_uvd','role'):
                        fk.copy(name,target)
                duvd = target.create_dataset('delta_uvd',(n-4,h,w,3),dtype='f2',chunks=(1,h,w,3),compression='lzf',shuffle=True)
                role = target.create_dataset('role',(n,h,w),dtype='u1',chunks=(1,h,w),compression='lzf')
                for t in range(n):
                    robot = fk['role'][t] == 1
                    labels = robot.astype(np.uint8)
                    robots += int(robot.sum())
                    if t < n-4:
                        delta = fk['delta_uvd'][t].astype(np.float32)
                        delta[~robot] = 0
                        # image frame k = t + shift shows joint state t (FK row t)
                        k = t + shift
                        if k < n-4:
                            p,i = indices[k]
                            flow = flows[p]['forward_flow_px'][i].astype(np.float32)
                            u, v = xx+flow[...,0], yy+flow[...,1]
                            future = cv2.remap(depth[k+4],u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                            valid = (np.isfinite(flow).all(-1) & np.isfinite(future) & np.isfinite(depth[k]) &
                                     (future>0) & (depth[k]>0) & (u>=0) & (u<=w-1) & (v>=0) & (v<=h-1))
                            moving = np.linalg.norm(flow,axis=-1) > threshold
                            if flow_only:
                                moving &= ~cv2.dilate(robot.astype(np.uint8),kernel).astype(bool)
                            else:
                                bits = sam['head_camera/mask_bits'][k,1]
                                moving &= np.unpackbits(bits,axis=-1,count=w,bitorder='little').astype(bool)
                            obj = moving & valid & ~robot
                            labels[obj] = 2
                            delta[obj,:2] = flow[obj]
                            delta[obj,2] = (future-depth[k])[obj]
                            objects += int(obj.sum())
                            invalid += int((moving & ~valid & ~robot).sum())
                        duvd[t] = delta.astype(np.float16)
                    role[t] = labels
                target.attrs['complete'] = True
            os.replace(tmp,dest/'track4d.h5')
        link = dest/'source.hdf5'
        if not link.exists():
            link.symlink_to(original/'source.hdf5')
        result = dict(task=row['task'],episode=row['episode'],status='ok',frames=n,flow_only=bool(flow_only),
                      object_pixels_per_transition=objects/(n-4),robot_pixels_per_frame=robots/n,
                      invalid_moving_pixels=invalid,seconds=round(time.time()-started,1))
        (dest/'object_meta.json').write_text(json.dumps(result))
        return result
    except (OSError, KeyError, ValueError) as exc:
        return dict(task=row['task'],episode=row['episode'],status='error',error=repr(exc))


def signal_qc(out):
    """Measure whether moving-object labels survive the actual latent/token pooling."""
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    import torch
    from metiswam4d.data.rt2.online_encoder import center_pad, majority_pool
    result=[]
    for task in TASKS:
        path=out/task/'train_4d/episode0'
        if not (path/'object_meta.json').exists():continue
        with h5py.File(path/'track4d.h5') as f:
            n=f['role'].shape[0]
            windows=[]
            for s in np.linspace(0,n-33,5,dtype=int):
                roles=np.stack([f['role'][int(s+4*k)] for k in range(8)])
                nine=np.concatenate(((roles[:1]==1).astype('u1'),roles))
                padded=center_pad(torch.from_numpy(nine)[None,...,None],256,320)[...,0]
                grouped=torch.stack([padded[:,a:z].max(1).values for a,z in [(0,1),(1,5),(5,9)]],dim=1)
                latent=majority_pool(grouped,(16,20))
                token=majority_pool(padded[:,1:],(8,10))
                windows.append(dict(start=int(s),object_pixels=int((roles==2).sum()),
                                    object_latent_cells=int((latent[:,1:]==2).sum()),
                                    object_role_tokens=int((token==2).sum())))
        record=dict(task=task,episode=0,windows=windows,
                    windows_with_object=sum(w['object_pixels']>0 for w in windows),
                    windows_with_object_latent=sum(w['object_latent_cells']>0 for w in windows),
                    windows_with_object_role=sum(w['object_role_tokens']>0 for w in windows))
        result.append(record);print(json.dumps(record),flush=True)
    (out/'signal_qc.json').write_text(json.dumps(result,indent=2))


def delta_qc(out):
    """Compare DA3 depth differences against FK on visible, moving robot points.

Absolute depth bias can cancel in a difference. This check measures differences
directly; the robot is a proxy where exact geometry is available.
"""
    result=[]
    cv2.setNumThreads(1)
    for task in TASKS:
        selected=[r for r in rows() if r['task']==task]
        for row in [selected[0],selected[len(selected)//2],selected[-1]]:
            original,work,_=paths(row)
            try:
                with h5py.File(original/'source.hdf5') as source, h5py.File(original/'track4d.h5') as fk, \
                     h5py.File(work/'da3.h5') as da3:
                    n,h,w=fk['role'].shape
                    yy,xx=np.mgrid[:h,:w].astype('f4')
                    errors=[];motion=[];boundary=[]
                    for t in sorted(set([0,28,31,32,n//2,n-5])):
                        if t+4>=n:continue
                        delta=fk['delta_uvd'][t].astype('f4')
                        u,v=xx+delta[...,0],yy+delta[...,1]
                        g=source['observation/head_camera/depth']
                        gt0=g[t].astype('f4')*.001
                        gt1=cv2.remap(g[t+4].astype('f4')*.001,u,v,cv2.INTER_LINEAR)
                        d0=da3['depth_m'][t].astype('f4')
                        d1=cv2.remap(da3['depth_m'][t+4].astype('f4'),u,v,cv2.INTER_LINEAR)
                        valid=((fk['role'][t]==1)&(gt0>0)&(gt1>0)&(d0>0)&(d1>0)&
                               np.isfinite(d0)&np.isfinite(d1)&(np.abs(gt1-gt0-delta[...,2])<.003)&
                               (np.linalg.norm(delta[...,:2],axis=-1)>1))
                        err=np.abs(d1-d0-delta[...,2])[valid]
                        errors.append(err);motion.append(np.abs(delta[...,2][valid]))
                        if t%32>=28:boundary.append(err)
                    err=np.concatenate(errors);gt=np.concatenate(motion)
                    rec=dict(task=task,episode=row['episode'],pixels=len(err),
                             delta_abs_error_mm_p50_p90_p99=(np.percentile(err,[50,90,99])*1000).tolist() if len(err) else None,
                             gt_abs_delta_mm_p50_p90=(np.percentile(gt,[50,90])*1000).tolist() if len(gt) else None,
                             boundary_abs_error_mm_p50_p90=(np.percentile(np.concatenate(boundary),[50,90])*1000).tolist()
                             if boundary and sum(map(len,boundary)) else None)
            except (OSError,KeyError,ValueError) as exc:
                rec=dict(task=task,episode=row['episode'],error=repr(exc))
            result.append(rec);print(json.dumps(rec),flush=True)
    (out/'depth_delta_qc.json').write_text(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['qc','build','finalize','signal-qc','delta-qc'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--dataset',type=Path,help='focused self-play dataset to overlay (finalize)')
    ap.add_argument('--tracks',type=Path,help='completed object track root (finalize)')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--flow-threshold',type=float,default=1.0)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--tasks',default=','.join(TASKS),help='comma-separated tasks (build)')
    ap.add_argument('--rgb-shift',type=int,default=0,
                    help='build: the image frame showing joint state t is t + shift (1 for the pre-2026-09-15 release '
                         'when the reader shifts RGB the same way)')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.command=='qc':
        qc(args.out)
    elif args.command=='signal-qc':
        signal_qc(args.out)
    elif args.command=='delta-qc':
        delta_qc(args.out)
    elif args.command=='finalize':
        if args.dataset is None or args.tracks is None:
            ap.error('finalize requires --dataset and --tracks')
        entries=[json.loads(l) for l in (args.dataset/'index.jsonl').read_text().splitlines()]
        seen=set(); replaced=0; fallback=0
        for row in entries:
            rel=Path(row['task'])/row['variant']/f"episode{row['episode']}"
            if rel in seen: continue
            seen.add(rel)
            track=args.tracks/rel
            use_object=(track/'object_meta.json').exists() and not row.get('rollout')
            source=track if use_object else args.dataset/rel
            link=args.out/rel
            link.parent.mkdir(parents=True,exist_ok=True)
            if not link.is_symlink() and not link.exists():link.symlink_to(source)
            replaced+=int(use_object)
            fallback+=int(not use_object)
        for name in ('text_cache','depth_stats.json','eef20_stats.json','uvd_stats.json','assets',
                     'episode_instructions_official.jsonl'):
            link=args.out/name
            if not link.exists() and not link.is_symlink():link.symlink_to(args.dataset/name)
        (args.out/'index.jsonl').write_text((args.dataset/'index.jsonl').read_text())
        info=dict(dataset=str(args.dataset),tracks=str(args.tracks),object_episodes=replaced,
                  robot_only_episodes=fallback,index_rows=len(entries))
        (args.out/'object_overlay.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
    else:
        selected=rows(tuple(args.tasks.split(',')))[:args.limit]
        with Pool(args.workers) as pool, (args.out/'build.jsonl').open('a') as log:
            jobs=[(r,str(args.out),args.flow_threshold,args.rgb_shift) for r in selected]
            for result in pool.imap_unordered(build_one,jobs):
                line=json.dumps(result)
                log.write(line+'\n');log.flush();print(line,flush=True)


if __name__=='__main__':
    main()
