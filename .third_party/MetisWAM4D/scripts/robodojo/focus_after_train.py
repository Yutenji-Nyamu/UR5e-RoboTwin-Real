"""Export the final three complete checkpoints, evaluate, and write the focus routing report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from metiswam4d.eval.rdj_campaign import prepare, summary
from scripts.eval.rdj_paired_report import focus_report

TASK_CONFIGS = ['match_and_pick_from_conveyor','cover_blocks','build_tower','pour_balls_into_vase',
                'insert_tubes','play_tic_tac_toe','stack_blocks','stack_blocks_random','pour_liquid_into_cup',
                'pour_liquid_into_cup_random','fold_clothes','fold_clothes_random']
REFERENCE = Path('/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/stage3_robodojo_v3_anneal/simeval_34962_hide')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True,type=Path)
    ap.add_argument('--port',type=int,default=33880)
    ap.add_argument('--previous-focus',type=Path,action='append',default=[],
                    help='earlier round campaigns (repeatable); the round gain is measured against the best of them')
    ap.add_argument('--variants',default='{"base": {}, "hide": {"hide_track": true}}',
                    help='JSON {label: policy options}; a model whose Action never reads the Track only needs {"hide": ...}')
    args=ap.parse_args()
    variants=json.loads(args.variants)
    config=args.config.resolve()
    run=Path(yaml.safe_load(config.read_text())['output_dir'])
    out=run/'simeval_focus'
    out.mkdir(exist_ok=True)
    def status(stage,**data):
        state=dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**data)
        (run/'focus_pipeline_status.json').write_text(json.dumps(state,indent=2))
        print(json.dumps(state),flush=True)
    checkpoints=sorted(p for p in (run/'checkpoints').glob('step_*') if (p/'complete.json').exists())[-3:]
    if len(checkpoints)!=3:
        raise RuntimeError(f'Expected at least 3 complete checkpoints; got {checkpoints}')
    checkpoint_arg=','.join(map(str,checkpoints))
    model=out/'checkpoint/model_bf16_tail3.pt'
    env={**os.environ,'PYTHONPATH':str(PROJECT),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1',
         'RDJ_BASE_PORT':str(args.port)}
    status('export',checkpoints=[str(p) for p in checkpoints])
    subprocess.run(['/usr/bin/python3.10','scripts/eval/export_bf16.py','--config',str(config),
                    '--checkpoint',checkpoint_arg,'--out',str(model)],cwd=PROJECT,env=env,check=True)
    prepare(out=out,config=str(config),model_file=str(model),source_checkpoint=checkpoint_arg,
            episodes_per_round=2,execute_steps=32,clients_per_gpu=3,policy={},info=run.name,
            only=TASK_CONFIGS,episodes=10,video_episodes=3,policy_seed=42,
            policy_variants=variants,layout_offset=0,record_dir=None,env_seed=0)
    status('evaluate',output=str(out),episodes=120*len(variants),port=args.port)
    subprocess.run(['bash','scripts/eval/run_rdj_simeval.sh',str(out),'0,1,2,3,4,5,6,7','3'],
                   cwd=PROJECT,env=env,check=True)
    summary(out)
    candidates=list(args.previous_focus)+[out]
    result=focus_report(REFERENCE,candidates,out,REFERENCE.with_name('simeval_34962'))
    old=result['paired_totals']['old-hide']['success']
    labels=[k for k in result['paired_totals'] if k.startswith(f'{run.name}/{out.name}/')]
    best=max(labels,key=lambda k:(result['paired_totals'][k]['success'],result['paired_totals'][k]['score_sum']))
    gain=result['paired_totals'][best]['success']-old
    gain_previous=None
    if args.previous_focus:
        previous=[k for k in result['paired_totals'] if not k.startswith('old-') and k not in labels]
        gain_previous=result['paired_totals'][best]['success']-max(result['paired_totals'][k]['success'] for k in previous)
    # Sprint rule (2026-10-02): keep rolling data while each round beats the previous one; otherwise stop training.
    decision='next_round' if (gain_previous if gain_previous is not None else gain)>0 else 'stop_training'
    status('round_complete',best=best,paired_gain=gain,paired_gain_previous=gain_previous,decision=decision,
           overall=result['routed']['overall'],milestone=result['milestone'],
           paired=result['paired_totals'],expanded=result['expanded_totals'])
    report=out/'focus_routed_report.md'
    doc=PROJECT/'docs/2026-09-30-RoboDojo闭环仿真评测-实现与运行记录.md'
    with doc.open('a') as handle:
        handle.write(f"\n### {run.name} 闭环评测完成（{time.strftime('%Y-%m-%d %H:%M:%S')}）\n\n")
        handle.write(f"**{result['milestone']}**；路由Overall SR / Score = "
                     f"**{result['routed']['overall']['sr']:.2f} / {result['routed']['overall']['score']:.2f}**。\n\n")
        handle.write(f"新权重最佳方式 `{best}`，对旧hide的90条配对增益 {gain:+d}；"
                     f"相对上一轮增益 {gain_previous if gain_previous is not None else '首轮'}。决策：`{decision}`。\n\n")
        handle.write(f"完整逐配置及五维度表：`{report}`；原始结果与视频在同级`results/`，"
                     f"均值权重来源：`{checkpoint_arg}`。\n\n")
        handle.write(f"### 交接（{time.strftime('%Y-%m-%d %H:%M:%S')}）\n\n"
                     f"- 已完成：`{run.name}`训练、尾部3ckpt导出、{120*len(variants)}条（{'/'.join(variants)}）评测及420条路由报表。\n"
                     f"- 当前结论：{result['milestone']}，SR {result['routed']['overall']['sr']:.2f}；"
                     f"后续分支 `{decision}`，具体状态 `{run/'focus_pipeline_status.json'}`。\n"
                     f"- 下一步：确认t2保活，再按配对成绩进入采集/下一轮训练或物体Track独立对照；"
                     "继续使用固定90条配对交集，额外30条不计入420分母。\n")


if __name__=='__main__':
    main()
