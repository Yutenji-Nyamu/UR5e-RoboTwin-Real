"""Per-task comparison of a RoboDojo campaign with the local baselines evaluated under the same 420-episode protocol
(OpenWAM-Alpha 60k and JanusAct4D-RDJ-ipft 40k, from the JanusTrack4d_260824 paired-report JSONs).

    PYTHONPATH=. /usr/bin/python3.10 scripts/eval/rdj_paired_report.py --output <campaign dir> [--output <dir2> ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metiswam4d.eval.rdj_campaign import DIM_ORDER, aggregate, read_details  # noqa: E402

BASELINES = {
    "Alpha-60k": ("/ytech_milm_intern/danglingwei/outputs/JanusAct4D_imperfect/rdj_v1/objects_v1_from_step7000/"
                  "diagnosis_20260916/paired_sync_results.json", "OpenWAM-Alpha"),
    "Janus-40k": ("/ytech_milm_intern/danglingwei/outputs/JanusAct4D_imperfect/rdj_v1/eval_40000_fixed_20260918/"
                  "paired_visible_vs_alpha.json", None),
}


def focus_report(base_out: Path, candidate_outs: list[Path], report_dir: Path,
                 old_base_out: Path | None = None) -> dict:
    """Route whole base tasks, comparing exactly the layouts in the original 420 protocol."""
    campaign = json.loads((base_out / 'campaign.json').read_text())
    base = {v['name']: read_details(base_out, v['name']) for v in campaign['variants']}
    if sum(map(len, base.values())) != 420:
        raise ValueError('Routing reference must contain the complete 420 episode protocol')
    candidates, expanded = {'old-hide': base}, {}
    focus = set()
    for out in candidate_outs:
        new = json.loads((out / 'campaign.json').read_text())
        for v in new['variants']:
            label = f"{out.parent.name}/{out.name}/{v.get('label', 'base')}"
            task = v['task']
            focus.add(task)
            rows = read_details(out, v['name'])
            if len(rows) != v['episodes']:
                raise ValueError(f"Incomplete candidate: {out}/{v['name']} {len(rows)}/{v['episodes']}")
            ids = {int(r['layout_id']) for r in base[task]}
            paired = [r for r in rows if int(r['layout_id']) in ids]
            if len(paired) != len(ids):
                raise ValueError(f'Candidate does not cover reference layouts: {task}')
            candidates.setdefault(label, {})[task] = paired
            expanded.setdefault(label, {})[task] = rows
    if old_base_out is not None:
        old_base = {name: read_details(old_base_out, name) for name in focus}
        for name, values in old_base.items():
            if {int(r['layout_id']) for r in values} != {int(r['layout_id']) for r in base[name]}:
                raise ValueError(f'Old base layout mismatch: {name}')
        candidates = {'old-hide': base, 'old-base': old_base,
                      **{k:v for k,v in candidates.items() if k != 'old-hide'}}
    def tally(rows):
        return dict(success=sum(bool(r['success']) for r in rows), n=len(rows),
                    score_sum=sum(float(r.get('score', 0)) for r in rows))
    paired_totals = {label: tally([r for name in focus for r in data.get(name,[])])
                     for label,data in candidates.items()}
    expanded_totals = {label: tally([r for rows in data.values() for r in rows])
                       for label,data in expanded.items()}
    route, routed = {}, dict(base)
    for task in dict.fromkeys(v['base_task'] for v in campaign['variants'] if v['name'] in focus):
        names = [v['name'] for v in campaign['variants'] if v['base_task']==task]
        eligible = [label for label,data in candidates.items() if all(name in data for name in names)]
        def metric(label):
            values=tally([r for name in names for r in candidates[label][name]])
            return values['success'],values['score_sum'],label=='old-hide'
        best=max(eligible,key=metric)
        route[task]=best
        for name in names:
            routed[name]=candidates[best][name]
    # two-model fusion: every focus task from one candidate, the other 33 tasks from the old model (no per-task choice)
    fused={}
    for label,data in candidates.items():
        if label=='old-hide' or not all(name in data for name in focus): continue
        fused[label]=aggregate(campaign,{**base,**{name:data[name] for name in focus}},None)
    result=dict(reference=str(base_out), paired_totals=paired_totals, expanded_totals=expanded_totals,
                route=route, routed=aggregate(campaign,routed,None), old=aggregate(campaign,base,None), fused=fused)
    sr=result['routed']['overall']['sr']
    result['milestone']=('冲刺达成（SR≥22）' if sr>=22 else '期望档（SR≥16）' if sr>=16 else
                         '基础要求达成（SR≥13）' if sr>=13 else '超过Alpha官方，但未达基础要求' if sr>11.92 else
                         '未达到基础要求')
    report_dir.mkdir(parents=True,exist_ok=True)
    lines=['# 重点任务配对与任务级路由', '',
           '原420协议中的重点子集为90条：6个独立配置各10条，6个Generalization配置各5条。',
           '扩展120条中的额外30条单列，不计入420条路由；standard/random共享同一基础任务路由。', '',
           '| 配置 | '+' | '.join(candidates)+' | 路由 |',
           '|---|'+'---:|'*len(candidates)+'---|']
    for v in campaign['variants']:
        name=v['name']
        if name not in focus: continue
        values=[]
        for data in candidates.values():
            t=tally(data.get(name,[]));values.append(f"{t['success']}/{t['n']}")
        lines.append('| '+name+' | '+' | '.join(values)+' | '+route[v['base_task']]+' |')
    lines+=['', '## 严格配对与扩展结果', '', '| 候选 | 配对成功 | 对旧模型增量 | 扩展成功 |', '|---|---:|---:|---:|']
    for label,t in paired_totals.items():
        ex=expanded_totals.get(label)
        lines.append(f"| {label} | {t['success']}/{t['n']} | {t['success']-paired_totals['old-hide']['success']:+d} | "+
                     (f"{ex['success']}/{ex['n']}" if ex else '未测')+' |')
    bases={name:baseline_tasks(*spec) for name,spec in BASELINES.items()}
    baseline_aggs={}
    for name,tasks in bases.items():
        dims={}
        for dim,names in campaign['dimensions'].items():
            entries=[tasks[t] for t in names if t in tasks]
            dims[dim]=100*sum(t['successes']/t['completed'] for t in entries)/len(entries)
        baseline_aggs[name]=dims
    lines+=['','## 420条口径', '', '| 维度 | 旧 SR / Score | 路由 SR / Score | Alpha本地 SR | Janus40k SR |',
            '|---|---:|---:|---:|---:|']
    for dim in DIM_ORDER:
        a,b=result['old']['dimensions'][dim],result['routed']['dimensions'][dim]
        lines.append(f"| {dim} | {a['sr']:.2f} / {a['score']:.2f} | {b['sr']:.2f} / {b['score']:.2f} | "+
                     f"{baseline_aggs['Alpha-60k'][dim]:.2f} | {baseline_aggs['Janus-40k'][dim]:.2f} |")
    a,b=result['old']['overall'],result['routed']['overall']
    lines.append(f"| Overall | {a['sr']:.2f} / {a['score']:.2f} | {b['sr']:.2f} / {b['score']:.2f} | 9.58 / 15.40 | 10.75 / 16.82 |")
    lines+=['', '## 两模型融合（9 个重点任务整体用同一新模型，其余 33 任务用旧 hide；不逐任务挑选）', '',
            '| 新模型 | 成功数 /420 | Overall SR / Score | Gen | Precision | LH | Memory | Open |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,agg in fused.items():
        d=agg['dimensions']
        lines.append(f"| {label} | {agg['successes']} | {agg['overall']['sr']:.2f} / {agg['overall']['score']:.2f} | "+
                     ' | '.join(f"{d[k]['sr']:.2f}" for k in DIM_ORDER)+' |')
    lines+=['', 'OpenWAM-α 官方 SR / Score：11.92 / 17.18。官方全量榜与本地缩减协议仅作近似比较。',
            '路由在这批评测结果上选择，属于任务级部署候选的回顾性结果；未提供独立布局验证的统计显著性。',
            f"路由成功数：{result['routed']['successes']}/420。",f"当前档位：**{result['milestone']}**。"]
    (report_dir/'focus_routed_report.json').write_text(json.dumps(result,indent=2))
    (report_dir/'focus_routed_report.md').write_text('\n'.join(lines)+'\n')
    return result


def baseline_tasks(path: str, run_name: str | None) -> dict[str, dict]:
    runs = json.loads(Path(path).read_text())["runs"]
    if run_name is None:
        run_name = [k for k in runs if "Alpha" not in k][0]
    return runs[run_name]["tasks"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", action="append")
    ap.add_argument("--route-base", type=Path)
    ap.add_argument("--route-old-base", type=Path)
    ap.add_argument("--focus-output", action="append", type=Path)
    ap.add_argument("--report-dir", type=Path)
    args = ap.parse_args()
    if args.focus_output:
        if args.route_base is None or args.report_dir is None:
            ap.error('--focus-output requires --route-base and --report-dir')
        result=focus_report(args.route_base,args.focus_output,args.report_dir,args.route_old_base)
        print(json.dumps({'paired':result['paired_totals'],'routed':result['routed']['overall']}))
        return
    if not args.output:
        ap.error('--output or --focus-output is required')
    columns = {}
    for out in args.output:
        out = Path(out)
        campaign = json.loads((out / "campaign.json").read_text())
        details = {v["name"]: read_details(out, v["name"]) for v in campaign["variants"]}
        columns[out.name] = aggregate(campaign, details, None)
    bases = {name: baseline_tasks(*spec) for name, spec in BASELINES.items()}
    first = columns[next(iter(columns))]
    lines = ["| Dimension | Task | " + " | ".join(columns) + " | " + " | ".join(bases) + " |",
             "|---|---|" + "---:|" * (len(columns) + len(bases))]
    for dim in DIM_ORDER:
        for task in campaign["dimensions"][dim]:
            row = [f"{c['tasks'][task]['success']}/{c['tasks'][task]['n']}" if task in c["tasks"] else "—"
                   for c in columns.values()]
            for b in bases.values():
                t = b.get(task)
                row.append(f"{t['successes']}/{t['completed']}" if t else "—")
            lines.append(f"| {dim} | {task} | " + " | ".join(row) + " |")
    totals = [f"{c['successes']}/{c['episodes']} (SR {c['overall']['sr']:.2f}, score {c['overall']['score']:.2f})"
              for c in columns.values()]
    lines.append("| | **total** | " + " | ".join(totals) + " | 41/420 (SR 9.58, score 15.40) | 46/420 (SR 10.75, score 16.82) |")
    text = "\n".join(lines) + "\n"
    for out in args.output:
        (Path(out) / "paired_vs_baselines.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
