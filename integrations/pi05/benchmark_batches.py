"""Small isolated-process probes; never starts a full SFT run or touches hardware."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from ur5e_real.adapters.robotwin_pi05.dataset import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-first-checkpoint", action="store_true")
    args = parser.parse_args()
    if (
        not 2 <= args.steps <= 5
        or not args.batches
        or len(args.batches) > 4
        or len(set(args.batches)) != len(args.batches)
    ):
        raise ValueError("use 1..4 distinct batch sizes and 2..5 steps per batch")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for ordinal, batch in enumerate(args.batches):
        name = f"{args.name}_b{batch}"
        command = [
            sys.executable,
            "-m",
            "ur5e_real.adapters.robotwin_pi05",
            "train",
            "--dataset",
            str(args.dataset),
            "--params",
            args.params,
            "--exp-name",
            name,
            "--steps",
            str(args.steps),
            "--batch-size",
            str(batch),
            "--warmup-steps",
            "0",
            "--schedule-steps",
            "3000",
            "--no-image-augmentation",
            "--num-workers",
            str(args.num_workers),
            "--log-base",
            str(args.output / "runs"),
            "--eval-points",
            "1",
        ]
        verify = ordinal == 0 and args.verify_first_checkpoint
        if not verify:
            command.append("--benchmark")
        started = time.monotonic()
        print(f"[PROBE] batch={batch}; optimizer_steps={args.steps}; checkpoint_test={verify}", flush=True)
        with (args.output / f"batch_{batch}.log").open("x", buffering=1) as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        row = {
            "batch_size": batch,
            "num_workers": args.num_workers,
            "exit_code": completed.returncode,
            "wall_s": time.monotonic() - started,
            "log": str(args.output / f"batch_{batch}.log"),
        }
        matches = sorted((args.output / "runs" / name).glob("*/result.json"))
        if completed.returncode == 0 and len(matches) == 1:
            row["result"] = json.loads(matches[0].read_text())
            row["resources"] = json.loads((matches[0].parent / "resources_summary.json").read_text())
        else:
            row["failure_tail"] = (args.output / f"batch_{batch}.log").read_text()[-3000:]
        rows.append(row)
        print(
            "[PROBE-DONE] "
            + json.dumps(
                {
                    "batch_size": batch,
                    "exit_code": completed.returncode,
                    "wall_s": row["wall_s"],
                    "warm_samples_per_s": row.get("result", {}).get("warm_samples_per_s"),
                }
            ),
            flush=True,
        )
        if completed.returncode != 0:
            # Do not blindly repeat a common environment/checkpoint failure for larger batches.
            break
    write_json(
        args.output / "summary.json",
        {
            "scope": "few_step_coarse_probe_not_convergence",
            "optimizer_steps_per_batch": args.steps,
            "physical_execution": False,
            "rows": rows,
        },
    )
    if any(row["exit_code"] != 0 for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
