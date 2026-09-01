# Replay

Replay is dry-run by default. First inspect segmentation:

```bash
ur5e-real replay --config configs/lab.yaml ACTION.csv --gripper-events EVENTS.csv
```

Set `--max-segments 1` for the first physical test. Only after reviewing the
summary, switching PolyScope to Remote Control, clearing the workspace, and
staffing the emergency stop, add `--execute`.
