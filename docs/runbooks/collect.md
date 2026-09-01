# Collect

1. Clear the workspace and keep a person at the emergency stop.
2. Confirm the robot mode required by the chosen freedrive workflow.
3. Run `ur5e-real doctor --config configs/lab.yaml --hardware`.
4. Run `ur5e-real collect --config configs/lab.yaml`.
5. Use `c` to close, `o` to open, and `q` to stop.
6. Convert only after checking the session manifest and both camera folders.

Raw output is written beneath the configured `data_root/raw` and is ignored by
Git. `--save-video` adds convenient MP4 previews; PNG frames remain the source
used for synchronized conversion.
