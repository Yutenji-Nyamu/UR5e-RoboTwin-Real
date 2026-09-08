from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Native RoboTwin pi05 / real UR5e joint adapter")
    parser.add_argument("command", choices=("data", "train", "serve", "infer", "prepare"))
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    if args.command == "prepare":
        from ...config import load_config
        from .contract import read_contract
        from .runtime import prepare

        command = argparse.ArgumentParser(description="Dry-run by default; explicit slow joint home")
        command.add_argument("--lab-config", type=Path, required=True)
        command.add_argument("--contract", type=Path, required=True)
        command.add_argument("--execute", action="store_true")
        options = command.parse_args(rest)
        prepare(load_config(options.lab_config), read_contract(options.contract), execute=options.execute)
    else:
        sys.argv = [sys.argv[0], *rest]
        module = "process_data" if args.command == "data" else args.command
        importlib.import_module(f"{__package__}.{module}").main()


if __name__ == "__main__":
    main()
