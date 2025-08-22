import os
import subprocess
import itertools
import argparse

from tracing.launch import build_cmd

def main():
    parser = argparse.ArgumentParser(description="Sweep TinyStories experiment parameters and launch jobs.")

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")

    args = parser.parse_args()

    sweep_configs = {
        "n_partial": [400000],
        "n_retrain": [50000],
        "n_finetune": [100, 500, 1000],
        "n_epochs": [1, 3, 5],
        "save_dir": args.save_dir,
        "log_dir": args.log_dir,
    }

    param_names = list(sweep_configs.keys())
    param_values = [sweep_configs[name] for name in param_names]
    param_combinations = list(itertools.product(*param_values))

    if not args.dry_run:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

    for i, params in enumerate(param_combinations):

        cmd = build_cmd(dict(zip(param_names, params)))

        print(f"Launching experiment {i+1}/{len(param_combinations)}")
        print(f"Parameters: {dict(zip(param_names, params))}")
        print("Command:", cmd)
        if not args.dry_run:
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
