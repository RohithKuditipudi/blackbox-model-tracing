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
        "n_partial_0": [450000],
        "n_base": [500000],
        "n_finetune": [0, 100, 1000, 5000, 10000, 20000, 50000],
        "num_partial_models": [5],
        "n_sample": [10, 100, 1000, 5000, 10000],
        "sampling_seed": list(range(10)),
        "num_shuffles": [10],
        "partial_model_index": list(range(5)),
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
