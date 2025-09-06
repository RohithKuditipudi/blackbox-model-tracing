import os
import subprocess
import itertools
import argparse

from tracing.launch import build_cmd

def main():
    parser = argparse.ArgumentParser(description="Sweep TinyStories experiment parameters and launch jobs.")

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="./slurm_logs")
    parser.add_argument("--num_jobs", type=int, default=10000)
    parser.add_argument("--start_job", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")

    args = parser.parse_args()

    sweep_configs = {
        "save_dir": [args.save_dir],
        "model_size": ["1B", "7B", "13B"],
        "sampling_model_id": list(range(3)),
        "n_sample": [1, 5, 10, 100, 200, 500, 1000],
        "sampling_seed": list(range(10)),
        "temperature": [1.0],
        "max_tokens": [32],
    }

    param_names = list(sweep_configs.keys())
    param_names.reverse()
    param_values = [sweep_configs[name] for name in param_names]
    param_combinations = list(itertools.product(*param_values))

    log_dir = os.path.join(args.log_dir, args.save_dir)

    if not args.dry_run:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    for i, params in enumerate(param_combinations[args.start_job:args.start_job+args.num_jobs]):

        cmd = build_cmd(
            args=dict(zip(param_names, params)), 
            log_path=os.path.join(log_dir, f"job_{i}.out"),
            script="./scripts/observation/dolmino.py",
        )

        print(f"Launching experiment {i+1}/{len(param_combinations)}")
        print(f"Parameters: {dict(zip(param_names, params))}")
        print("Command:", cmd)
        if not args.dry_run:
            subprocess.run(cmd, shell=True,check=True)

if __name__ == "__main__":
    main()
