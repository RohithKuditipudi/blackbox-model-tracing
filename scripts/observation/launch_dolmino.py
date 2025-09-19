import os
import argparse

from tracing.launch import launch_jobs

def main():
    parser = argparse.ArgumentParser(description="Sweep TinyStories experiment parameters and launch jobs.")

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--script", type=str, default="./scripts/observation/dolmino.py")
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

    launch_jobs(
        sweep_configs=sweep_configs, 
        script=args.script, 
        save_dir=args.save_dir, 
        log_dir=os.path.join(args.log_dir, args.save_dir), 
        num_jobs=args.num_jobs, 
        start_job=args.start_job, 
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
