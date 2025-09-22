import os
import argparse

from tracing.launch import launch_jobs

def main():
    parser = argparse.ArgumentParser(description="Sweep TinyStories experiment parameters and launch jobs.")

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--script", type=str, default="./scripts/query/tinystories_distillation.py")
    parser.add_argument("--log_dir", type=str, default="./slurm_logs/distillation")
    parser.add_argument("--num_jobs", type=int, default=10000)
    parser.add_argument("--start_job", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")

    args = parser.parse_args()

    sweep_configs = {
        "save_dir": [args.save_dir],
        "n_teacher": [100000],
        "n_student": [100000],
        "n_distill": [100000],
        "num_teacher_checkpoints": [10],
        "num_student_checkpoints": [10],
        "num_distillation_checkpoints": [10],
        "teacher_checkpoint_idx": [9],
        "student_checkpoint_idx": [0, 9],
        "distillation_checkpoint_idx": list(range(10))[::-1],
        "n_test": [100000],
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
