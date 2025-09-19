from typing import Any
import itertools
import os
import subprocess

def args_to_string(args : dict[str, Any]):
    args_string = ''
    for key in args:
        args_string += f"--{key}={args[key]} "
    return args_string.strip()

def build_cmd(
    args : dict[str, Any], 
    slurm_prefix="nlprun -g 1 -d a6000",
    run_prefix=".venv/bin/python",
    script="./scripts/observation/tinystories.py",
    log_path=None,
):
    cmd_args = args_to_string(args)
    log_str = f"-o {log_path} " if log_path is not None else ""

    cmd = f"{slurm_prefix} {log_str}'{run_prefix} {script} {cmd_args}'"

    return cmd


def launch_jobs(sweep_configs, script, save_dir, log_dir, num_jobs, start_job, dry_run):
    param_names = list(sweep_configs.keys())
    param_names.reverse()
    param_values = [sweep_configs[name] for name in param_names]
    param_combinations = list(itertools.product(*param_values))

    if not dry_run:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    for i, params in enumerate(param_combinations):
        if i < start_job or i >= start_job + num_jobs:
            continue

        cmd = build_cmd(
            args=dict(zip(param_names, params)), 
            log_path=os.path.join(log_dir, f"job_{i}.out"),
            script=script,
        )

        print(f"Launching experiment {i+1}/{len(param_combinations)}")
        print(f"Parameters: {dict(zip(param_names, params))}")
        print("Command:", cmd)
        if not dry_run:
            subprocess.run(cmd, shell=True,check=True)