from typing import Any

def args_to_string(args : dict[str, Any]):
    args_string = ''
    for key in args:
        args_string += f"--{key}={args[key]} "
    return args_string.strip()

def build_cmd(
    args : dict[str, Any], 
    slurm_prefix="nlprun -g 1 -d a6000",
    python_prefix="uv run python ./scripts/observation/tinystories.py",
    log_path=None,
):
    cmd_args = args_to_string(args)
    log_str = f"-o {log_path} " if log_path is not None else ""

    cmd = f"{slurm_prefix} {log_str}'{python_prefix} {cmd_args}'"

    return cmd