from typing import Any

def args_to_string(args : dict[str, Any]):
    args_string = ''
    for key in args:
        args_string += f"--{key}={args[key]} "
    return args_string.strip()

def build_cmd(args : dict[str, Any], cmd_prefix="nlprun -g 1 -d a6000", log_path=None):
    cmd_args = args_to_string(args)
    log_str = f"-o {log_path} " if log_path is not None else ""

    cmd = f"{cmd_prefix} {log_str}'uv run ./scripts/observation/tinystories.py {cmd_args}'"

    return cmd