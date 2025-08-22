from typing import Any

def args_to_string(args : dict[str, Any]):
    args_string = ''
    for key in args:
        args_string += f"--{key}={args[key]} "
    return args_string.strip()

def build_cmd(args : dict[str, Any], cmd_prefix="nlprun -g 1 -d a6000"):
    cmd_args = args_to_string(args)
    cmd = f"{cmd_prefix} 'uv run ./scripts/observation/tinystories.py {cmd_args}'"

    return cmd