import subprocess
import signal
from functools import wraps

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def flatten_list(list_of_lists):
    flat_list = [item for sublist in list_of_lists for item in sublist]
    if isinstance(flat_list[0], list):
        return flatten_list(flat_list)
    else:
        return flat_list

def timeout(seconds, return_on_timeout=lambda: []):
    """
    Decorator: allow fn to run up to `seconds` (float), else return None.
    Unix-only, main thread.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            # 1) install our handler
            def _handler(signum, frame):
                raise TimeoutError

            old_handler = signal.signal(signal.SIGALRM, _handler)
            # 2) arm fractional‐second timer
            old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)

            try:
                return fn(*args, **kwargs)
            except TimeoutError:
                return return_on_timeout()
            finally:
                # 3) disarm and restore
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
                    signal.setitimer(signal.ITIMER_REAL, *old_timer)
                except Exception as e:
                    print(f"Error: {e}")
                    pass
        return wrapped
    return decorator