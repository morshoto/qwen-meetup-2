"""Run the reload probe under independent time, memory and process limits."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.cli.qwen38_canary import resource_limit, resources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    command = [sys.executable, '-u', '-m', 'src.cli.qwen38_canary_verify', '--model-path', args.model_path, '--run-dir', args.run_dir]
    samples = []; reason = None; child = None; start = time.monotonic()
    with (root / 'probe.log').open('x') as log:
        try:
            current = resources(); reason = resource_limit(*current, 10, 16 * 1024**3)
            samples.append({'seconds': 0, 'free_percent': current[0], 'swap_bytes': current[1]})
            if not reason:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    elapsed = time.monotonic() - start
                    current = resources(); samples.append({'seconds': elapsed, 'free_percent': current[0], 'swap_bytes': current[1]})
                    reason = resource_limit(*current, 10, 16 * 1024**3)
                    if elapsed >= 900: reason = 'timeout'
                    if reason: break
                    time.sleep(5)
        finally:
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
    gone = True
    if child:
        try: os.killpg(child.pid, 0); gone = False
        except ProcessLookupError: pass
    result = {'command': command, 'exit_code': child.returncode if child else None, 'reason': reason,
              'process_group_gone': gone, 'elapsed_seconds': time.monotonic() - start, 'samples': samples}
    (root / 'probe-execution.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
