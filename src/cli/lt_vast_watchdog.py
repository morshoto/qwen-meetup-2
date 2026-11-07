"""Destroy one explicitly named Vast instance after a bounded delay."""
import argparse
import json
import subprocess
import time


def destroy_and_verify(instance_id, *, attempts=10):
    if type(instance_id) is not int or instance_id <= 0 or type(attempts) is not int or attempts <= 0:
        raise ValueError('invalid watchdog arguments')
    for attempt in range(attempts):
        try:
            deleted = subprocess.run(['vastai', 'destroy', 'instance', str(instance_id), '--yes', '--raw'],
                                     capture_output=True, text=True, timeout=30)
            if deleted.returncode == 0:
                result = subprocess.run(['vastai', 'show', 'instances', '--raw'], capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    instances = json.loads(result.stdout)
                    if (isinstance(instances, list) and all(isinstance(x, dict) and 'id' in x for x in instances)
                            and all(str(x['id']) != str(instance_id) for x in instances)):
                        return True
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        if attempt + 1 < attempts:
            time.sleep(15)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instance_id', type=int)
    parser.add_argument('--after-seconds', type=int, required=True)
    args = parser.parse_args()
    if args.instance_id <= 0 or args.after_seconds <= 0:
        parser.error('instance ID and delay must be positive')
    print(f'watchdog armed for instance {args.instance_id}, delay {args.after_seconds}s', flush=True)
    time.sleep(args.after_seconds)
    if destroy_and_verify(args.instance_id):
        print('instance absent from Vast listing', flush=True)
    else:
        parser.exit(1, 'watchdog could not verify teardown; inspect Vast immediately\n')


if __name__ == '__main__':
    main()
