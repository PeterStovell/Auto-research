import json
import argparse
import os
import sys


def json_read(path: str):
    with open(path) as f:
        return json.load(f)


if __name__ == '__main__':
    # print("python", sys.version)
    import kfp
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', required=True, nargs='+', help='KFP run ID(s)')
    args = parser.parse_args()

    kfp_cfg = json_read(os.path.expanduser('~/.config/kfp/client.json'))
    client = kfp.Client(**kfp_cfg)

    for run_id in args.run_id:
        run = client.get_run(run_id)
        print(json.dumps({"run_id": run_id, "status": run.run.status}))
