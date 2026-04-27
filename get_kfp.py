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
    parser.add_argument('--run_id', required=True, help='KFP run ID')
    args = parser.parse_args()

    kfp_cfg = json_read(os.path.expanduser('~/.config/kfp/client.json'))
    client = kfp.Client(**kfp_cfg)

    run = client.get_run(args.run_id)
    print(f"status={run.run.status}")
