import json
import argparse
import os
import sys
import kfp
from dex_auth import DexSessionManager
from run_kfp import get_kfp_client

def load_dotenv(path=".env"):
    from pathlib import Path
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
load_dotenv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', required=True, nargs='+', help='KFP run ID(s)')
    args = parser.parse_args()

    client = get_kfp_client()

    for run_id in args.run_id:
        run = client.get_run(run_id)
        print(json.dumps({"run_id": run_id, "status": run.run.status}))
