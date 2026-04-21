import sys
import argparse
import yaml
import time
import os
from pathlib import Path
import kfp
import subprocess
from src.io_utils import conf_read, conf_write, conf_hash, json_read, exists


def get_component_output_uri(run_id: str, component_name: str, output_name: str):
    run_detail = client.get_run(run_id)
    manifest = yaml.safe_load(run_detail.pipeline_runtime.workflow_manifest)
    for node_id, node in manifest["status"]["nodes"].items():
        if "outputs" in node:
            if "artifacts" in node["outputs"]:
                for artifact in node["outputs"]["artifacts"]:
                    print(artifact)
    return None


if __name__ == '__main__':
    print("python", sys.version)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=1)
    args = parser.parse_args()

    cfg = json_read(os.path.expanduser('~/.config/kfp/client.json'))
    client = kfp.Client(**cfg)
    pipeline_hash = conf_hash(conf_read('pipeline.yaml'))[-4:]

    experiment_name = 'pytorch_mini'
    run_name = 'test'
    print(experiment_name, '/', run_name)

    cfg = conf_read('src/config.yaml')
    seeds = [cfg.get('seed', 0)] if args.n_seeds == 1 else range(args.n_seeds)
    for seed in seeds:
        cfg["seed"] = seed

        run = client.create_run_from_pipeline_package(
            pipeline_file='pipeline.yaml',
            run_name=run_name + '.' + pipeline_hash + '.' + str(seed),
            arguments={
            },
            experiment_name=experiment_name,
        )
        print(run)
