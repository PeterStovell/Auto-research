import argparse
import os
import sys
import tarfile
import tempfile

import fsspec
import kfp

from src.io_utils import conf_read, json_read

if __name__ == '__main__':
    print("python", sys.version)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--output', required=True, help='path to output folder on gcs')
    args = parser.parse_args()
    assert args.output.startswith("gs://demand-vision/temp/marc")  # for safety

    cfg = conf_read(args.config)
    experiment_name = "auto-research"
    run_name = os.path.basename(args.config).split('.')[0]

    # Package src/ and upload
    code_uri = f"{args.output}/src.tar.gz"
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tmp_path = tmp.name
    with tarfile.open(tmp_path, 'w:gz') as tf:
        tf.add('src', arcname='.')
    with open(tmp_path, 'rb') as f_in:
        with fsspec.open(code_uri, 'wb') as f_out:
            f_out.write(f_in.read())
    os.unlink(tmp_path)
    print(f"code     -> {code_uri}")

    # Upload config
    config_uri = f"{args.output}/config.yaml"
    with open(args.config, 'rb') as f_in:
        with fsspec.open(config_uri, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"config   -> {config_uri}")

    print(f"output   -> {args.output}")

    kfp_cfg = json_read(os.path.expanduser('~/.config/kfp/client.json'))
    client = kfp.Client(**kfp_cfg)

    run = client.create_run_from_pipeline_package(
        pipeline_file='pipeline.yaml',
        run_name=run_name,
        arguments={
            'code_uri': code_uri,
            'config_uri': config_uri,
            'output_path': args.output,
        },
        experiment_name=experiment_name,
    )
    print(run)
