import sys
import os
import argparse
import tarfile
import tempfile
import kfp
import fsspec
from src.io_utils import conf_read, json_read


if __name__ == '__main__':
    print("python", sys.version)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to experiment config yaml')
    args = parser.parse_args()

    cfg = conf_read(args.config)
    experiment_name = cfg.experiment_name
    run_name = cfg.run_name
    gcs_root = cfg.gcs_root
    run_root = f"{gcs_root}/{experiment_name}/{run_name}"

    # Package src/ and upload
    code_uri = f"{run_root}/src.tar.gz"
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
    config_uri = f"{run_root}/config.yaml"
    with open(args.config, 'rb') as f_in:
        with fsspec.open(config_uri, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"config   -> {config_uri}")

    output_path = f"{run_root}/output"
    print(f"output   -> {output_path}")

    kfp_cfg = json_read(os.path.expanduser('~/.config/kfp/client.json'))
    client = kfp.Client(**kfp_cfg)

    run = client.create_run_from_pipeline_package(
        pipeline_file='pipeline.yaml',
        run_name=run_name,
        arguments={
            'code_uri': code_uri,
            'config_uri': config_uri,
            'output_path': output_path,
        },
        experiment_name=experiment_name,
    )
    print(run)
