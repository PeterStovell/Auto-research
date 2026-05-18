import json
import argparse
import os
import sys
import tarfile
import tempfile
import fsspec
import kfp

from dex_auth import DexSessionManager


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


def json_read(path: str):
    with open(path) as f:
        return json.load(f)


def get_kfp_client() -> kfp.Client:
    endpoint = os.environ.get("KUBEFLOW_ENDPOINT", "").rstrip("/")
    if not endpoint:
        raise ValueError("KUBEFLOW_ENDPOINT is required in .env")
    username = os.environ.get("KUBEFLOW_USERNAME")
    password = os.environ.get("KUBEFLOW_PASSWORD")
    if not username or not password:
        raise ValueError("KUBEFLOW_USERNAME and KUBEFLOW_PASSWORD are required in .env")
    skip_tls = os.environ.get("KUBEFLOW_SKIP_TLS_VERIFY", "").lower() == "true"

    dex = DexSessionManager(
        endpoint_url=endpoint,
        dex_username=username,
        dex_password=password,
        dex_auth_type="local",
        skip_tls_verify=skip_tls,
    )
    session_cookies = dex.get_session_cookies()

    namespace = os.environ.get("KUBEFLOW_NAMESPACE")
    if not namespace:
        raise ValueError("KUBEFLOW_NAMESPACE is required in .env")

    kfp_cfg = json_read(os.path.expanduser('~/.config/kfp/client.json'))
    pipeline_host = endpoint if endpoint.endswith("/pipeline") else endpoint + "/pipeline"

    client = kfp.Client(
        host=pipeline_host,
        cookies=session_cookies,
        namespace=namespace,
    )

    if skip_tls:
        import ssl
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        client._run_api.api_client.rest_client.pool_manager = urllib3.PoolManager(
            num_pools=4,
            cert_reqs=ssl.CERT_NONE,
        )

    return client


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='path to output folder on GCS')
    parser.add_argument('--pretrain_path', default='gs://demand-vision/temp/peter/runs/pretrain_data',
                        help='GCS folder containing m5_daily.parquet and electricity_daily.parquet')
    args = parser.parse_args()
    gcs_user = os.environ.get("KUBEFLOW_USERNAME", "").split("@")[0]
    assert args.output.startswith(f"gs://demand-vision/temp/{gcs_user}/runs/"), \
        f"Output path must be under gs://demand-vision/temp/{gcs_user}/runs/"

    experiment_name = "auto-research-pretrain-finetune"
    run_name = os.path.basename(args.output)

    pretrain_output_path  = f"{args.output}/pretrain"
    pretrain_backbone_uri = f"{pretrain_output_path}/pretrain_backbone.pt"
    pretrain_data_uri     = args.pretrain_path

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

    client = get_kfp_client()

    run = client.create_run_from_pipeline_package(
        pipeline_file='pipeline.yaml',
        run_name=run_name,
        arguments={
            'code_uri':              code_uri,
            'pretrain_data_uri':     pretrain_data_uri,
            'pretrain_output_path':  pretrain_output_path,
            'pretrain_backbone_uri': pretrain_backbone_uri,
            'output_path':           args.output,
        },
        experiment_name=experiment_name,
    )
    print(f"run_id={run.run_id}")
