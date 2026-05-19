import argparse
import sys
import tarfile
import os
import subprocess
import fsspec

LOCAL_OUTPUT = '/workspace/output'
LOCAL_PRETRAIN_DATA = '/workspace/pretrain_data'

parser = argparse.ArgumentParser()
parser.add_argument('--code_uri', required=True)
parser.add_argument('--output_path', required=True)
parser.add_argument('--mode', default='finetune', choices=['finetune', 'pretrain'],
                    help='finetune: run train_switch.py; pretrain: run train_pretrain.py')
# finetune-only
parser.add_argument('--pretrain_backbone_uri', default='',
                    help='GCS URI of pretrain_backbone.pt (finetune mode only)')
# pretrain-only
parser.add_argument('--pretrain_data_uri', default='',
                    help='GCS folder with m5_daily.parquet and electricity_daily.parquet (pretrain mode only)')
args = parser.parse_args()

os.makedirs('/workspace', exist_ok=True)

# Download and extract src tarball
with fsspec.open(args.code_uri, 'rb') as f:
    with tarfile.open(fileobj=f) as tf:
        tf.extractall('/workspace')

os.chdir('/workspace')
os.makedirs(LOCAL_OUTPUT, exist_ok=True)

if args.mode == 'pretrain':
    # Download pretrain parquets from GCS
    assert args.pretrain_data_uri, '--pretrain_data_uri is required in pretrain mode'
    os.makedirs(LOCAL_PRETRAIN_DATA, exist_ok=True)
    for fname in ['m5_daily.parquet', 'electricity_daily.parquet']:
        remote = f"{args.pretrain_data_uri.rstrip('/')}/{fname}"
        local = os.path.join(LOCAL_PRETRAIN_DATA, fname)
        print(f"Downloading {remote} -> {local}")
        with fsspec.open(remote, 'rb') as src, open(local, 'wb') as dst:
            dst.write(src.read())

    cmd = [sys.executable, 'train_pretrain.py',
           '--output', LOCAL_OUTPUT,
           '--data_dir', LOCAL_PRETRAIN_DATA]

else:  # finetune
    # Optionally download pre-trained backbone
    local_backbone = None
    if args.pretrain_backbone_uri:
        local_backbone = '/workspace/pretrain_backbone.pt'
        print(f"Downloading backbone from {args.pretrain_backbone_uri} ...")
        fs, _ = fsspec.url_to_fs(args.pretrain_backbone_uri)
        fs.get(args.pretrain_backbone_uri, local_backbone)

    cmd = [sys.executable, 'train_switch.py', '--output', LOCAL_OUTPUT]
    if local_backbone:
        cmd += ['--pretrain_backbone', local_backbone]

log_path = os.path.join(LOCAL_OUTPUT, f'{args.mode}.log')
with open(log_path, 'w') as log_file:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        sys.stdout.write(line)
        log_file.write(line)
    process.wait()

if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, process.args)

# Upload output to GCS
fs_out, _ = fsspec.url_to_fs(args.output_path)
for filename in os.listdir(LOCAL_OUTPUT):
    local_file = os.path.join(LOCAL_OUTPUT, filename)
    remote_file = f"{args.output_path}/{filename}"
    fs_out.put(local_file, remote_file)
    print(f"Uploaded {filename} -> {remote_file}")
