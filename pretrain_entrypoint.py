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
parser.add_argument('--pretrain_data_uri', required=True, help='GCS folder containing m5_daily.parquet and electricity_daily.parquet')
parser.add_argument('--output_path', required=True)
args = parser.parse_args()

os.makedirs('/workspace', exist_ok=True)

# Download code
with fsspec.open(args.code_uri, 'rb') as f:
    with tarfile.open(fileobj=f) as tf:
        tf.extractall('/workspace')

os.chdir('/workspace')
os.makedirs(LOCAL_OUTPUT, exist_ok=True)
os.makedirs(LOCAL_PRETRAIN_DATA, exist_ok=True)

# Download pretrain parquets from GCS
print(f"Downloading pretrain data from {args.pretrain_data_uri} ...")
fs, _ = fsspec.url_to_fs(args.pretrain_data_uri)
for fname in ['m5_daily.parquet', 'electricity_daily.parquet']:
    remote = f"{args.pretrain_data_uri.rstrip('/')}/{fname}"
    local = os.path.join(LOCAL_PRETRAIN_DATA, fname)
    print(f"  {remote} -> {local}")
    fs.get(remote, local)

# Run pre-training
log_path = os.path.join(LOCAL_OUTPUT, 'pretrain.log')
with open(log_path, 'w') as log_file:
    process = subprocess.Popen(
        [sys.executable, 'train_pretrain.py',
         '--output', LOCAL_OUTPUT,
         '--data_dir', LOCAL_PRETRAIN_DATA],
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
