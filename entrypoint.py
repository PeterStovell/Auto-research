import argparse
import sys
import tarfile
import os
import subprocess
import fsspec

LOCAL_OUTPUT = '/workspace/output'

parser = argparse.ArgumentParser()
parser.add_argument('--code_uri', required=True)
parser.add_argument('--output_path', required=True)
args = parser.parse_args()

os.makedirs('/workspace', exist_ok=True)
with fsspec.open(args.code_uri, 'rb') as f:
    with tarfile.open(fileobj=f) as tf:
        tf.extractall('/workspace')

os.chdir('/workspace')
os.makedirs(LOCAL_OUTPUT, exist_ok=True)

log_path = os.path.join(LOCAL_OUTPUT, 'train.log')
with open(log_path, 'w') as log_file:
    process = subprocess.Popen(
        [sys.executable, 'train.py', '--output', LOCAL_OUTPUT],
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

# Upload local output directory to GCS
fs, _ = fsspec.url_to_fs(args.output_path)
fs.put(LOCAL_OUTPUT, args.output_path, recursive=True)
print(f"Output uploaded to {args.output_path}")
