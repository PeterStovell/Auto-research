import yaml
import os
import json
import kfp
from glob import glob
import sys

with open(os.path.expanduser('~/.config/kfp/client.json')) as f:
    cfg = json.load(f)

client = kfp.Client(**cfg)

experiment_name = sys.argv[1]
output_path = sys.argv[2]

experiment_id = client.get_experiment(experiment_name=experiment_name).id
runs = client.list_runs(
    experiment_id=experiment_id,
    sort_by = "created_at desc",
    page_size=200,
)
for run in runs.runs:
    if run.storage_state == 'STORAGESTATE_ARCHIVED':
        continue
    run_detail = client.get_run(run.id)
    manifest = yaml.safe_load(run_detail.pipeline_runtime.workflow_manifest)
    if not manifest['status']['finishedAt']:
        continue
    print(run.name)
    bucket = manifest['status']['artifactRepositoryRef']['artifactRepository']['s3']['bucket']
    for task_name, task in manifest["status"]["nodes"].items():
        if "outputs" in task and "artifacts" in task["outputs"]:
            for artifact in task["outputs"]["artifacts"]:
                if artifact['name'] not in {'train-output_path', 'main-logs'}:
                    continue
                if 's3' in artifact:
                    key = artifact['s3']['key']
                    uri = f"gs://{bucket}/{key}"
                    folder = os.path.join(output_path, run.name, task_name, artifact['name'])
                    folder = folder.replace('@', '.')
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                        os.system(f'gsutil cp {uri} {folder}')

for path in glob(f'{output_path}/*/*/*/*.tgz'):
    out = os.path.dirname(path)
    os.system(f'tar xvf {path} -C {out}')
