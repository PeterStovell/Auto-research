set -a && source .env && set +a
GCS_USER=$(echo "$KUBEFLOW_USERNAME" | cut -d@ -f1)
make pipeline.yaml && conda run -n kfp python run_kfp.py --output gs://demand-vision/temp/${GCS_USER}/runs/$1
