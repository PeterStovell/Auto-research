# Pytorch

This is a PyTorch prototype for DV. 

## Environments

We need an environment for local runs and an environment for kubeflow runs.
This is because our kubeflow cluster is running an old version that is clashing with
more recent libraries needed for the model.

    conda env create -f conda/kfp.yml
    conda env create -f conda/pytorch.yml
    conda activate pytorch

Provide a kfp client configuration here:

    ~/.config/kfp/client.json

e.g:

    {
        "host": "https://kubeflow18.endpoints.dev-stovell-ai.cloud.goog/pipeline", 
        "client_id": "811609456607-61mmvhq0o7vq4rgu25hkl25g7bulug1t.apps.googleusercontent.com", 
        "other_client_id": "811609456607-i25dvmvcr9ousd48mp03fsrnhsq0ajdc.apps.googleusercontent.com", 
        "other_client_secret": "...", 
        "namespace": "marc"
    }


## Usage

Local runs:

    python src/train.py --output_path out/local

Kubeflow runs:

    make pipeline.yaml
    conda run -n kfp python run.py
