from kfp import dsl, components

train_component = components.load_component_from_file('train.yaml')


@dsl.pipeline(name='train', description='A simple Kubeflow pipeline running train.py')
def my_pipeline(
    code_uri: str,
    output_path: str,
):
    train_op = train_component(
        code_uri=code_uri,
        output_path=output_path,
    )
    train_op.set_gpu_limit(1)
    train_op.set_memory_request("16G")
    train_op.set_memory_limit("128G")
    train_op.add_node_selector_constraint(
        "cloud.google.com/gke-accelerator",
        "nvidia-tesla-t4",
    )
