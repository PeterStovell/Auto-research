from kfp import dsl, components

pretrain_component = components.load_component_from_file('pretrain.yaml')
train_component = components.load_component_from_file('train.yaml')


@dsl.pipeline(name='pretrain-finetune', description='Pre-train on M5+Electricity, fine-tune on gas volume')
def my_pipeline(
    code_uri: str,
    pretrain_data_uri: str,
    pretrain_output_path: str,
    pretrain_backbone_uri: str,
    output_path: str,
):
    pretrain_op = pretrain_component(
        code_uri=code_uri,
        pretrain_data_uri=pretrain_data_uri,
        output_path=pretrain_output_path,
    )
    pretrain_op.set_gpu_limit(1)
    pretrain_op.set_memory_request('16G')
    pretrain_op.set_memory_limit('128G')
    pretrain_op.add_node_selector_constraint(
        'cloud.google.com/gke-accelerator', 'nvidia-tesla-t4',
    )

    train_op = train_component(
        code_uri=code_uri,
        output_path=output_path,
        pretrain_backbone_uri=pretrain_backbone_uri,
    )
    train_op.set_gpu_limit(1)
    train_op.set_memory_request('16G')
    train_op.set_memory_limit('128G')
    train_op.add_node_selector_constraint(
        'cloud.google.com/gke-accelerator', 'nvidia-tesla-t4',
    )
    train_op.after(pretrain_op)
