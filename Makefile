full_image_name := gcr.io/dev-stovell-ai/pytorch-research/auto-research:latest

image : Dockerfile entrypoint.py
	docker build --platform linux/amd64 -t $(full_image_name) .
	docker push $(full_image_name)
	docker inspect --format="{{index .RepoDigests 0}}" $(full_image_name) > image

train.yaml: train_template.yaml image
	sed -e "s|IMAGE|$(shell cat image)|g" train_template.yaml > train.yaml

pretrain.yaml: pretrain_template.yaml image
	sed -e "s|IMAGE|$(shell cat image)|g" pretrain_template.yaml > pretrain.yaml

pipeline.yaml : train.yaml pretrain.yaml pipeline.py
	conda run -n kfp dsl-compile --py pipeline.py --output pipeline.yaml
