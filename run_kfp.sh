make pipeline.yaml && conda run -n kfp python run_kfp.py --config configs/$1.yaml --output gs://demand-vision/temp/marc/runs/$1/$2
