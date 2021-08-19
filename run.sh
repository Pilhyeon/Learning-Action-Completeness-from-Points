model_path="./models/LACP"
output_path="./outputs/LACP"
log_path="./logs/LACP" 
seed=0

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed}