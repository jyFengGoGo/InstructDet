#!/usr/bin/env bash

start_idx=$1
stride=$2
sample_path=$3

out_path=$(dirname ${sample_path})/$(basename -s .json ${sample_path})/${start_idx}-$((${start_idx}+${stride}))".json"
echo ${sample_path}
echo "save to ->"${out_path}

prompt="./data/llama_prompts/level_instruct/prompt/instruct.txt"
seed_path="./data/llama_prompts/level_instruct/seed/v0.txt"

python3 -m fastchat.serve.ginstruct --model-path llama_data/vicuna-13b-v1.3/ --gpus 1 --max-new-tokens 1024 \
     --task_des ${prompt} \
     --seed_path ${seed_path} \
     --sample_path ${sample_path} \
     --out_path ${out_path} \
     --task level_instruct \
     --temperature 0.7 \
     --start_idx ${start_idx} \
     --stride ${stride}
