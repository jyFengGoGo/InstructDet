#!/usr/bin/env bash

start_idx=$1
stride=$2
sample_path=$3

out_path=$(dirname ${sample_path})/$(basename -s .json ${sample_path})/${start_idx}-$((${start_idx}+${stride}))".json"
echo ${sample_path}
echo "save to ->"${out_path}

prompt="./data/llama_prompts/multi_tgt/prompt/instruct.txt"
seed_path="./data/llama_prompts/multi_tgt/seed/v0/"

python3 -m fastchat.serve.ginstruct --model-path llama_data/vicuna-13b-v1.3/ --gpus 1 --max-new-tokens 1024 \
     --task_des ${prompt} \
     --seed_path ${seed_path} \
     --sample_path ${sample_path} \
     --out_path ${out_path} \
     --task multi_tgt \
     --temperature 0.5 \
     --start_idx ${start_idx} \
     --stride ${stride}
