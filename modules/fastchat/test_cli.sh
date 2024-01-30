#!/usr/bin/env bash
python3 -m fastchat.serve.cli --model-path llama_data/vicuna-13b-v1.3/ --gpus 1 --max-new-tokens 1024 