#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1
python3 -m llava.eval.run_llava \
   --model-path "LLAVA/weights/llava_v1.5_13b" \
   --image-file "images/000000000139.jpg" \
   --query "Provide a detailed description of the given image."
