#!/usr/bin/env bash

startidx=$1
stride=$2
meta_file=$3  # refcoco or jsonline format
image_prefix=$4

outpath=$(dirname ${meta_file})/$(basename -s .json ${meta_file})/${startidx}-$((${startidx}+${stride}))".json"

CUDA_VISIBLE_DEVICES=0
python3 -m llava.eval.run_llava_dsp \
   --model-path "LLAVA/weights/llava_v1.5_13b/" \
   --meta-file ${meta_file} \
   --image-prefix ${img_prefix} \
   --outpath ${outpath} \
   --hint \
   --repeat 1 \
   --startidx ${startidx} \
   --stride ${stride}
